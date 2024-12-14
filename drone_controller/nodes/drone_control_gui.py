#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from python_qt_binding import loadUi
import traceback
from functools import partial

import os
import cv2 as cv
import sys
import numpy as np
from numpy.linalg import solve, norm, det
import math
import rospy
import rosgraph

from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from drone_internals.msg import MotorSpeed
from cv_bridge import CvBridge

from homography import HomographyProcessor

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from python_qt_binding import loadUi
import traceback
from functools import partial

import os
import cv2 as cv
import sys
import numpy as np
from numpy.linalg import solve, norm, det
import math
import rospy
import rosgraph

from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from drone_internals.msg import MotorSpeed
from cv_bridge import CvBridge, CvBridgeError
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
from std_msgs.msg import String


model = None
sess = None
graph = None
bridge = CvBridge()

# Define the HSV range
lower_hsv = np.array([113, 113, 44])
upper_hsv = np.array([167, 255, 255])
character_set = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9',' ']

clue_type_to_location = {
    "SIZE": 1,
    "VICTIM": 2,
    "CRIME": 3,
    "TIME": 4,
    "PLACE": 5,
    "MOTIVE": 6,
    "WEAPON": 7,
    "BANDIT": 8
}

def load_nn_model(model_path):
    global model, sess, graph
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        model = load_model(model_path)

def preprocess_patch(patch):
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)  # Resize
    patch = np.expand_dims(patch, axis=(0, -1))  # Add batch and channel dimensions
    patch = patch / 255.0  # Normalize
    return patch

def predict_character(patch):
    global model, sess, graph
    patch = preprocess_patch(patch)
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        predictions = model.predict(patch)

    predicted_index = np.argmax(predictions)
    predicted_char = character_set[predicted_index]
    return predicted_char

def publish_clue(clue_location, clue_prediction):

    pub = rospy.Publisher('/score_tracker', String, queue_size=10)
    team_id = "teamsix"
    team_password = "password"

    # # Validate inputs
    # if not (-1 <= clue_location <= 8):
    #     rospy.logerr("Invalid clue_location. Must be between -1 and 8.")
    #     return
    # if not (clue_prediction.isupper() and ' ' not in clue_prediction):
    #     rospy.logerr("Invalid clue_prediction. Must be uppercase, no spaces.")
    #     return

    # Create and publish the message
    message = f"{team_id},{team_password},{clue_location},{clue_prediction}"
    pub.publish(message)
    rospy.loginfo(f"Published message: {message}")

def image_callback(msg):
    global model
    try:
        # Convert the ROS Image message to OpenCV format
        # frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        frame = cv.cvtColor(msg,cv.COLOR_RGB2BGR)

        # Step 1: Convert the frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 2: Apply the HSV threshold to get the mask
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

        # cv2.imshow("Thresholded Mask", mask)  # Thresholded mask (with white regions)

        # Step 3: Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: If contours are found, find the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Check if the approximated contour has 4 points (rectangular)
            if len(approx) == 4:
                # Extract the points of the rectangle
                points = approx.reshape(4, 2)

                # Sort points to ensure proper order: top-left, top-right, bottom-right, bottom-left
                rect = order_points(points)

                # Define the width and height of the desired rectangle
                width = 350
                height = 200
                dst_rect = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")

                # Compute the perspective transform matrix
                M = cv2.getPerspectiveTransform(rect, dst_rect)

                # Warp the perspective to get a top-down view
                transformed = cv2.warpPerspective(frame, M, (width, height))

                # Show the full transformed clue board
                # cv2.imshow("Full Transformed Clue Board", transformed)

                # Step 1: Convert the transformed frame to HSV
                hsv_frame2 = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)

                # Step 2: Apply the HSV threshold to get the mask
                mask2 = cv2.inRange(hsv_frame2, lower_hsv, upper_hsv)

                # Step 3: Invert the mask to get the areas that are not in the HSV range
                mask2 = cv2.bitwise_not(mask2)

                # Step 4: Find contours in the inverse mask
                contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Step 4: If contours are found, find the largest one
                if contours2:
                    largest_contour2 = max(contours2, key=cv2.contourArea)

                    # Approximate the contour to a polygon
                    epsilon2 = 0.02 * cv2.arcLength(largest_contour2, True)
                    approx2 = cv2.approxPolyDP(largest_contour2, epsilon2, True)

                    # Check if the approximated contour has 4 points (rectangular)
                    if len(approx2) == 4:
                        # Extract the points of the rectangle
                        points2 = approx2.reshape(4, 2)

                        # Sort points to ensure proper order: top-left, top-right, bottom-right, bottom-left
                        rect2 = order_points(points2)

                        # Compute the perspective transform matrix
                        M2 = cv2.getPerspectiveTransform(rect2, dst_rect)

                        # Warp the perspective to get a top-down view
                        transformed2 = cv2.warpPerspective(transformed, M2, (width, height))

                        # Show the full transformed clue board
                        # cv2.imshow("Ideal Transformed Clue Board", transformed2)

                        # Debug: Show the original mask and inverted mask
                        # cv2.imshow("Original Mask", mask)
                        # cv2.imshow("Inverted Mask", mask2)

                        cv2.waitKey(1)

                        # Define more specific points for the bottom rectangle and top-right rectangle
                        # Adjust these points based on your requirements
                        bottom_rect_top_left = (13, 110)  # Bottom rectangle starting point
                        bottom_rect_bottom_right = (338, 180)  # Bottom rectangle ending point
                        top_right_rect_top_left = (145, 5)  # Top-right rectangle starting point
                        top_right_rect_bottom_right = (335, 70)  # Top-right rectangle ending point

                        # Crop the bottom rectangle using the specific points
                        bottom_rect = transformed2[bottom_rect_top_left[1]:bottom_rect_bottom_right[1],
                                                bottom_rect_top_left[0]:bottom_rect_bottom_right[0]]
                        # cv2.imshow("Bottom Rectangle", bottom_rect)

                        # Crop the top-right rectangle using the specific points
                        top_right_rect = transformed2[top_right_rect_top_left[1]:top_right_rect_bottom_right[1],
                                                    top_right_rect_top_left[0]:top_right_rect_bottom_right[0]]
                        # cv2.imshow("Top Right Rectangle", top_right_rect)

                        # Now, cut both top and bottom rectangles into smaller images

                        # Define the x-offset and number of smaller images for top and bottom rectangles
                        top_x_offset = 0  # Starting point offset for top rectangle cuts
                        top_width = top_right_rect.shape[1] // 7  # Number of smaller images for top
                        bottom_x_offset = 0  # Starting point offset for bottom rectangle cuts
                        bottom_width = bottom_rect.shape[1] // 12  # Number of smaller images for bottom

                        clue_type_imgs = []
                        # Cut and show the top rectangle into 7 smaller images
                        for i in range(7):
                            top_img = top_right_rect[
                                :, top_x_offset + i*top_width : top_x_offset + (i+1)*top_width
                            ]
                            # cv2.imshow(f"top_img{i+1}", top_img)
                            clue_type_imgs.append(top_img)

                        clue_val_imgs = []
                        # Cut and show the bottom rectangle into 12 smaller images
                        for i in range(12):
                            bottom_img = bottom_rect[
                                :, bottom_x_offset + i*bottom_width : bottom_x_offset + (i+1)*bottom_width
                            ]
                            # cv2.imshow(f"bottom_img{i+1}", bottom_img)
                            clue_val_imgs.append(bottom_img)

                        clue_type = ''.join([predict_character(img) for img in clue_type_imgs]).replace(" ", "")
                        clue_val = ''.join([predict_character(img) for img in clue_val_imgs]).replace(" ", "")
                        clue_location = clue_type_to_location.get(clue_type, 1)

                        rospy.loginfo(f"Clue Type: {clue_type}")
                        rospy.loginfo(f"Clue Value: {clue_val}")

                        publish_clue(clue_location=clue_location, clue_prediction=clue_val)

    except CvBridgeError as e:
        rospy.logerr(f"Error converting image: {e}")

def order_points(pts):
    """
    Sort the points in clockwise order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

class Control_Gui(QtWidgets.QMainWindow):

    send_velocity_signal = QtCore.pyqtSignal(list)
    send_delta_signal = QtCore.pyqtSignal(list)
    send_pid_signal = QtCore.pyqtSignal(list)
    send_stage_signal = QtCore.pyqtSignal(int)
    send_pid_index_signal = QtCore.pyqtSignal(int)
    send_target_signal = QtCore.pyqtSignal(list)
    send_target_angles_signal = QtCore.pyqtSignal(list)
    send_start_signal = QtCore.pyqtSignal(int)
    send_stop_signal = QtCore.pyqtSignal(int)
    send_clue_signal = QtCore.pyqtSignal(int)

    def __init__(self):

        super(Control_Gui,self).__init__()
        loadUi("/home/fizzer/ros_ws/src/drone_controller/nodes/drone_control.ui",self)

        self.setWindowTitle("Drone Control Panel")

        self.motor_speed_button.clicked.connect(self.SLOT_speed_button)
        self.motor_stop_button.clicked.connect(self.SLOT_motor_stop_button)
        # self.motor_delta_button.clicked.connect(self.SLOT_delta_button)
        self.pid_button.clicked.connect(self.SLOT_pid_button)
        self.set_target_button.clicked.connect(self.SLOT_set_target_button)
        self.set_target_angles_button.clicked.connect(self.SLOT_set_target_angles)
        self.start_button.clicked.connect(self.SLOT_start_button)
        self.stop_button.clicked.connect(self.SLOT_stop_button)

        self.stage_box.currentIndexChanged.connect(self.SLOT_stage_box)
        self.pid_box.currentIndexChanged.connect(self.SLOT_pid_box)

    
        if rosgraph.is_master_online():
            # Start the ROS Worker thread
            self.ros_worker = ROSWorker()
            # connect image signals
            self.ros_worker.top_cam_img_signal.connect(partial(self.display_img,display=self.cam_top_display))
            self.ros_worker.clue_img_signal.connect(partial(self.display_img,display=self.clue_display))
            cam_displays = (self.cam0_display,self.cam1_display,self.cam2_display,self.cam3_display)
            # need to do this because pyQt needs the signals to be referenced as class attributes to bind
            cam_signals = (self.ros_worker.cam0_signal,self.ros_worker.cam1_signal,self.ros_worker.cam2_signal,self.ros_worker.cam3_signal)
            for i,signal in enumerate(cam_signals):
                signal.connect(partial(self.display_img,display=cam_displays[i]))
            # connect data signals
            self.ros_worker.top_cam_data_signal.connect(self.update_center_label)
            self.ros_worker.processing_data_signal.connect(self.update_processing_label)
            self.ros_worker.cam0_data.connect(self.update_cam0_data_label)
            self.ros_worker.cam1_data.connect(self.update_cam1_data_label)
            self.ros_worker.time_signal.connect(self.update_time_label)
            self.ros_worker.pid_input_signal.connect(self.update_pid_inputs)

            self.ros_worker.start()

            self.send_velocity_signal.connect(self.ros_worker.send_new_speed)
            self.send_delta_signal.connect(self.ros_worker.set_new_delta)
            self.send_pid_signal.connect(self.ros_worker.set_pid)
            self.send_stage_signal.connect(self.ros_worker.set_new_stage)
            self.send_pid_index_signal.connect(self.ros_worker.set_pid_index)
            self.send_target_signal.connect(self.ros_worker.set_new_target)
            self.send_target_angles_signal.connect(self.ros_worker.set_new_target_angles)
            self.send_start_signal.connect(self.ros_worker.start_signal)
            self.send_stop_signal.connect(self.ros_worker.stop_signal)

            self.send_clue_signal.connect(self.ros_worker.process_clue)

            self.ros_worker.set_pid_index(0)

        else:
            rospy.logwarn("ROS master not running. Skipping ROS worker initialization.")


    def display_img(self,q_image,display):
        display.setPixmap(QtGui.QPixmap.fromImage(q_image))
        display.setScaledContents(True)


    def update_center_label(self,data):
        self.center_label.setText(f"X: {data[0]}, Y: {data[1]}")
        self.delta_label.setText(f"DX: {data[2]}, DY: {data[3]}")
        self.pitch_label.setText(f"Pitch: {data[4]:.2f}°")
        self.roll_label.setText(f"Roll: {data[5]:.2f}°")

    def update_cam0_data_label(self,data):
        self.intersection_label.setText(f"Intersection: X: {data[0]}, Y: {data[1]}")
        self.pitch_offset_label.setText(f"Offset: X: {data[2]}, Y: {data[3]}")
        self.adjusted_intersection_label.setText(f"Adjusted Intersection: X: {data[0]+data[2]}, Y: {data[1]+data[3]}")
        self.line_angle_label.setText(f"Angle: {data[4]:.2f}")
        self.theta_label.setText(f"Theta: {data[5]*180/np.pi:.3f}")
        self.phi_label.setText(f"Phi: {data[6]*180/np.pi:.3f}")
        self.dir_label.setText(f"Dir: X: {data[7]:.3f}, Y: {data[8]:.3f}, Z: {data[9]:.3f}")


    def update_cam1_data_label(self,data):
        self.intersection_label_2.setText(f"Intersection: X: {data[0]}, Y: {data[1]}")
        self.pitch_offset_label_2.setText(f"Offset: X: {data[2]}, Y: {data[3]}")
        self.adjusted_intersection_label_2.setText(f"Adjusted Intersection: X: {data[0]+data[2]}, Y: {data[1]+data[3]}")
        self.line_angle_label_2.setText(f"Angle: {data[4]:.2f}")
        self.theta_label_2.setText(f"Theta: {data[5]*180/np.pi:.3f}")
        self.phi_label_2.setText(f"Phi: {data[6]*180/np.pi:.3f}")
        self.dir_label_2.setText(f"Dir: X: {data[7]:.3f}, Y: {data[8]:.3f}, Z: {data[9]:.3f}")

    def update_processing_label(self,data):
        self.label_Q1.setText(f"Q1: X: {data[0]:.2f}, Y: {data[1]:.2f}, Z: {data[2]:.2f}")
        self.label_Q2.setText(f"Q2: X: {data[3]:.2f}, Y: {data[4]:.2f}, Z: {data[5]:.2f}")
        self.label_Qavg.setText(f"Qavg: X: {data[6]:.2f}, Y: {data[7]:.2f}, Z: {data[8]:.2f}")
        self.target_pos_label.setText(f"Target Pos: X: {data[9]:.2f}, Y: {data[10]:.2f}, Z: {data[11]:.2f}")
        self.target_tilt_label_2.setText(f"Delta(Scaled): X: {data[12]*1000:.3f}, Y: {data[13]*1000:.3f}")
        self.target_tilt_label.setText(f"Target Tilt: Pitch: {data[14]:.3f}, Roll: {data[15]:.3f}")
        if data[16] is not None:
            pal = QPalette()
            if data[16]:
                pal.setColor(QPalette.Window, Qt.green)
            else:
                pal.setColor(QPalette.Window,Qt.red)
            self.at_height_widget.setPalette(pal)

    def update_time_label(self,data):
        self.frame_time_label.setText(f"Frame Time(ms): {(data[0] - data[1])*1000:.1f}")
        self.time_label.setText(f"Time: {data[0]:.2f}")

    def update_pid_inputs(self,data):
        self.hover_p_input.setValue(data[0])
        self.hover_i_input.setValue(data[1])
        self.hover_d_input.setValue(data[2])

    def SLOT_speed_button(self):
        new_speed = self.motor_speed_input.value()
        back_delta = self.back_speed_input.value()
        velocities = [new_speed, -new_speed-back_delta, new_speed+back_delta, -new_speed]
        self.send_velocity_signal.emit(velocities)
    
    def SLOT_motor_stop_button(self):
        # velocities = [0.0, 0.0, 0.0, 0.0]
        # self.send_velocity_signal.emit(velocities)
        self.send_clue_signal.emit(1)

    def SLOT_delta_button(self):
        delta = self.motor_delta_input.value()
        self.send_delta_signal.emit([delta])

    def SLOT_pid_button(self):
        self.send_pid_signal.emit([self.hover_p_input.value(),self.hover_i_input.value(),self.hover_d_input.value()])

    def SLOT_set_target_button(self):
        self.send_target_signal.emit(
            [self.target_x_input.value(),self.target_y_input.value(),self.target_z_input.value()])

    def SLOT_set_target_angles(self):
        self.send_target_angles_signal.emit([self.target_pitch_input.value(),self.target_roll_input.value()])

    def SLOT_start_button(self):
        self.send_start_signal.emit(1)

    def SLOT_stop_button(self):
        self.send_stop_signal.emit(1)
    
    def SLOT_stage_box(self,index):
        self.send_stage_signal.emit(index)

    def SLOT_pid_box(self,index):
        self.send_pid_index_signal.emit(index)



# HELPER FUNCTIONS

# convert an opencv image to a QImage
def cvToQImg(cv_image):
    height, width, channel = cv_image.shape
    bytes_per_line = 3 * width
    return QtGui.QImage(cv_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

# rotate a point around another ccw - angle in radians
def rotate(origin, point, angle):

    dx = point[0] - origin[0]
    dy = point[1] - origin[1]

    cosa = np.cos(angle)
    sina = np.sin(angle)

    qx = origin[0] + cosa * (dx) - sina * (dy)
    qy = origin[1] + sina * (dx) + cosa * (dy)
    return qx, qy

# find the closest intersection of 2 lines
# inputs are numpy arrays
# returns results as lists
def intersectLines(p1,d1,p2,d2):
    d3 = np.cross(d1,d2)

    A = np.array([d1, -d2, d3]).T
    b = p2 - p1

    if det(A) != 0: # A needs to be non singular (invertible)
        t = solve(A,b)

        Q1 = p1 + d1 * t[0]
        Q2 = p2 + d2 * t[1]
        Qavg = (Q1+Q2) / 2
        return Qavg.tolist(),Q1.tolist(),Q2.tolist()
    return None, None, None

# intersect a line of for p + d*t with a plane of form a=a0,
#   where a=x,y,z and axis is the corresponding index 0,1,2
# p and d are numpy arrays
# returns results as a list
def intersectLineAndPlane(p,d,a0,axis):
    t = (a0-p[axis]) / d[axis]
    intersection = p + d * t
    return intersection.tolist()

def clamp(val,low,high):
    return max(low, min(val, high))



class PIDControl:
    def __init__(self,Kp,Ki,Kd,dt,integral_initial=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_error = None
        self.integral = integral_initial
        self.int_init = integral_initial

    def compute(self,target,predicted_value,p=False):
        error = target - predicted_value
        if self.prev_error is None:
            self.prev_error = error
        P = self.Kp * error
        self.integral += error * self.dt
        I = self.Ki * self.integral
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative
        self.prev_error = error
        if p:
            print(f"E: {error}, P: {P}, I: {I}, D: {D}, int: {self.integral}")
        return P + I + D
    
    def setParams(self,params):
        self.Kp, self.Ki, self.Kd = params

    def getParams(self):
        return [self.Kp,self.Ki,self.Kd]

    def reset(self):
        self.integral = self.int_init
        self.prev_error = None


# IMAGE PROCESSING STAGES
STAGE_ORIGINAL = 0
STAGE_MASK1 = 1
STAGE_MASK2 = 2
STAGE_MASK3 = 3
STAGE_MASK = 4
STAGE_EDGES = 5
STAGE_LINES = 6
STAGE_ADJUSTED = 7



class ROSWorker(QtCore.QThread):

    # PROCESSING VARIABLES

    processing_data_signal = QtCore.pyqtSignal(list)

    # MISC VARIABLES
    current_stage = STAGE_ADJUSTED
    pid_index = 0
    update_rate_hz = 20
    update_period = 1.0 / update_rate_hz
    time_signal = QtCore.pyqtSignal(list)
    pid_input_signal = QtCore.pyqtSignal(list)
    running = False


    # CLUE DETECTION VARIABLES

    clue_img_signal = QtCore.pyqtSignal(QtGui.QImage)
    clue_img = None

    current_clue_cam = 0

    clue_processor = HomographyProcessor()


    # TOP CAMERA VARIABLES

    top_cam_img_signal = QtCore.pyqtSignal(QtGui.QImage)
    top_cam_data_signal = QtCore.pyqtSignal(list)

    center_target = [128,131]
    center_delta = [0,0]

    pixel_delta = 0

    pitch_deg = 0.0
    roll_deg = 0.0


    # SIDE CAMERA VARIABLES

    # camera positions:
    # 0 - front
    # 1 - back
    # 2 - left
    # 3 - right

    # pyQt needs the signals to be class attributes
    cam0_signal = QtCore.pyqtSignal(QtGui.QImage)
    cam1_signal = QtCore.pyqtSignal(QtGui.QImage)
    cam2_signal = QtCore.pyqtSignal(QtGui.QImage)
    cam3_signal = QtCore.pyqtSignal(QtGui.QImage)
    cam0_data = QtCore.pyqtSignal(list)
    cam1_data = QtCore.pyqtSignal(list)
    cam2_data = QtCore.pyqtSignal(list)
    cam3_data = QtCore.pyqtSignal(list)

    
    
    # normalized vector pointing to each corner, if detected
    # currently just represents which corner each camera has detected
    # TODO better way to determine which corner is detected
    corner_dirs = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]


    # for each camera:
    # pitch_mult,roll_mult,x_mult,y_mult, pitch_rotate
    # used to change sign of or ignore certain features for certain cameras
    cam_adjustments = ((1,-1,1,1,False),(-1,1,-1,-1,False),(-1,1,-1,-1,True),(1,1,1,1,True))

    # how much to scale the camera feeds by
    cam_scale = 0.5

    tilt_mult = 0.5

    tilt_add = 0.001


    # MOVEMENT VARIABLES

    # target_pos = [5.5,0.5,0.2]

    target_pitch = 0
    target_roll = 0

    predicted_pos = None

    # moving_axis = 1 # y axis is default foward direction

    # all motors at this speed keeps the drone at constant height if there is no initial momentum
    hover_speed = 70.689

    # hoverPID = PIDControl(0.1,0.002,0.4,update_period)
    hoverPID = PIDControl(0.1,0,0.4,update_period)
    # tiltPID = PIDControl(0.002,0,0.002,update_period)
    tilt_p,tilt_i,tilt_d = 0.0025, 0, 0.002
    # tilt_p,tilt_i,tilt_d = 0.0008, 0, 0.0008
    # tilt_p,tilt_i,tilt_d = tilt_mult, tilt_add, 0.002
    pitchPID = PIDControl(tilt_p,tilt_i,tilt_d,update_period)
    rollPID = PIDControl(tilt_p,tilt_i,tilt_d,update_period)
    # movePID = PIDControl(0,0,0,update_period)
    move_p,move_i,move_d = 0.95,0,1.2
    # move_p,move_i,move_d = 0.00125,0.00001,0.0049
    xPID = PIDControl(move_p,move_i,move_d,update_period)
    yPID = PIDControl(move_p,move_i,move_d,update_period)

    # pos_deadzone = 0.05 # set PID output to 0 if this close to target


    # current_assumption = 5.5
    using_assumption = False

    just_switched_mode = False

    prev_pos = None

    # how close the height should be to the target height before moving
    # height_delta = 0.1
    height_delta = 100 # disabled

    # for display
    del_x = 0
    del_y = 0



    # ROUTING VARIABLES

    # start at 5.5, 2.5, 0.125
    waypoints = [
        # (x,y,z), movement axis, assumption, assumption axis, clue cam
        # ((5.5,3.0,0.1),1,5.5,0),
        ((5.5,-0.8,0.2),1,5.5,0,0),
        ((5.5,-1.37,0.2),1,5.5,0,0),
        ((5.5,-1.37,0.4),2,-1.37,1,0),
        ((4.5,-1.37,0.4),0,-1.37,1,3),
        
    ]
    
    current_waypoint_index = 0

    # how close each axis needs to be to be considered at the waypoint
    waypoint_distance = 0.1

    # how many frames the drone needs to be near the waypoint before moving to the next
    waypoint_time = 0

    waypoint_counter = 0

    stopped = False

    wait_time = 5

    prev_waypoint_stop_time = 0


    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.speeds = [0.0,0.0,0.0,0.0]
        self.cam_img_signals = (self.cam0_signal,self.cam1_signal,self.cam2_signal,self.cam3_signal)
        self.cam_data_signals = (self.cam0_data,self.cam1_data,self.cam2_data,self.cam3_data)

        self.current_time = 0
        self.prev_update_time = 0

        self.set_waypoint()

        self.clock_sub = rospy.Subscriber('/clock',Clock,self.clock_callback)
        self.cam_top_sub = rospy.Subscriber('/B1/camera_top/image_raw', Image, self.top_cam_callback)
        self.cam0_sub = rospy.Subscriber('/B1/camera0/image_raw', Image, partial(self.cam_callback_generic,cam=0))
        self.cam1_sub = rospy.Subscriber('/B1/camera1/image_raw', Image, partial(self.cam_callback_generic,cam=1))
        self.cam2_sub = rospy.Subscriber('/B1/camera2/image_raw', Image, partial(self.cam_callback_generic,cam=2))
        self.cam3_sub = rospy.Subscriber('/B1/camera3/image_raw', Image, partial(self.cam_callback_generic,cam=3))
        self.speed_pub = rospy.Publisher('/motor_speed_cmd', MotorSpeed, queue_size=1)
        self.score_pub = rospy.Publisher('/score_tracker',String,queue_size=5)

        load_nn_model(os.path.expanduser('~/ros_ws/nn/model_biggest.keras'))

    start_time = 0

    def clock_callback(self,data):
        self.current_time = (data.clock.nsecs / 1000000000.0) + data.clock.secs

        # print(f"clock: {self.di}, {self.current_time}, {data.clock.secs}, {(data.clock.nsecs / 1000000000.0)}, {data.clock.nsecs}")

        if self.current_time - self.prev_update_time > self.update_period:
            self.time_signal.emit([self.current_time - self.start_time,self.prev_update_time - self.start_time])
            self.prev_update_time = self.current_time
            if self.running:
                self.update()

    detected_first_clue = False
    stopped_first_clue = False

    def update(self):


        if self.current_time - self.start_time > 9 and self.current_time - self.prev_waypoint_stop_time > self.wait_time and self.stopped:
            self.stopped = False
            self.set_waypoint
            self.process_clue(self.clue_img)

        if self.stopped:
            self.stop_motors()
            return

        if self.current_time - self.start_time > 3.5 and not self.stopped_first_clue:
            self.stopped_first_clue = True
            self.stopped = True
            self.stop_motors()

        if self.current_time - self.start_time > 8.5 and not self.detected_first_clue:
            self.stopped = False
            self.detected_first_clue = True
            self.process_clue(self.clue_img)
        

        # if self.current_time - self.start_time > 3.5 and not self.stopped_first_clue:
        #     self.process_clue(self.clue_img)
        #     self.stopped_first_clue = True


        self.processInputs()
   


        base_speed = self.hover_speed
        motor_delta_x = 0
        motor_delta_y = 0

        if self.predicted_pos is not None and self.prev_pos is not None:

            # all_close = True
            # for a, b in zip(self.predicted_pos, self.target_pos):
            #     if abs(a - b) > self.waypoint_distance:
            #         all_close = False

            all_close = abs(self.predicted_pos[self.moving_axis]-self.target_pos[self.moving_axis]) < self.waypoint_distance

            if all_close and self.current_waypoint_index != len(self.waypoints):

                print(f"Reached waypoint {self.current_waypoint_index}")
                self.prev_waypoint_stop_time = self.current_time
                if self.moving_axis != 2:
                    self.stopped = True
                    self.stop_motors()
                    self.process_clue(self.clue_img)
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.waypoints):
                    self.set_waypoint()
                return


            hover_amount = self.hoverPID.compute(self.target_pos[2],self.predicted_pos[2])

            # stronger stabilization power if above the target height
            if self.predicted_pos[2] > self.target_pos[2]:
                hover_amount *= 1.5

            base_speed += hover_amount

            # if self.moving_axis == 0:
            #     dx = self.target_pos[0] - self.predicted_pos[0]
            #     self.target_pitch = clamp(-0.5 * dx,-0.5,0.5)
            # elif self.moving_axis == 1:
            #     dy = self.target_pos[1] - self.predicted_pos[1]
            #     self.target_pitch = clamp(-0.5 * dy,-0.5,0.5)
            #     # print(f"tp: {self.target_pitch}")

            # self.target_roll = clamp(-self.xPID.compute(self.target_pos[0],self.predicted_pos[0]),-0.5,0.5)
            # self.target_pitch = clamp(-self.yPID.compute(self.target_pos[1],self.predicted_pos[1]),-0.5,0.5)

            at_height = np.abs(self.predicted_pos[2] - self.target_pos[2]) < self.height_delta

            self.target_pitch = 0
            self.target_roll = 0

            if at_height:

                if self.moving_axis == 0:
                    self.target_roll = clamp(-self.xPID.compute(self.target_pos[0],self.predicted_pos[0]),-0.5,0.5)
                    # if not self.using_assumption:
                    #     self.target_pitch = clamp(-self.yPID.compute(self.target_pos[1],self.predicted_pos[1]),-0.25,0.25)
                elif self.moving_axis == 1:
                    self.target_pitch = clamp(-self.yPID.compute(self.target_pos[1],self.predicted_pos[1]),-0.5,0.5)
                    # if not self.using_assumption:
                    #     self.target_roll = clamp(-self.xPID.compute(self.target_pos[0],self.predicted_pos[0]),-0.25,0.25)

            # if np.abs(self.target_pos[0] - self.predicted_pos[0]) < self.pos_deadzone:
            #     self.target_roll = 0
            #     print("X")
            # if np.abs(self.target_pos[1] - self.predicted_pos[1]) < self.pos_deadzone:
            #     self.target_pitch = 0
            #     print("Y")


            if self.moving_axis == 0:
                motor_delta_x = -self.rollPID.compute(self.target_roll,self.roll_deg)
                # if not self.using_assumption:
                #     motor_delta_y =  -self.pitchPID.compute(self.target_pitch,self.pitch_deg)
            elif self.moving_axis == 1:
                motor_delta_y =  -self.pitchPID.compute(self.target_pitch,self.pitch_deg)
                # if not self.using_assumption:
                #     motor_delta_x = -self.rollPID.compute(self.target_roll,self.roll_deg)


            # motor_delta_x = -self.rollPID.compute(self.target_roll,self.roll_deg)
            # motor_delta_y =  -self.pitchPID.compute(self.target_pitch,self.pitch_deg)


        # V2

            # motor_delta_x = self.xPID.compute(self.target_pos[0],self.predicted_pos[0])
            # motor_delta_y = self.yPID.compute(self.target_pos[1],self.predicted_pos[1])
            # # print(f"Y: Target: {self.target_pos[1]}, Pred: {self.predicted_pos[1]}, DY: {self.target_pos[1]-self.predicted_pos[1]}")

            # self.del_x = motor_delta_x
            # self.del_y = motor_delta_y

            # # if self.pitch_deg > 0.5:
            # #     motor_delta_y = 0

            # # if self.roll_deg > 0.5:
            # #     motor_delta_x = 0
            
            # motor_delta_x /= (1+np.abs(self.roll_deg*self.tilt_mult))
            # motor_delta_y /= (1+np.abs(self.pitch_deg*self.tilt_mult))
            
            # dx = self.target_pos[0] - self.predicted_pos[0]
            # dy = self.target_pos[1] - self.predicted_pos[1]

            # t1 = 1
            # t2 = 0.25

            # if self.moving_axis == 0:
            #     self.target_roll = clamp(-0.5 * dx,-t1,t1)
            #     self.target_pitch = clamp(-0.5 * dy,-t2,t2)
            # elif self.moving_axis == 1:
            #     self.target_pitch = clamp(-0.5 * dy,-t1,t1)
            #     self.target_roll = clamp(-0.5 * dx,-t2,t2)

            # motor_delta_x -= np.abs(self.roll_deg) * self.rollPID.compute(self.target_roll,self.roll_deg)
            # motor_delta_y -= np.abs(self.pitch_deg) * self.pitchPID.compute(self.target_pitch,self.pitch_deg)
            



        base_speed /= np.cos(np.deg2rad(self.roll_deg))
        base_speed /= np.cos(np.deg2rad(self.pitch_deg))


        # if self.speeds[0] == 0.0:
        #     motor_delta_x = 0
        #     motor_delta_y = 0

        # print(f"delx: {motor_delta_x}, dely: {motor_delta_y}")




        # adjusted_vel = [self.speeds[0] + motor_delta_x, self.speeds[1], self.speeds[2], self.speeds[3] - motor_delta_x]
        self.prev_pos = None if self.predicted_pos is None else self.predicted_pos

        adjusted_vel = [
            base_speed + motor_delta_y - motor_delta_x,
            -(base_speed - motor_delta_y - motor_delta_x),
            base_speed - motor_delta_y + motor_delta_x,
            -(base_speed + motor_delta_y + motor_delta_x)
            ]

        self.pub_vel(adjusted_vel)


    def pub_vel(self,velocities):
        names = ["propeller1", "propeller2", "propeller3", "propeller4"]
        ms = MotorSpeed(name=names, velocity=velocities)
        self.speed_pub.publish(ms)

    def run(self): 
        rospy.spin()

    # process data aggregated from cameras
    def processInputs(self):
        p1 = np.array([6,-2.75,0.5]) # corner of north wall and east wall
        p2 = np.array([6,2.75,0.5]) # corner of north wall and west wall
        d1 = np.array(self.corner_dirs[0])
        d2 = np.array(self.corner_dirs[1])

        l1_exists = not all(i==0 for i in d1)
        l2_exists = not all(i==0 for i in d2)

        assumption_value = self.current_assumption
        assumption_axis = self.assumption_axis

        previously_using_assumption = self.using_assumption

        self.using_assumption = False

        data = []
        if l1_exists and l2_exists:
            Qavg, Q1, Q2 = intersectLines(p1,d1,p2,d2)

            self.predicted_pos = Qavg
            if Qavg is not None:
                data.extend(Q1)
                data.extend(Q2)
                data.extend(Qavg)
                
        elif l1_exists:
            self.using_assumption = True
            intersection = intersectLineAndPlane(p1,d1,assumption_value,assumption_axis)
            self.predicted_pos = intersection
            data = [0,0,0,0,0,0]
            data.extend(intersection)
        elif l2_exists:
            self.using_assumption = True
            intersection = intersectLineAndPlane(p2,d2,assumption_value,assumption_axis)
            self.predicted_pos = intersection
            data = [0,0,0,0,0,0]
            data.extend(intersection)
        else:
            self.predicted_pos = None
            data = [0,0,0,0,0,0,0,0,0]

        if previously_using_assumption != self.using_assumption:
            # self.just_switched_mode = True
            self.xPID.reset()
            self.yPID.reset()
            # print(f"Switching Mode: Using Assumption {self.using_assumption}")

        data.extend(self.target_pos)
        data.extend([self.del_x,self.del_y,self.target_pitch,self.target_roll])

        if self.predicted_pos is not None:
            at_height = np.abs(self.predicted_pos[2] - self.target_pos[2]) < self.height_delta
            data.append(at_height)
        else:
            data.append(None)

        self.processing_data_signal.emit(data)




    def detect_corner(self,img,adjustments):

        pitch_mult,roll_mult,x_mult,y_mult,pitch_rotate = adjustments

        cv_image = img.copy()

        # first pass

        lowerH = 0
        upperH = 255
        lowerS = 0
        upperS = 255
        lowerV = 150
        upperV = 255

        lower_hsv = np.array([lowerH,lowerS,lowerV])
        upper_hsv = np.array([upperH,upperS,upperV])

        hsv = cv.cvtColor(cv_image, cv.COLOR_RGB2HSV)

        mask1 = cv.bitwise_not(cv.inRange(hsv, lower_hsv, upper_hsv))
        # mask1 = cv.inRange(hsv, lower_hsv, upper_hsv)

        # second pass

        lowerH = 100
        upperH = 255
        lowerS = 0
        upperS = 255
        lowerV = 150
        upperV = 255

        lower_hsv = np.array([lowerH,lowerS,lowerV])
        upper_hsv = np.array([upperH,upperS,upperV])

        hsv = cv.cvtColor(cv_image, cv.COLOR_RGB2HSV)

        mask2 = cv.inRange(hsv, lower_hsv, upper_hsv)

        # filter pixels above mask1

        mask3 = np.zeros_like(mask1, dtype=np.uint8)

        for col in range(mask1.shape[1]):
            # Find the row of the first non-zero pixel (lowest allowed pixel in this column)
            rows = np.where(mask1[:, col] > 0)[0]
            if rows.size > 0:
                # Get the lowest allowed pixel index
                lowest_pixel_row = rows[0]
                # Set all pixels strictly below this lowest pixel in the mask
                mask3[lowest_pixel_row + 1:, col] = 255



        # mask = cv.bitwise_and(cv.bitwise_not(mask2),mask3)
        mask = mask3
        # mask = mask1

        edges = cv.Canny(mask,100,200)

        lines = cv.HoughLinesP(edges,1,np.pi/180,int(100*self.cam_scale),minLineLength=int(100*self.cam_scale),maxLineGap=int(100*self.cam_scale))

        delta = 5
        adjusted_lines = []


        # if there are several colinear lines, pick the longest one
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                theta = np.arctan2(y2-y1,x2-x1) * 180/np.pi
                length = np.sqrt(((x2-x1)**2+(y2-y1)**2))

                in_list = False

                for i in range(len(adjusted_lines)):
                    _,_,_,_,t,l = adjusted_lines[i]
                    if np.abs(t-theta) < delta:
                        # print(f"t: {t}, theta: {theta}")
                        in_list = True
                        if length > l:
                            adjusted_lines[i] = [x1,y1,x2,y2,theta,length]

                if not in_list:
                    adjusted_lines.append([x1,y1,x2,y2,theta,length])

        line_thickness = max(int(4 * self.cam_scale),1)
        radius = max(int(4 * self.cam_scale),1)
        circle_thickness = max(int(2 * self.cam_scale),1)
        big_circle_thickness = max(int(20 * self.cam_scale),1)

        for line in adjusted_lines:
            x1,y1,x2,y2,theta,length = line
            cv.line(cv_image,(x1,y1),(x2,y2),(255,0,0),line_thickness)
            cv.circle(cv_image,(x1,y1),radius,(255,0,255),circle_thickness)
            cv.circle(cv_image,(x2,y2),radius,(0,255,255),circle_thickness)



        
        # adjusted image - corrected for pitch

        height, width, _ = cv_image.shape

        offsetx = 0
        offsety = 0
        if pitch_rotate:
            offsety = int(self.roll_deg * 7.41 * pitch_mult * self.cam_scale) # from spreadsheet
        else:
            offsety = int(self.pitch_deg * 7.41 * pitch_mult * self.cam_scale)


        # further processing if both wall edges are detected
        x_int = 0
        y_int = 0
        angle = 0
        theta = 0
        phi = 0
        x_int_adj = 0
        y_int_adj = 0
        corner_dir = [0,0,0]
        if len(adjusted_lines) == 2:
            x1,y1,x2,y2,_,_ = adjusted_lines[0]
            x3,y3,x4,y4,_,_ = adjusted_lines[1]

            if x1!=x2 and x4!=x3:
                try:
                    # find intersection point of the wall edges
                    m1 = (y2-y1)/(x2-x1)
                    m2 = (y4-y3)/(x4-x3)
                    b1 = y1 - m1 * x1
                    b2 = y3 - m2 * x3

                    x_int = int((b2-b1)/(m1-m2))
                    y_int = int(m1*x_int + b1)

                    dx1 = x2-x1
                    dy1 = y2-y1
                    dx2 = x4-x3
                    dy2 = y4-y3

                    costheta = (dx1*dx2 + dy1*dy2)/np.sqrt((dx1**2+dy1**2)*(dx2**2+dy2**2))
                    angle = np.arccos(costheta) * 180 / np.pi

                    

                    # calculate direction to corner
                    x_int_adj = x_int + offsetx

                    y_int_adj = y_int + offsety

                    if pitch_rotate:
                        x_int_adj,y_int_adj = rotate((width/2,height/2),(x_int_adj,y_int_adj),self.pitch_deg * -np.pi/180 * pitch_mult)
                    else:
                        x_int_adj,y_int_adj = rotate((width/2,height/2),(x_int_adj,y_int_adj),self.roll_deg * -np.pi/180 * roll_mult)


                    horizontal_fov = 90 * np.pi / 180
                    vertical_fov = 54 * np.pi / 180

                    dx = x_int_adj - width / 2
                    dy = y_int_adj - height / 2

                    theta = dx * horizontal_fov / width
                    phi = dy * vertical_fov / height

                    corner_dir = [-np.cos(phi)*np.sin(theta)*x_mult,-np.cos(phi)*np.cos(theta)*y_mult,-np.sin(phi)]
                except Exception as e:
                    # print(traceback.format_exc())
                    x_int = 0
                    y_int = 0
                    angle = 0
                    theta = 0
                    phi = 0
                    corner_dir = [0,0,0]


        adjusted = cv_image.copy()

        trans_mat = np.array([[1, 0, offsetx], [0, 1, offsety]], dtype=np.float32)
        adjusted = cv.warpAffine(adjusted, trans_mat, (adjusted.shape[1], adjusted.shape[0]))

        image_center = tuple(np.array(adjusted.shape[1::-1]) / 2)
        if pitch_rotate:
            rot_mat = cv.getRotationMatrix2D(image_center, self.pitch_deg * pitch_mult, 1.0)
            adjusted = cv.warpAffine(adjusted, rot_mat, adjusted.shape[1::-1], flags=cv.INTER_LINEAR)
        else:
            rot_mat = cv.getRotationMatrix2D(image_center, self.roll_deg * roll_mult, 1.0)
            adjusted = cv.warpAffine(adjusted, rot_mat, adjusted.shape[1::-1], flags=cv.INTER_LINEAR)

        if int(x_int_adj) != 0 or int(y_int_adj) !=0:
            cv.circle(adjusted,(int(x_int_adj),int(y_int_adj)),big_circle_thickness,(255,255,0),circle_thickness)
        

        if self.current_stage == STAGE_ORIGINAL:
            img_final = img
        elif self.current_stage == STAGE_MASK:
            img_final = cv.cvtColor(mask,cv.COLOR_GRAY2RGB)
        elif self.current_stage == STAGE_EDGES:
            img_final = cv.cvtColor(edges,cv.COLOR_GRAY2RGB)
        elif self.current_stage == STAGE_LINES:
            img_final = cv_image
        elif self.current_stage == STAGE_ADJUSTED:
            img_final = adjusted
        elif self.current_stage == STAGE_MASK1:
            img_final = cv.cvtColor(mask1,cv.COLOR_GRAY2RGB)
        elif self.current_stage == STAGE_MASK2:
            img_final = cv.cvtColor(mask2,cv.COLOR_GRAY2RGB)
        elif self.current_stage == STAGE_MASK3:
            img_final = cv.cvtColor(mask3,cv.COLOR_GRAY2RGB)
        else:
            img_final = img

        return [img_final,corner_dir,theta,phi,x_int,y_int,offsetx,offsety,angle]


    # cam is the cam number, from 0-3
    def cam_callback_generic(self,msg,cam):
        try:
            # Convert the ROS image message to an OpenCV image 
            img_full = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            cv_image = cv.resize(img_full, (0,0), fx=self.cam_scale, fy=self.cam_scale) # scale down camera images


            img_final,corner_dir,theta,phi,x_int,y_int,offsetx,offsety,angle = (
                self.detect_corner(cv_image,self.cam_adjustments[cam])
            )

            self.corner_dirs[cam] = corner_dir
            
            # save image for clue detection if correct camera
            if self.current_clue_cam == cam:
                self.clue_img = img_full.copy()

            # emit image signal
            self.cam_img_signals[cam].emit(cvToQImg(img_final))
            
            # emit data signal
            data = [x_int,y_int,offsetx,offsety,angle,theta,phi]
            data.extend(corner_dir)
            self.cam_data_signals[cam].emit(data)

        except Exception as e:
            print(traceback.format_exc())


    def top_cam_callback(self, msg):

        try:
            # Convert the ROS image message to an OpenCV image (cv::Mat)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') # 64x64

            # OpenCV image processing here
            lowerH = 0
            upperH = 255
            lowerS = 0
            upperS = 255
            lowerV = 245
            upperV = 255

            lower_hsv = np.array([lowerH,lowerS,lowerV])
            upper_hsv = np.array([upperH,upperS,upperV])

            hsv = cv.cvtColor(cv_image, cv.COLOR_RGB2HSV)

            mask = cv.inRange(hsv, lower_hsv, upper_hsv)

            w = 256
            fov = 20

            col_avg = 0
            col_num = 0
            for y in range(w):
                row_avg = 0
                row_num = 0
                for x in range(w):
                    if mask[y,x]>0:
                        row_avg += x
                        row_num += 1
                        # print(f"x={x}")
                if row_num > 0:
                    col_avg += y
                    col_num += 1
                    # print(f"{y}: {row_avg}, {row_avg / row_num}, {row_avg * y / row_num}")
            
            col_max = int(col_avg / max(col_num,1))
            # print(f"CMAX:{col_max}")
            row_max = 0
            row_max_num = 0
            for x in range(w):
                    if mask[col_max,x]>0:
                        row_max += x
                        row_max_num += 1
            row_max = int(row_max / max(row_max_num,1))
            mask_rgb = cv.cvtColor(mask,cv.COLOR_GRAY2RGB)

            cv.circle(mask_rgb,self.center_target,1,(255,0,255),2)
            
            
            if row_max != 0 or col_max != 0:
                cv.circle(mask_rgb,(row_max,col_max),35,(255,0,0),4)

                self.center_delta = [row_max - self.center_target[0],col_max - self.center_target[1]]

                self.pitch_deg = -self.center_delta[1] * fov / w
                self.roll_deg = -self.center_delta[0] * fov / w

            # print(f"Center: {col_max}, {row_max}")

            # self.pitch_deg = -0.356 * self.center_delta[1] # equation from spreadsheet
            # self.roll_deg = -0.356 * self.center_delta[0]


            self.top_cam_data_signal.emit([row_max,col_max,self.center_delta[0],self.center_delta[1],self.pitch_deg,self.roll_deg])

            img_final = mask_rgb
            
            # Emit signal with QImage
            self.top_cam_img_signal.emit(cvToQImg(img_final))

        except Exception as e:
            print(traceback.format_exc())



    def send_new_speed(self, data):
        self.speeds = data
        if data[0] == 0.0:
            self.pitchPID.reset()
            self.rollPID.reset()
            self.hoverPID.reset()

    def set_new_delta(self,data):
        self.pixel_delta = data[0]
        
    def set_pid(self,data):
        index = self.pid_index
        if index == 0:
            self.hoverPID.setParams(data)
        elif index == 1:
            self.pitchPID.setParams(data)
            self.rollPID.setParams(data)
            # self.tilt_mult = data[0]
            # self.tilt_add = data[1]
        elif index == 2:
            self.xPID.setParams(data)
            self.yPID.setParams(data)

    def set_pid_index(self,index):
        self.pid_index = index
        if index == 0:
            self.pid_input_signal.emit(self.hoverPID.getParams())
        elif index == 1:
            self.pid_input_signal.emit(self.pitchPID.getParams())
        elif index == 2:
            self.pid_input_signal.emit(self.xPID.getParams())


    def set_new_stage(self,data):
        self.current_stage = data

    def set_new_target(self,data):
        self.target_pos = data

    def set_new_target_angles(self,data):
        self.target_pitch, self.target_roll = data

    started = False

    def start_signal(self,data):
        self.stopped = False
        if not self.started:
            self.score_pub.publish('teamsix,password,0,NA') # start timer
            self.started = True
            self.start_time = self.current_time
        if not self.running:
            self.running = True
            # print("Starting")

    def stop_signal(self,data):
        if data != 0:
            print("Stopping")
            self.stop_motors()
            self.running = False
            self.current_waypoint_index = 0
            self.set_waypoint()
            self.score_pub.publish('teamsix,password,-1,NA') #stop timer

    def stop_motors(self):
        self.reset_PID()
        self.pub_vel([0,0,0,0])
        

    def reset_PID(self):
        self.pitchPID.reset()
        self.rollPID.reset()
        self.hoverPID.reset()
        self.xPID.reset()
        self.yPID.reset()

    def set_waypoint(self):
        self.target_pos, self.moving_axis, self.current_assumption, self.assumption_axis, self.current_clue_cam = self.waypoints[self.current_waypoint_index]

    def process_clue(self,data):
        print("CLUE")
        if self.clue_img is not None:
            print("YES")
            image_callback(self.clue_img)





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    if rosgraph.is_master_online():
        rospy.init_node('qt_ros_node', anonymous=True)
    else:
        rospy.logwarn("ROS master not running. Skipping ROS node initialization.")

    gui = Control_Gui()
    gui.show()
    sys.exit(app.exec_())
