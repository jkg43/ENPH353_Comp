#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import traceback
from functools import partial

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




# IMAGE PROCESSING STAGES
STAGE_ORIGINAL = 0
STAGE_MASK = 1
STAGE_EDGES = 2
STAGE_LINES = 3
STAGE_ADJUSTED = 4

# HELPER FUNCTIONS

# rotate a point around another ccw - angle in radians
def rotate(origin, point, angle):

    dx = point[0] - origin[0]
    dy = point[1] - origin[1]

    cosa = np.cos(angle)
    sina = np.sin(angle)

    qx = origin[0] + cosa * (dx) - sina * (dy)
    qy = origin[1] + sina * (dx) + cosa * (dy)
    return qx, qy


class Control_Gui(QtWidgets.QMainWindow):


    # CLUE DETECTION VARIABLES

    clue_img = None

    current_clue_cam = 0


    # TOP CAMERA VARIABLES


    center_target = [119,119]
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

    
    # normalized vector pointing to each corner, if detected
    # currently just represents which corner each camera has detected
    # TODO better way to determine which corner is detected
    corner_dirs = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]


    # for each camera:
    # pitch_mult,roll_mult,x_mult,y_mult, pitch_rotate
    # used to change sign of or ignore certain features for certain cameras
    cam_adjustments = ((2,-2,1,1,False),(-2,2,-1,-1,False),(-2,2,1,1,True),(2,2,1,1,True))

    # how much to scale the camera feeds by
    cam_scale = 0.5

    # MISC VARIABLES
    current_stage = STAGE_ADJUSTED


    def __init__(self):

        super(Control_Gui,self).__init__()
        loadUi("/home/fizzer/ros_ws/src/drone_controller/nodes/drone_control.ui",self)

        self.setWindowTitle("Drone Control Panel")

        self.motor_speed_button.clicked.connect(self.SLOT_speed_button)
        self.motor_stop_button.clicked.connect(self.SLOT_stop_button)
        self.motor_delta_button.clicked.connect(self.SLOT_delta_button)
        self.stage_box.currentIndexChanged.connect(self.SLOT_stage_box)

        self.cam_displays = (self.cam0_display,self.cam1_display,self.cam2_display,self.cam3_display)
        self.cam_data_funcs = (self.update_cam0_data_label,self.update_cam1_data_label,lambda *args: None,lambda *args: None)

    
        if rosgraph.is_master_online():
            self.bridge = CvBridge()
            self.clock_sub = rospy.Subscriber('/clock',Clock,self.clock_callback)
            self.cam_top_sub = rospy.Subscriber('/B1/camera_top/image_raw', Image, self.top_cam_callback)
            self.cam0_sub = rospy.Subscriber('/B1/camera0/image_raw', Image, partial(self.cam_callback_generic,cam=0))
            self.cam1_sub = rospy.Subscriber('/B1/camera1/image_raw', Image, partial(self.cam_callback_generic,cam=1))
            self.cam2_sub = rospy.Subscriber('/B1/camera2/image_raw', Image, partial(self.cam_callback_generic,cam=2))
            self.cam3_sub = rospy.Subscriber('/B1/camera3/image_raw', Image, partial(self.cam_callback_generic,cam=3))
            self.speed_pub = rospy.Publisher('/motor_speed_cmd', MotorSpeed, queue_size=1)
            self.speeds = [0.0,0.0,0.0,0.0]

            self.current_time = 0
            self.prev_update_time = 0
            self.update_period = 1.0 / 20

        else:
            rospy.logwarn("ROS master not running. Skipping ROS worker initialization.")

    # QTGUI FUNCTIONS

    def display_img(self,cv_image,display):
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(cv_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        display.setPixmap(QtGui.QPixmap.fromImage(q_image))
        display.setScaledContents(True)


    def update_center_label(self,data):
        self.center_label.setText(f"X: {data[0]}, Y: {data[1]}")
        self.delta_label.setText(f"DX: {data[2]}, DY: {data[3]}")
        pitch_deg = -0.356 * data[3]
        roll_deg = -0.356 * data[2]
        self.pitch_label.setText(f"Pitch: {pitch_deg:.2f}°")
        self.roll_label.setText(f"Roll: {roll_deg:.2f}°")

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

    def SLOT_speed_button(self):
        new_speed = self.motor_speed_input.value()
        back_delta = self.back_speed_input.value()
        self.speeds = [new_speed, -new_speed-back_delta, new_speed+back_delta, -new_speed]
    
    def SLOT_stop_button(self):
        self.speeds = [0.0, 0.0, 0.0, 0.0]

    def SLOT_delta_button(self):
        self.pixel_delta = self.motor_delta_input.value()

    def SLOT_stage_box(self,index):
        self.current_stage = index
    


    # ROS FUNCTIONS

    def clock_callback(self,data):
        self.current_time = data.clock.secs + data.clock.nsecs / 1000000000  
        if self.current_time - self.prev_update_time >= self.update_period:
            self.frame_time_label.setText(f"Frame Time(ms): {(self.current_time - self.prev_update_time)*1000:.1f}")
            self.time_label.setText(f"Time: {self.current_time:.2f}")
            self.prev_update_time = self.current_time
            self.update()



    def update(self):
        self.processInputs()

        names = ["propeller1", "propeller2", "propeller3", "propeller4"]

        motor_delta = self.pixel_delta * self.center_delta[1]

        # rospy.loginfo(f"Motor Delta: {motor_delta}")

        adjusted_vel = [self.speeds[0] + motor_delta, self.speeds[1], self.speeds[2], self.speeds[3] - motor_delta]

        ms = MotorSpeed(name=names, velocity=adjusted_vel)
        self.speed_pub.publish(ms)
        # rospy.loginfo(f"PUBLISHING SPEED {self.speeds}")

    # process data aggregated from cameras
    def processInputs(self):
        p1 = np.array([6,-2.75,0.5]) # corner of north wall and east wall
        p2 = np.array([6,2.75,0.5]) # corner of north wall and west wall
        d1 = np.array(self.corner_dirs[0])
        d2 = np.array(self.corner_dirs[1])

        d3 = np.cross(d1,d2)

        A = np.array([d1, -d2, d3]).T
        b = p2 - p1

        if det(A) != 0: # A needs to be non singular (invertible)
            t = solve(A,b)

            Q1 = p1 + d1 * t[0]
            Q2 = p2 + d2 * t[1]
            Qavg = (Q1+Q2) / 2

            data = []
            data.extend(Q1.tolist())
            data.extend(Q2.tolist())
            data.extend(Qavg.tolist())
            self.update_processing_label(data)

        # clue detection
        if self.clue_img is not None:
            clue_processed = self.clue_detection(self.clue_img)
            self.display_img(clue_processed,self.clue_display)


    # do all clue detection here
    def clue_detection(self,img):

        cv.circle(img,(100,100),20,(255,255,0),2)


        return img # return processed image



    def detect_corner(self,img,adjustments):

        pitch_mult,roll_mult,x_mult,y_mult,pitch_rotate = adjustments

        cv_image = img.copy()

        lowerH = 100
        upperH = 255
        lowerS = 0
        upperS = 255
        lowerV = 150
        upperV = 255

        lower_hsv = np.array([lowerH,lowerS,lowerV])
        upper_hsv = np.array([upperH,upperS,upperV])

        hsv = cv.cvtColor(cv_image, cv.COLOR_RGB2HSV)

        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
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

        offsetx = 0 # TODO will need to adjust these for roll
        offsety = int(self.pitch_deg * 7.41 * pitch_mult * self.cam_scale) # from spreadsheet


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

            # display camera image
            self.display_img(img_final,self.cam_displays[cam])
            
            # show cam data
            data = [x_int,y_int,offsetx,offsety,angle,theta,phi]
            data.extend(corner_dir)
            self.cam_data_funcs[cam](data)

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
            fov = 30 * np.pi / 180

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

            if row_max != 0 or col_max != 0:
                cv.circle(mask_rgb,(row_max,col_max),10,(255,0,0),2)
            cv.circle(mask_rgb,self.center_target,1,(255,0,255),2)

            self.center_delta = [row_max - self.center_target[0],col_max - self.center_target[1]]

            self.update_center_label([row_max,col_max,self.center_delta[0],self.center_delta[1]])
            # print(f"Center: {col_max}, {row_max}")

            # self.pitch_deg = -0.356 * self.center_delta[1] # equation from spreadsheet
            # self.roll_deg = -0.356 * self.center_delta[0]

            self.pitch_deg = -self.center_delta[1] * fov / w
            self.roll_deg = -self.center_delta[0] * fov / w

            img_final = mask_rgb
            
            # show top cam
            self.display_img(img_final,self.cam_top_display)

        except Exception as e:
            print(traceback.format_exc())





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    if rosgraph.is_master_online():
        rospy.init_node('qt_ros_node', anonymous=True)
    else:
        rospy.logwarn("ROS master not running. Skipping ROS node initialization.")

    gui = Control_Gui()
    gui.show()
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(traceback.format_exc())
