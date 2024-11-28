#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import traceback

import cv2 as cv
import sys
import numpy as np
import math
import rospy
import rosgraph

from sensor_msgs.msg import Image
from drone_internals.msg import MotorSpeed
from cv_bridge import CvBridge


class Control_Gui(QtWidgets.QMainWindow):

    send_velocity_signal = QtCore.pyqtSignal(list)
    send_delta_signal = QtCore.pyqtSignal(list)

    def __init__(self):

        super(Control_Gui,self).__init__()
        loadUi("/home/fizzer/ros_ws/src/drone_controller/nodes/drone_control.ui",self)

        self.setWindowTitle("Drone Control Panel")

        self.motor_speed_button.clicked.connect(self.SLOT_speed_button)
        self.motor_stop_button.clicked.connect(self.SLOT_stop_button)
        self.motor_delta_button.clicked.connect(self.SLOT_delta_button)

    
        if rosgraph.is_master_online():
            # Start the ROS Worker thread
            self.ros_worker = ROSWorker()
            self.ros_worker.image1_ready.connect(self.update_image1)  # Connect signal to update slot
            self.ros_worker.image2_ready.connect(self.update_image2)
            self.ros_worker.image3_ready.connect(self.update_image3)
            self.ros_worker.clue_img_ready.connect(self.update_clue_img)
            self.ros_worker.center_data_signal.connect(self.update_center_label)
            self.ros_worker.detection_data_signal.connect(self.update_detection_label)
            self.ros_worker.start()

            self.send_velocity_signal.connect(self.ros_worker.send_new_speed)
            self.send_delta_signal.connect(self.ros_worker.set_new_delta)

        else:
            rospy.logwarn("ROS master not running. Skipping ROS worker initialization.")


    def update_image1(self, q_image):
        self.cam_display.setPixmap(QtGui.QPixmap.fromImage(q_image))
        self.cam_display.setScaledContents(True)

    def update_image2(self, q_image):
        self.cam_display2.setPixmap(QtGui.QPixmap.fromImage(q_image))
        self.cam_display2.setScaledContents(True)

    def update_image3(self, q_image):
        self.cam_display_adjusted.setPixmap(QtGui.QPixmap.fromImage(q_image))
        self.cam_display_adjusted.setScaledContents(True)

    def update_clue_img(self,q_image):
        self.clue_cam.setPixmap(QtGui.QPixmap.fromImage(q_image))
        self.clue_cam.setScaledContents(True)

    def update_center_label(self,data):
        self.center_label.setText(f"X: {data[0]}, Y: {data[1]}")
        self.delta_label.setText(f"DX: {data[2]}, DY: {data[3]}")
        pitch_deg = -0.356 * data[3]
        self.angle_label.setText(f"Pitch: {pitch_deg:.2f}°")

    def update_detection_label(self,data):
        self.intersection_label.setText(f"Intersection: X: {data[0]}, Y: {data[1]}")
        self.pitch_offset_label.setText(f"Offset: X: {data[2]}, Y: {data[3]}")
        self.adjusted_intersection_label.setText(f"Adjusted Intersection: X: {data[0]+data[2]}, Y: {data[1]+data[3]}")
        self.line_angle_label.setText(f"Angle: {data[4]:.2f}")
        self.theta_label.setText(f"Theta: {data[5]*180/np.pi:.3f}")
        self.phi_label.setText(f"Phi: {data[6]*180/np.pi:.3f}")
        self.dir_label.setText(f"Dir: X: {data[7]:.3f}, Y: {data[8]:.3f}, Z: {data[9]:.3f}")

    def SLOT_speed_button(self):
        new_speed = self.motor_speed_input.value()
        back_delta = self.back_speed_input.value()
        velocities = [new_speed, -new_speed-back_delta, new_speed+back_delta, -new_speed]
        self.send_velocity_signal.emit(velocities)
    
    def SLOT_stop_button(self):
        velocities = [0.0, 0.0, 0.0, 0.0]
        self.send_velocity_signal.emit(velocities)

    def SLOT_delta_button(self):
        delta = self.motor_delta_input.value()
        self.send_delta_signal.emit([delta])
    




class ROSWorker(QtCore.QThread):
    image1_ready = QtCore.pyqtSignal(QtGui.QImage)
    image2_ready = QtCore.pyqtSignal(QtGui.QImage)
    image3_ready = QtCore.pyqtSignal(QtGui.QImage)
    clue_img_ready = QtCore.pyqtSignal(QtGui.QImage)
    center_data_signal = QtCore.pyqtSignal(list)
    detection_data_signal = QtCore.pyqtSignal(list)

    center_target = [64,64]
    center_delta = [0,0]

    pixel_delta = 0

    pitch_deg = 0.0


    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cam1_sub = rospy.Subscriber('/B1/camera1/image_raw', Image, self.cam1_callback)
        self.cam2_sub = rospy.Subscriber('/B1/camera2/image_raw', Image, self.cam2_callback)
        self.speed_pub = rospy.Publisher('/motor_speed_cmd', MotorSpeed, queue_size=1)
        self.speeds = [0.0,0.0,0.0,0.0]

    def run(self): 
        # rospy.spin()
        rate = rospy.Rate(20)  # rate in Hz
        while not rospy.is_shutdown():
            names = ["propeller1", "propeller2", "propeller3", "propeller4"]

            motor_delta = self.pixel_delta * self.center_delta[1]

            # rospy.loginfo(f"Motor Delta: {motor_delta}")

            adjusted_vel = [self.speeds[0] + motor_delta, self.speeds[1], self.speeds[2], self.speeds[3] - motor_delta]

            ms = MotorSpeed(name=names, velocity=adjusted_vel)
            self.speed_pub.publish(ms)
            # rospy.loginfo(f"PUBLISHING SPEED {self.speeds}")
            rate.sleep()


    # do all clue detection here
    def clue_detection(self,img):

        cv.circle(img,(100,100),20,(255,255,0),2)


        return img # return processed image

    def cam1_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image (cv::Mat)
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            cv_image = img.copy()

            #detect and mark lines

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

            lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=100)

            delta = 1
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

            for line in adjusted_lines:
                x1,y1,x2,y2,theta,length = line
                cv.line(cv_image,(x1,y1),(x2,y2),(255,0,0),4)
                cv.circle(cv_image,(x1,y1),4,(255,0,255),2)
                cv.circle(cv_image,(x2,y2),4,(0,255,255),2)

            x_int = 0
            y_int = 0
            angle = 0
            if len(adjusted_lines) >= 2:
                x1,y1,x2,y2,_,_ = adjusted_lines[0]
                x3,y3,x4,y4,_,_ = adjusted_lines[1]

                try:
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

                    cv.circle(cv_image,(x_int,y_int),20,(255,255,0),2)
                except Exception as e:
                    print(f"Error: x={x_int},y={y_int}")
                    x_int = 0
                    y_int = 0
                    angle = 0


            # for line in lines:
            #     x1,y1,x2,y2 = line[0]
            #     cv.line(cv_image,(x1,y1),(x2,y2),(255,0,0),4)
            #     cv.circle(cv_image,(x1,y1),4,(255,0,255),2)
            #     cv.circle(cv_image,(x2,y2),4,(0,255,255),2)
            
            # print(f"{len(lines)}, {len(adjusted_lines)}")


            img_final = cv_image

            # adjusted image

            height, width, channel = cv_image.shape



            offsetx = 0
            offsety = int(self.pitch_deg * 7.41) # from spreadsheet
            trans_mat = np.array([[1, 0, offsetx], [0, 1, offsety]], dtype=np.float32)
            adjusted = cv_image.copy()
            adjusted = cv.warpAffine(adjusted, trans_mat, (adjusted.shape[1], adjusted.shape[0]))

            img_final_adjusted = adjusted

            x_int_adj = x_int + offsetx
            y_int_adj = y_int + offsety # TODO will need to adjust these for roll

            # calculating direction to corner
            corner_pos = (6, -2.75, 0.5)

            horizontal_fov = 90 * np.pi / 180
            vertical_fov = 54 * np.pi / 180

            dx = x_int_adj - width / 2
            dy = y_int_adj - height / 2

            theta = dx * horizontal_fov / width
            phi = dy * vertical_fov / height

            corner_dir = [np.cos(phi)*np.cos(theta),np.cos(phi)*np.sin(theta),np.sin(phi)]



            
            #clue detection
            clue_img = self.clue_detection(img.copy())

            # Convert the OpenCV image to QImage (which can be displayed in Qt)
            height, width, channel = img_final.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(img_final.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            height, width, channel = img_final_adjusted.shape
            bytes_per_line = 3 * width
            q_image_adjusted = QtGui.QImage(img_final_adjusted.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

            height, width, channel = clue_img.shape
            bytes_per_line = 3 * width
            q_image_clue = QtGui.QImage(clue_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

            # emit signal with QImage
            self.image1_ready.emit(q_image)
            self.image3_ready.emit(q_image_adjusted)
            self.clue_img_ready.emit(q_image_clue)
            # emit data signal
            data = [x_int,y_int,offsetx,offsety,angle,theta,phi]
            data.extend(corner_dir)
            self.detection_data_signal.emit(data)
        except Exception as e:
            print(traceback.format_exc())

    def cam2_callback(self, msg):
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

            w = 128

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

            self.center_data_signal.emit([row_max,col_max,self.center_delta[0],self.center_delta[1]])
            # print(f"Center: {col_max}, {row_max}")

            self.pitch_deg = -0.356 * self.center_delta[1] # equation from spreadsheet

            img_final = mask_rgb
            # Convert the OpenCV image to QImage (which can be displayed in Qt)
            height, width, channel = img_final.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(img_final.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # Emit signal with QImage
            self.image2_ready.emit(q_image)
        except Exception as e:
            print(traceback.format_exc())

    def send_new_speed(self, data):
        self.speeds = data
        # names = ["propeller1", "propeller2", "propeller3", "propeller4"]
        # velocities = data
        # ms = MotorSpeed(name=names, velocity=velocities)
        # self.speed_pub.publish(ms)

    def set_new_delta(self,data):
        self.pixel_delta = data[0]
        # rospy.loginfo(f"Receving new delta {data[0]}")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    if rosgraph.is_master_online():
        rospy.init_node('qt_ros_node', anonymous=True)
    else:
        rospy.logwarn("ROS master not running. Skipping ROS node initialization.")

    gui = Control_Gui()
    gui.show()
    sys.exit(app.exec_())
