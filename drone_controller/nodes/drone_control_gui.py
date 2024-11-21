#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

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
            self.ros_worker.center_data_signal.connect(self.update_center_label)
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

    def update_center_label(self,data):
        self.center_label.setText(f"X: {data[0]}, Y: {data[1]}")
        self.delta_label.setText(f"DX: {data[2]}, DY: {data[3]}")

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
    center_data_signal = QtCore.pyqtSignal(list)

    center_target = [31,23]
    center_delta = [0,0]

    pixel_delta = 0


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

    def cam1_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image (cv::Mat)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Convert the OpenCV image to QImage (which can be displayed in Qt)
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(cv_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # Emit signal with QImage
            self.image1_ready.emit(q_image)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

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

            w = 64

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
                cv.circle(mask_rgb,(row_max,col_max),5,(255,0,0),2)


            cv.circle(mask_rgb,self.center_target,1,(255,0,255),2)

            self.center_delta = [row_max - self.center_target[0],col_max - self.center_target[1]]

            self.center_data_signal.emit([row_max,col_max,self.center_delta[0],self.center_delta[1]])


            # print(f"Center: {col_max}, {row_max}")

            img_final = mask_rgb
            # Convert the OpenCV image to QImage (which can be displayed in Qt)
            height, width, channel = img_final.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(img_final.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # Emit signal with QImage
            self.image2_ready.emit(q_image)
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

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
