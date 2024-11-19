#!/usr/bin/env python3
import csv
import math
import numpy as np
import rospy
import random

from drone_internals.msg import MotorSpeed
from std_msgs.msg import Float32



class DroneController():

    def __init__(self):
        self.speed_pub = rospy.Publisher('/motor_speed_cmd',MotorSpeed)
        self.sub = rospy.Subscriber('/drone_speed_control',Float32,self.drone_speed_callback)
        self.speed = 0
        rospy.loginfo("STARTING DRONE CONTROLLER")

    def pub(self):
        rate = rospy.Rate(10)  # 10 Hz (publishes 10 times per second)
        while not rospy.is_shutdown():
            names = ["propeller1", "propeller2", "propeller3", "propeller4"]
            velocities = [self.speed, -self.speed, self.speed, -self.speed]
            ms = MotorSpeed(name=names, velocity=velocities)
            self.speed_pub.publish(ms)
            # rospy.loginfo(f"PUBLISHING SPEED {self.speed}")
            rate.sleep()

    def drone_speed_callback(self,msg):
        self.speed = msg.data
        rospy.loginfo(f"RECEIVED NEW SPEED {msg.data}")





rospy.init_node('drone_controller', anonymous=True)
dc = DroneController()
try:
    dc.pub()
except rospy.ROSInterruptException:
    pass
except KeyboardInterrupt:
    rospy.loginfo("Closing")
