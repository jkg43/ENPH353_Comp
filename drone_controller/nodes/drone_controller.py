#!/usr/bin/env python3
import csv
import math
import numpy as np
import rospy
import random

from drone_internals.msg import MotorSpeed
from std_msgs.msg import Float32, String


class DroneController():

    hz = 20
    time = 2 # run for 2 seconds
    names = ["propeller1", "propeller2", "propeller3", "propeller4"]

    def __init__(self):
        self.running = False
        self.speed_pub = rospy.Publisher('/motor_speed_cmd',MotorSpeed,queue_size=1)
        self.score_pub = rospy.Publisher('/score_tracker',String,queue_size=5)
        self.speed_sub = rospy.Subscriber('/drone_speed_control',Float32,self.drone_speed_callback)
        self.backd_sub = rospy.Subscriber('/drone_back_control',Float32,self.drone_back_callback)
        self.start_sub = rospy.Subscriber('/start_drone',Float32,self.start_drone)
        self.speeds = [0.0, 0.0, 0.0, 0.0]
        self.rate = rospy.Rate(self.hz) # 20 Hz (publishes 20 times per second)

        self.fwd_speed = 45.9
        self.back_delta = -1.015
        self.count=0
        rospy.loginfo("STARTING DRONE CONTROLLER")

    # repeatedly publish current speed
    def pub(self):
        while not rospy.is_shutdown():
            ms = MotorSpeed(name=self.names, velocity=self.speeds)
            self.speed_pub.publish(ms)
            # rospy.loginfo(f"PUBLISHING SPEED {self.speed}")
            if self.running:
                self.count += 1
                if(self.count >= self.time * self.hz): # stop drone and timer once desired amount of time has passed
                    self.count = 0
                    self.running = False
                    self.score_pub.publish('Team6,password,-1,NA') #stop timer
                    self.speeds = [0.0, 0.0, 0.0, 0.0]
                    rospy.loginfo("STOPPING DRONE")
            self.rate.sleep()

    # start the timer and drone
    def start(self):
        self.running = True
        self.speeds = [self.fwd_speed, -self.fwd_speed-self.back_delta, self.fwd_speed+self.back_delta, -self.fwd_speed]
        self.score_pub.publish('Team6,password,0,NA') # start timer

        
        
    def drone_speed_callback(self,msg): # rostopic pub  -1 /drone_speed_control std_msgs/Float32 -- 45
        self.fwd_speed = msg.data
        rospy.loginfo(f"RECEIVED NEW SPEED {msg.data}")

    def drone_back_callback(self,msg): # rostopic pub  -1 /drone_back_control std_msgs/Float32 -- -1.04
        self.back_delta = msg.data
        rospy.loginfo(f"RECEIVED NEW BACK SPEED {msg.data}")
    
    # msg is seconds to run for
    def start_drone(self,msg): # rostopic pub -1 /start_drone std_msgs/Float32 -- 5
        self.time = msg.data
        rospy.loginfo(f"RECEIVED NEW BACK SPEED {msg.data}")
        if not self.running and msg.data != 0:
            rospy.loginfo("STARTING DRONE")
            self.start()




rospy.init_node('drone_controller', anonymous=True)
dc = DroneController()
try:
    dc.pub()
except rospy.ROSInterruptException:
    pass
except KeyboardInterrupt:
    rospy.loginfo("Closing")
