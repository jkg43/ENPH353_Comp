#!/usr/bin/env python3
import csv
import math
import numpy as np
import rospy
import random

from geometry_msgs.msg import Twist, PoseStamped, Pose, Vector3, Wrench
from gazebo_msgs.msg import ModelStates, ModelState
from std_msgs.msg import Float32



class FlightTestControl():

    def __init__(self):
        self.pub = rospy.Publisher('/thruster_force', Wrench, queue_size=10)
        self.sub = rospy.Subscriber('/thruster_force_control',Float32,self.force_control_callback)
        self.name = 'B1'

        self.zThrust = 0.0


        rospy.loginfo("STARTING THRUSTER CONTROL")
        rospy.sleep(1)

    def force_control_callback(self,msg):
        self.zThrust = msg.data

    def apply_thrust(self):
        wrench = Wrench()
        wrench.force = Vector3(0.0,0.0,self.zThrust)
        wrench.torque = Vector3(0.0,0.0,0.0)

        self.pub.publish(wrench)





if __name__ == '__main__':
    rospy.init_node('thruster_force_publisher')
    ftc = FlightTestControl()
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            ftc.apply_thrust()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass