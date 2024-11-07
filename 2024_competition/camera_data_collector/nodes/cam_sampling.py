#!/usr/bin/env python3
import csv
import math
import numpy as np
import rospy
import random

from geometry_msgs.msg import Twist, PoseStamped, Pose
from tf.transformations import euler_from_quaternion
import tf.transformations as tft
from gazebo_msgs.msg import ModelStates, ModelState

def random_orientation():
    roll = random.uniform(-3.14, 3.14)  # Roll (rotation around x-axis)
    pitch = random.uniform(-3.14, 3.14) # Pitch (rotation around y-axis)
    yaw = random.uniform(-3.14, 3.14)   # Yaw (rotation around z-axis)
    quat = tft.quaternion_from_euler(roll, pitch, yaw)
    return quat

class CamSampler():

    def __init__(self):
        self.pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_callback)
        self.name = 'B1'

        rospy.loginfo("STARTING CAM SAMPLING")

        self.num_captures = 0
        self.max_captures = 2

        rospy.sleep(1)

    def model_state_callback(self,msg):
        if self.num_captures > self.max_captures:
            return
        try:
            idx = msg.name.index(self.name)
            current_position = msg.pose[idx].position
            current_orientation = msg.pose[idx].orientation

            # Print current position and orientation
            rospy.loginfo("Current Position: x=%f, y=%f, z=%f", 
                          current_position.x, 
                          current_position.y, 
                          current_position.z)
            rospy.loginfo("Current Orientation: x=%f, y=%f, z=%f, w=%f", 
                          current_orientation.x, 
                          current_orientation.y, 
                          current_orientation.z, 
                          current_orientation.w)

            # Set a new random position
            state_msg = ModelState()
            state_msg.model_name = self.name
            state_msg.pose.position.x = random.uniform(-5.0, 5.0)  # Random x position
            state_msg.pose.position.y = random.uniform(-5.0, 5.0)  # Random y position
            state_msg.pose.position.z = random.uniform(0.5, 2.0)   # Random z position

            # Set a random orientation
            quat = random_orientation()
            state_msg.pose.orientation.x = quat[0]
            state_msg.pose.orientation.y = quat[1]
            state_msg.pose.orientation.z = quat[2]
            state_msg.pose.orientation.w = quat[3]

            # Publish the new position
            self.pose_pub.publish(state_msg)

            self.num_captures = self.num_captures + 1

            rospy.loginfo("Object position modified and set to random position.")

        except ValueError:
            rospy.logwarn("Model name not found in Gazebo.")


if __name__ == '__main__':
    rospy.init_node('cam_sampling')
    cw = CamSampler()
    rate = rospy.Rate(1)
    rospy.sleep(10)
    try:
        rate.sleep()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass