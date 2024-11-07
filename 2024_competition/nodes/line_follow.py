#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import cv2 as cv
import numpy as np


def find_gaps(list):
    """
    @brief finds continuous gaps of zeros in a list of numbers

    @param list the list of numbers to find gaps for

    @return a list of tuples containing the start and end of each gap
    """
    gaps = []
    gap_start = -1
    gap_end = -1
    in_gap = False
    for i in range(len(list)):
        if list[i] == 0:
            if not in_gap:
                gap_start = i
                in_gap = True
        elif in_gap:
            gap_end = i
            in_gap = False
            gaps.append((gap_start,gap_end))
    if in_gap:
        gap_end = len(list)
        gaps.append((gap_start,gap_end))
    return gaps

def contour(frame,target):
    """
    @brief finds the contours of an image

    @param frame the image to find the contours of
    @param target the image to draw the contours on

    @return the target image with the contours of the frame image draw on it
    """
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(gray_frame,127,255,0)
    contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return cv.drawContours(target, contours, -1, (0,255,0), 3)



class line_follower:
    """
    @brief inputs a video feed and outputs move commands to follow a line
    """

    def __init__(self):
        """
        @brief initializes the follower

        Subscribes to the robot camera image topic and publishes movement commands to cmd_vel

        Keeps track of the previous direction the robot was rotating
        """
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw",Image,self.image_callback,queue_size = 3)
        self.vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        self.prev_rot_dir = 1

    def image_callback(self,data):
        """
        @brief detects a line and outputs movement commands to follow it

        @param data the current image from the camera
        
        Uses contours to find a gap in the bottom of the screen where the line is, and tries to 
        rotate towards it while moving forward
        """
        try:

            frame = self.bridge.imgmsg_to_cv2(data,"bgr8")

            #draw contours on a blank image
            frame_contours = contour(frame,np.zeros(frame.shape, dtype=np.uint8))

            #detect a gap in the bottom of the contour image to know where the line is
            #contours outputs an image of either 0 or 255 on the green channel, so just look at that
            last_row = frame_contours[-1,:,1]

            gaps = find_gaps(last_row)

            max_gap_len = 0
            max_gap_start = 0
            max_gap_end = 0

            #select largest gap
            for gap in gaps:
                gap_start = gap[0]
                gap_end = gap[1]
                if gap_end - gap_start > max_gap_len:
                    max_gap_len = gap_end - gap_start
                    max_gap_start = gap_start
                    max_gap_end = gap_end


            gap_center = (max_gap_start + max_gap_end)/2

            width = frame.shape[1]

            screen_center = width / 2

            rot_speed = 2.0

            
            move_twist = Twist()

            #rotate towards the direction of the line, turning faster the further away it is from centered
            if max_gap_len != 0:

                move_twist.linear.x = 0.5
                #go right - neg z rot
                #go left - pos z rot
                move_twist.angular.z = rot_speed * (screen_center - gap_center) / width

                self.vel_pub.publish(move_twist)
                self.prev_rot_dir = np.sign(screen_center - gap_center)
            #if the line is not detected, stop moving forward and rotate in the same direction as 
            #  the robot was previously moving, until the line is found
            else:
                move_twist.angular.z = rot_speed * self.prev_rot_dir
                self.vel_pub.publish(move_twist)



        except CvBridgeError as e:
            print(e)
        
#initialize the ROS node and line follower class, and waits for the callback to be triggered
rospy.init_node('line_follower',anonymous=True)
lf = line_follower()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Closing")



