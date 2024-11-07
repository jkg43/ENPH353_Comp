#! /usr/bin/env python3


import rospy
from geometry_msgs.msg import Twist


def move_callback(data):
  pub.publish(data)


rospy.init_node('topic_publisher')

sub = rospy.Subscriber('/cmd_vel',Twist,move_callback,queue_size=1)
pub = rospy.Publisher('/B1/cmd_vel', Twist,queue_size=1)


try:
    rospy.spin()
except KeyboardInterrupt:
    print("Closing")