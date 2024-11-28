#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os

# Initialize the CvBridge
bridge = CvBridge()

# Path to your template image
template_path = os.path.join(os.path.expanduser('~'), 'ros_ws', 'src', 'sift', 'fizz_clue.png')

# Initialize SIFT and FLANN matcher
sift = cv2.SIFT_create()

# Read the template image in grayscale
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
template_keypoints, template_desc = sift.detectAndCompute(template, None)
template = cv2.drawKeypoints(template, template_keypoints, template)

# Set up feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()  # Default parameters
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

def compute_homography(frame, frame_keypoints, frame_desc, good_matches):
    if len(good_matches) > 10:
        # Extract the coordinates of the keypoints
        template_points = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        frame_points = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography matrix and associated mask
        matrix, mask = cv2.findHomography(template_points, frame_points, cv2.RANSAC, 5.0)
        mask = mask.ravel().tolist()

        # Handle template dimensions correctly for both grayscale and color images
        if len(template.shape) == 2:  # Grayscale template
            height, width = template.shape
        else:  # Color template (3 channels)
            height, width, _ = template.shape 

        # Transform points from template to frame using homography matrix
        corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        transform = cv2.perspectiveTransform(corners, matrix)  # Transformed positions of the corners

        # Draw polygon on frame
        homography_frame = cv2.polylines(frame, [np.int32(transform)], True, (0, 255, 0), 3)  # Draws a closed shape
        return homography_frame
    else:
        # If not enough matches, just draw the keypoints on the frame
        return cv2.drawKeypoints(frame, frame_keypoints, None)


def image_callback(msg):
    try:
        # Convert the ROS Image message to OpenCV format
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect SIFT features in the current frame
        frame_keypoints, frame_desc = sift.detectAndCompute(gray_frame, None)

        # Match descriptors using FLANN
        matches = flann_matcher.knnMatch(template_desc, frame_desc, k=2)
        good_matches = [m1 for m1, m2 in matches if m1.distance < 0.6 * m2.distance]

        # Compute homography and draw matches
        frame_with_homography = compute_homography(frame, frame_keypoints, frame_desc, good_matches)

        # Display the processed frame
        cv2.imshow("Homography View", frame_with_homography)
        cv2.waitKey(1)

    except CvBridgeError as e:
        rospy.logerr(f"Error converting image: {e}")

def main():
    rospy.init_node('homography_node', anonymous=True)
    rospy.Subscriber('/B1/camera1/image_raw', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
