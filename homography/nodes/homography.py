#!/usr/bin/env python3

import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from std_msgs.msg import String


model = None
sess = None
graph = None
bridge = CvBridge()

# Define the HSV range
lower_hsv = np.array([113, 113, 44])
upper_hsv = np.array([167, 255, 255])
character_set = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9',' ']

clue_type_to_location = {
    "SIZE": 1,
    "VICTIM": 2,
    "CRIME": 3,
    "TIME": 4,
    "PLACE": 5,
    "MOTIVE": 6,
    "WEAPON": 7,
    "BANDIT": 8
}

def load_nn_model(model_path):
    global model, sess, graph
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        model = load_model(model_path)

def preprocess_patch(patch):
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)  # Resize
    patch = np.expand_dims(patch, axis=(0, -1))  # Add batch and channel dimensions
    patch = patch / 255.0  # Normalize
    return patch

def predict_character(patch):
    global model, sess, graph
    patch = preprocess_patch(patch)
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        predictions = model.predict(patch)

    predicted_index = np.argmax(predictions)
    predicted_char = character_set[predicted_index]
    return predicted_char

def publish_clue(clue_location, clue_prediction):

    pub = rospy.Publisher('/score_tracker', String, queue_size=10)
    team_id = "teamsix"
    team_password = "password"

    # Validate inputs
    if not (-1 <= clue_location <= 8):
        rospy.logerr("Invalid clue_location. Must be between -1 and 8.")
        return
    if not (clue_prediction.isupper() and ' ' not in clue_prediction):
        rospy.logerr("Invalid clue_prediction. Must be uppercase, no spaces.")
        return

    # Create and publish the message
    message = f"{team_id},{team_password},{clue_location},{clue_prediction}"
    pub.publish(message)
    rospy.loginfo(f"Published message: {message}")


def image_callback(msg):
    global model
    try:
        # Convert the ROS Image message to OpenCV format
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Step 1: Convert the frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 2: Apply the HSV threshold to get the mask
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

        # cv2.imshow("Thresholded Mask", mask)  # Thresholded mask (with white regions)

        # Step 3: Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: If contours are found, find the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Check if the approximated contour has 4 points (rectangular)
            if len(approx) == 4:
                # Extract the points of the rectangle
                points = approx.reshape(4, 2)

                # Sort points to ensure proper order: top-left, top-right, bottom-right, bottom-left
                rect = order_points(points)

                # Define the width and height of the desired rectangle
                width = 350
                height = 200
                dst_rect = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")

                # Compute the perspective transform matrix
                M = cv2.getPerspectiveTransform(rect, dst_rect)

                # Warp the perspective to get a top-down view
                transformed = cv2.warpPerspective(frame, M, (width, height))

                # Show the full transformed clue board
                # cv2.imshow("Full Transformed Clue Board", transformed)

                # Step 1: Convert the transformed frame to HSV
                hsv_frame2 = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)

                # Step 2: Apply the HSV threshold to get the mask
                mask2 = cv2.inRange(hsv_frame2, lower_hsv, upper_hsv)

                # Step 3: Invert the mask to get the areas that are not in the HSV range
                mask2 = cv2.bitwise_not(mask2)

                # Step 4: Find contours in the inverse mask
                contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Step 4: If contours are found, find the largest one
                if contours2:
                    largest_contour2 = max(contours2, key=cv2.contourArea)
                    
                    # Approximate the contour to a polygon
                    epsilon2 = 0.02 * cv2.arcLength(largest_contour2, True)
                    approx2 = cv2.approxPolyDP(largest_contour2, epsilon2, True)

                    # Check if the approximated contour has 4 points (rectangular)
                    if len(approx2) == 4:
                        # Extract the points of the rectangle
                        points2 = approx2.reshape(4, 2)

                        # Sort points to ensure proper order: top-left, top-right, bottom-right, bottom-left
                        rect2 = order_points(points2)

                        # Compute the perspective transform matrix
                        M2 = cv2.getPerspectiveTransform(rect2, dst_rect)

                        # Warp the perspective to get a top-down view
                        transformed2 = cv2.warpPerspective(transformed, M2, (width, height))

                        # Show the full transformed clue board
                        # cv2.imshow("Ideal Transformed Clue Board", transformed2)

                        # Debug: Show the original mask and inverted mask
                        # cv2.imshow("Original Mask", mask)
                        # cv2.imshow("Inverted Mask", mask2)

                        cv2.waitKey(1)

                        # Define more specific points for the bottom rectangle and top-right rectangle
                        # Adjust these points based on your requirements
                        bottom_rect_top_left = (13, 110)  # Bottom rectangle starting point
                        bottom_rect_bottom_right = (338, 180)  # Bottom rectangle ending point
                        top_right_rect_top_left = (145, 5)  # Top-right rectangle starting point
                        top_right_rect_bottom_right = (335, 70)  # Top-right rectangle ending point

                        # Crop the bottom rectangle using the specific points
                        bottom_rect = transformed2[bottom_rect_top_left[1]:bottom_rect_bottom_right[1], 
                                                bottom_rect_top_left[0]:bottom_rect_bottom_right[0]]
                        cv2.imshow("Bottom Rectangle", bottom_rect)

                        # Crop the top-right rectangle using the specific points
                        top_right_rect = transformed2[top_right_rect_top_left[1]:top_right_rect_bottom_right[1], 
                                                    top_right_rect_top_left[0]:top_right_rect_bottom_right[0]]
                        cv2.imshow("Top Right Rectangle", top_right_rect)

                        # Now, cut both top and bottom rectangles into smaller images

                        # Define the x-offset and number of smaller images for top and bottom rectangles
                        top_x_offset = 0  # Starting point offset for top rectangle cuts
                        top_width = top_right_rect.shape[1] // 7  # Number of smaller images for top
                        bottom_x_offset = 0  # Starting point offset for bottom rectangle cuts
                        bottom_width = bottom_rect.shape[1] // 12  # Number of smaller images for bottom

                        clue_type_imgs = []
                        # Cut and show the top rectangle into 7 smaller images
                        for i in range(7):
                            top_img = top_right_rect[
                                :, top_x_offset + i*top_width : top_x_offset + (i+1)*top_width
                            ]
                            # cv2.imshow(f"top_img{i+1}", top_img)
                            clue_type_imgs.append(top_img)

                        clue_val_imgs = []
                        # Cut and show the bottom rectangle into 12 smaller images
                        for i in range(12):
                            bottom_img = bottom_rect[
                                :, bottom_x_offset + i*bottom_width : bottom_x_offset + (i+1)*bottom_width
                            ]
                            # cv2.imshow(f"bottom_img{i+1}", bottom_img)
                            clue_val_imgs.append(bottom_img)

                        clue_type = ''.join([predict_character(img) for img in clue_type_imgs]).replace(" ", "")
                        clue_val = ''.join([predict_character(img) for img in clue_val_imgs]).replace(" ", "")
                        clue_location = clue_type_to_location.get(clue_type, 1)

                        rospy.loginfo(f"Clue Type: {clue_type}")
                        rospy.loginfo(f"Clue Value: {clue_val}")

                        publish_clue(clue_location=clue_location, clue_prediction=clue_val)

    except CvBridgeError as e:
        rospy.logerr(f"Error converting image: {e}")

def order_points(pts):
    """
    Sort the points in clockwise order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

def main():
    rospy.init_node('homography_node', anonymous=True)
    load_nn_model(os.path.expanduser('~/ros_ws/nn/model_biggest.keras'))
    rospy.Subscriber('/B1/camera1/image_raw', Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
