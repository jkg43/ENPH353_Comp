#!/usr/bin/env python3

import os
import cv2
import rospy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class HomographyProcessor:
    def __init__(self):
        self.bridge = cv2  # OpenCV will handle image processing
        self.model = None
        self.sess = None
        self.graph = None
        self.character_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
        self.clue_type_to_location = {
            "SIZE": 1,
            "VICTIM": 2,
            "CRIME": 3,
            "TIME": 4,
            "PLACE": 5,
            "MOTIVE": 6,
            "WEAPON": 7,
            "BANDIT": 8
        }
        self.lower_hsv = np.array([113, 113, 44])
        self.upper_hsv = np.array([167, 255, 255])
        self.load_nn_model()

    def load_nn_model(self):
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session()
        self.graph = tf.compat.v1.get_default_graph()
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            self.model = load_model(os.path.expanduser('~/ros_ws/nn/model_biggest.keras'))

    def preprocess_patch(self, patch):
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)  # Resize
        patch = np.expand_dims(patch, axis=(0, -1))  # Add batch and channel dimensions
        patch = patch / 255.0  # Normalize
        return patch

    def predict_character(self, patch):
        patch = self.preprocess_patch(patch)
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            predictions = self.model.predict(patch)
        predicted_index = np.argmax(predictions)
        return self.character_set[predicted_index]

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect
    
    def publish_clue(self, clue_location, clue_prediction):

        # Validate inputs
        if not (-1 <= clue_location <= 8):
            rospy.logerr("Invalid clue_location. Must be between -1 and 8.")
            return
        if not (clue_prediction.isupper() and ' ' not in clue_prediction):
            rospy.logerr("Invalid clue_prediction. Must be uppercase, no spaces.")
            return

        # Create and publish the message
        message = f"{self.team_id},{self.team_password},{clue_location},{clue_prediction}"
        self.pub.publish(message)
        rospy.loginfo(f"Published message: {message}")

    def process_image(self, frame):
        """
        Accepts an OpenCV image and processes it to extract clues.
        """
        print(frame.shape)
        cv2.imshow(frame)
        try:
            # Step 1: Convert the frame to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Step 2: Apply the HSV threshold to get the mask
            mask = cv2.inRange(hsv_frame, self.lower_hsv, self.upper_hsv)
            
            # Step 3: Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                print("IN1")
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                if len(approx) == 4:
                    print("IN2")
                    points = approx.reshape(4, 2)
                    rect = self.order_points(points)

                    # Define the width and height of the desired rectangle
                    width, height = 350, 200
                    dst_rect = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype="float32")
                    print("IN3")
                    M = cv2.getPerspectiveTransform(rect, dst_rect)
                    transformed = cv2.warpPerspective(frame, M, (width, height))
                    print("IN4")
                    # Additional steps to process `transformed` as required...
                    # Extract and predict top and bottom clues
                    # Debug images, splitting logic, and prediction steps...

                    # Example of extracting and predicting:
                    clue_type_imgs = []  # Replace with actual segmented images
                    clue_val_imgs = []  # Replace with actual segmented images

                    clue_type = ''.join([self.predict_character(img) for img in clue_type_imgs]).replace(" ", "")
                    clue_val = ''.join([self.predict_character(img) for img in clue_val_imgs]).replace(" ", "")
                    clue_location = self.clue_type_to_location.get(clue_type, 1)
                    print("IN5")
                    print(f"Clue Type: {clue_type}")
                    print(f"Clue Value: {clue_val}")
                    self.publish_clue(clue_location=clue_location, clue_prediction=clue_val)
                    print("IN6")

        except Exception as e:
            print(f"Error processing image: {e}")
            return None


# def main():
#     rospy.init_node('homography_node', anonymous=True)
#     print("Loaded")
#     load_nn_model(os.path.expanduser('~/ros_ws/nn/model_biggest.keras'))
#     rospy.Subscriber('/B1/camera0/image_raw', Image, image_callback)

#     rospy.spin()

# if __name__ == '__main__':
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass
#     finally:
#         cv2.destroyAllWindows()
