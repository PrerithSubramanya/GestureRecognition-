import numpy as np
import cv2
import mediapipe as mp


class FeatureExtraction:
    """
    Helps in extraction of landmark from each frame
    and concatenate 60 landmark files to be labelled
    to a particular class
    """

    def __init__(self):
        self.holistic = mp.solutions.holistic
        self.drawing = mp.solutions.drawing_utils
        self.pose = self.lh = self.rh = None
        self.image = None
        self.results = None
        self.coordinates = None
        self.sequence = []
        self.className = None
        self.labels = []

    def mp_detect(self, image, model):
        """
        Helps in processing each frame to detect the landmarks over a human in the frame.

        :param image: frame from each video to be processed
        :param model: mediapipe pipelines, here we use holistic pipeline to extract pose, left and right hand points
        :return: image: frames from the video dataset
               : result: landmark coordinates of all recognisable points
        """
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        self.image.flags.writeable = False  # Image is no longer writeable
        self.results = model.process(self.image)  # Make prediction
        self.image.flags.writeable = True  # Image is now writeable
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
        return self.image, self.results

    def plot_landmarks(self):
        """
        This function helps in plotting the detected landmarks in each frame
        Here, it is only restricted to plotting pose, left and right hand.

        :return: None
        """
        # Draw pose landmarks
        self.drawing.draw_landmarks(self.image, self.results.pose_landmarks, self.holistic.POSE_CONNECTIONS,
                                    self.drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                    self.drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        # Draw left hand landmarks
        self.drawing.draw_landmarks(self.image, self.results.left_hand_landmarks, self.holistic.HAND_CONNECTIONS,
                                    self.drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    self.drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        # Draw right hand landmarks
        self.drawing.draw_landmarks(self.image, self.results.right_hand_landmarks, self.holistic.HAND_CONNECTIONS,
                                    self.drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                    self.drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def get_coordinates(self):
        """
        This function helps in separating the key coordinates extracted from each frame into pose,
        left and right hand

        :return: returns a concatenated array of coordinate points for pose, right and left hand of a human
        """
        self.pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                              self.results.pose_landmarks.landmark]).flatten() if self.results.pose_landmarks \
            else np.zeros(33 * 4)
        self.lh = np.array([[res.x, res.y, res.z] for res in
                            self.results.left_hand_landmarks.landmark]).flatten() if self.results.left_hand_landmarks \
            else np.zeros(21 * 3)
        self.rh = np.array([[res.x, res.y, res.z] for res in
                            self.results.right_hand_landmarks.landmark]).flatten() if self.results.right_hand_landmarks \
            else np.zeros(21 * 3)
        return np.concatenate([self.pose, self.lh, self.rh])
