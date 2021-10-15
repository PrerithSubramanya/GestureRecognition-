import numpy as np
import cv2
import os
import mediapipe as mp


class FeatureExtraction:
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
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        self.image.flags.writeable = False  # Image is no longer writeable
        self.results = model.process(self.image)  # Make prediction
        self.image.flags.writeable = True  # Image is now writeable
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
        return self.image, self.results

    def plot_landmarks(self):
        # Draw pose connections
        self.drawing.draw_landmarks(self.image, self.results.pose_landmarks, self.holistic.POSE_CONNECTIONS,
                                    self.drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                    self.drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        # Draw left hand connections
        self.drawing.draw_landmarks(self.image, self.results.left_hand_landmarks, self.holistic.HAND_CONNECTIONS,
                                    self.drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    self.drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        # Draw right hand connections
        self.drawing.draw_landmarks(self.image, self.results.right_hand_landmarks, self.holistic.HAND_CONNECTIONS,
                                    self.drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                    self.drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def get_coordinates(self):
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

    def extract_coordinates(self, folder_name, no_frames):
        with self.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            try:
                os.mkdir(os.path.join(os.getcwd(),'data_points_'+folder_name))
            except:
                pass
            path = os.path.join(os.getcwd(), 'AUSLAN_'+folder_name+'videos')
            for video in os.listdir(path):
                file_name = video.split(".")
                cap = cv2.VideoCapture(os.path.join(os.getcwd(), 'AUSLAN_'+folder_name+'videos', video))
                for frames in range(no_frames):
                    ret, frame = cap.read()
                    if ret is True:
                        if not os.path.isfile(os.path.join(os.getcwd(),'data_points_'+folder_name, file_name[0]+'_'+str(frames)+'.npy')):
                            self.mp_detect(frame, holistic)
                            self.coordinates = self.get_coordinates()
                            coord_path = os.path.join(os.getcwd(),'data_points_'+folder_name, file_name[0]+'_'+str(frames))
                            np.save(coord_path,self.coordinates)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def concatenate_data_pts(self, folder_name, no_frames):
        iter_folder = os.listdir(os.path.join(os.getcwd(), 'Data', folder_name))
        self.className = os.listdir(os.path.join(os.getcwd(), 'Data', folder_name, iter_folder[0]))
        label_mapping = {label: num for num, label in enumerate(self.className)}
        for folder in iter_folder:
            for objects in os.listdir(os.path.join(os.getcwd(), 'Data', folder_name, folder)):
                for vid_samples in os.listdir(os.path.join(os.getcwd(), 'Data', folder_name, folder, objects)):
                    single_videopts = []
                    for frames in range(no_frames):
                        res = np.load(os.path.join(os.getcwd(), 'data_points_'+folder_name, folder+objects+'-vid'+vid_samples+'_'+str(frames)+'.npy'))
                        single_videopts.append(res)
                    self.sequence.append(single_videopts)
                    # print(objects)
                    self.labels.append(label_mapping[objects])
        return self.sequence, self.labels







