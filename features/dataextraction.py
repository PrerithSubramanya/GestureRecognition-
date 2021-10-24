from src.FeatureExtract import FeatureExtraction
import numpy as np
import cv2
import os

class DataExtraction(FeatureExtraction):
    def __init__(self, folder_name, num_frames):
        super().__init__()
        self.folder_name = folder_name
        self.num_frames = num_frames


    def extract_coordinates(self):
        """
        This function helps extract coordinates from each frame of each video from the video dataset
        and place it in one particular directory with correct naming conventions, the files are saved
        numpy files to be helpful for further modification.

        :return: None
        """
        with self.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            try:
                os.mkdir(os.path.join(os.getcwd(),'data_points_'+self.folder_name))
            except:
                pass
            path = os.path.join(os.getcwd(), 'AUSLAN_'+self.folder_name+'videos')
            for video in os.listdir(path):
                file_name = video.split(".")
                cap = cv2.VideoCapture(os.path.join(os.getcwd(), 'AUSLAN_'+self.folder_name+'videos', video))
                for frames in range(self.num_frames):
                    ret, frame = cap.read()
                    if ret is True and not os.path.isfile(os.path.join(os.getcwd(),'data_points_'+self.folder_name+'copy',
                                                                       file_name[0]+'_'+str(frames)+'.npy')):
                        self.mp_detect(frame, holistic)
                        self.coordinates = self.get_coordinates()
                        coord_path = os.path.join(os.getcwd(), 'data_points_'+self.folder_name+'copy', file_name[0]+'_'
                                                  + str(frames))
                        np.save(coord_path, self.coordinates)
                        self.plot_landmarks()
                        cv2.imshow('frames', self.image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def concatenate_data_pts(self):
        """
        Concatenate numpy files obtained from extract_coordinates function and label them accordingly.

        :return: sequence: concatenated data points for training machine learning model
               : labels: labels for each concatenated data points for supervised training
        """
        iter_folder = os.listdir(os.path.join(os.getcwd(), '../capturedframes', self.folder_name))
        self.className = os.listdir(os.path.join(os.getcwd(), '../capturedframes', self.folder_name, iter_folder[0]))
        label_mapping = {label: num for num, label in enumerate(self.className)}
        for folder in iter_folder:
            for objects in os.listdir(os.path.join(os.getcwd(), '../capturedframes', self.folder_name, folder)):
                for vid_samples in os.listdir(os.path.join(os.getcwd(), '../capturedframes', self.folder_name, folder, objects)):
                    single_videopts = []
                    for frames in range(self.num_frames):
                        res = np.load(os.path.join(os.getcwd(), 'data_points_'+self.folder_name, folder+objects+'-vid'
                                                   + vid_samples+'_'+str(frames)+'.npy'))
                        single_videopts.append(res)
                    self.sequence.append(single_videopts)
                    self.labels.append(label_mapping[objects])
        return self.sequence, self.labels
