import cv2
import os
import time

class collectData:
    def __init__(self,classType, folderName, numClasses):
        self.classType = classType
        self.folderName = folderName
        self.numClasses = numClasses

    def captureFrames(self, numFrames):
        save_path = os.getcwd()
        for i in range(self.numClasses):
            class_name = input("Enter the class name: ")
            for counter in range(1,31):
                try:
                    os.chdir(os.path.join(save_path, 'Data', self.classType, self.folderName, class_name, str(counter)))
                except FileNotFoundError:
                    os.makedirs(os.path.join(save_path, 'Data', self.classType, self.folderName, class_name, str(counter)))
                    cap = cv2.VideoCapture(0)
                    for frame_num in range(numFrames):
                            ret, frame = cap.read()
                            jpg_path = os.path.join(save_path,'Data',self.classType, self.folderName, class_name, str(counter), 'frames'
                                                    + str(frame_num - 2) + '.jpg')
                            if ret == True:
                                if frame_num == 0:
                                    cv2.putText(frame, 'Start action after \'Go\' ', (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                                                (255, 0, 0), 2, cv2.LINE_AA)
                                    cv2.imshow('Frame', frame)
                                    cv2.waitKey(2000)
                                elif frame_num == 1:
                                    cv2.putText(frame, 'GO', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                (255, 0, 0), 2, cv2.LINE_AA)
                                    cv2.imshow('Frame', frame)
                                    cv2.waitKey(1000)
                                elif frame_num>1:
                                    cv2.imshow('Frame', frame)
                                    cv2.waitKey(1)
                                    cv2.imwrite(jpg_path, frame)
                                elif cv2.waitKey(0) == ord('q'):
                                    break

                    cap.release()
                    cv2.destroyAllWindows()
                time.sleep(10)

    def frameVideo(self):
        current_dir = os.getcwd()
        try:
            os.mkdir(current_dir + "\\AUSLAN_" + self.classType + "videos")
        except:
            pass
        for dir in os.listdir(current_dir):
            if dir == 'Data':
                for folders in os.listdir(os.path.join(current_dir, 'Data', self.classType)):
                    for word in os.listdir(os.path.join(current_dir, 'Data', self.classType, folders)):
                        for fol_num in range(1, 31):
                            img_stack = []
                            for frames in range(60):
                                frames_path = os.path.join(os.getcwd(), 'Data', self.classType, folders, word, str(fol_num),
                                                           'frames' + str(frames) + '.jpg')
                                img = cv2.imread(frames_path)
                                height, width, layers = img.shape
                                size = (width, height)
                                img_stack.append(img)
                            video_path = os.path.join(current_dir, "AUSLAN_" + self.classType + "videos",
                                                      folders + word + "-" + "vid" + str(fol_num) + '.avi')
                            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
                            for i in range(len(img_stack)):
                                out.write(img_stack[i])
                            out.release()



                        



