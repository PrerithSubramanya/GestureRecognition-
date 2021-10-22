import cv2
import os
import time


class CollectData:
    """
    Helps collection of data and conversion of data frames into video.
    """
    def __init__(self, classType, numClasses=None, folderName=None):
        self.classType = classType
        self.folderName = folderName
        self.numClasses = numClasses

    def captureFrames(self, numframes):
        """
        Helps capture photos mimics burst shot on phones
        also iterates over 30 times capturing variation
        of the same action being performed.

        :param numframes: number of frames required in a
         burst shot
        :return: None
        """

        save_path = os.getcwd()
        for _ in range(self.numClasses):
            class_name = input("Enter the class name: ")
            for counter in range(1,31):
                try:
                    os.chdir(os.path.join(save_path, '../capturedframes', self.classType, self.folderName, class_name, str(counter)))
                except FileNotFoundError:
                    os.makedirs(os.path.join(save_path, '../capturedframes', self.classType, self.folderName, class_name,
                                             str(counter)))
                    cap = cv2.VideoCapture(0)
                    for frame_num in range(numframes):
                        ret, frame = cap.read()
                        jpg_path = os.path.join(save_path, '../capturedframes', self.classType, self.folderName, class_name,
                                                str(counter), 'frames' + str(frame_num - 2) + '.jpg')
                        if ret is True:
                            if frame_num == 0:
                                self.put_text_display(
                                    frame, 'Start action after \'Go\' ', 1, 2000
                                )

                            elif frame_num == 1:
                                self.put_text_display(frame, 'GO', 2, 1000)

                            elif frame_num > 1:
                                cv2.imshow('Frame', frame)
                                cv2.waitKey(1)
                                cv2.imwrite(jpg_path, frame)

                            elif cv2.waitKey(0) == ord('q'):
                                break

                    cap.release()
                    cv2.destroyAllWindows()
                time.sleep(10)

    @staticmethod
    def put_text_display(frame, text, thickness, delay):
        """
        Function to put text on the frames being displayed.

        :param frame: frame being captured and displayed
        :param text: text to displayed over the frames
        :param thickness: thickness of the text
        :param delay: frame pause time
        :return: None
        """
        cv2.putText(
            frame,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            thickness,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow('Frame', frame)
        cv2.waitKey(delay)

    def frametovideo(self):
        """
        Converts the frames captured by captureFrames
        function into videos and stores into one
        video directory.

        :return: None
        """
        current_dir = os.getcwd()
        try:
            os.mkdir(current_dir + "\\AUSLAN_" + self.classType + "videos")
        except:
            pass
        for directory in os.listdir(current_dir):
            if directory == 'capturedframes':
                for folders in os.listdir(os.path.join(current_dir, '../capturedframes', self.classType)):
                    for word in os.listdir(os.path.join(current_dir, '../capturedframes', self.classType, folders)):
                        for fol_num in range(1, 31):
                            img_stack = []
                            size = None
                            for frames in range(60):
                                frames_path = os.path.join(os.getcwd(), '../capturedframes', self.classType, folders, word,
                                                           str(fol_num),'frames' + str(frames) + '.jpg')
                                img = cv2.imread(frames_path)
                                height, width, layers = img.shape
                                size = (width, height)
                                img_stack.append(img)
                            video_path = os.path.join(current_dir, "AUSLAN_" + self.classType + "videos",
                                                      folders + word + "-" + "vid" + str(fol_num) + '.avi')
                            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
                            for item_ in img_stack:
                                out.write(item_)
                            out.release()



                        



