#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing necessary dependencies
# from imports import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from features.datacreation import CollectData
from features.dataextraction import DataExtraction
from src.ModelBuild import ModelBuild
import numpy as np
import cv2
import os
import mediapipe as mp

# In[6]:


Data = CollectData('Numbers', None, None)


# In[24]:


Data.frametovideo() # convert frames to videos to for a video dataset


# In[3]:


extract_data = DataExtraction('Numbers', 60)


# In[ ]:


extract_data.extract_coordinates() # extract features for the dataset


# In[4]:


landmark, classes = extract_data.concatenate_data_pts() #concatenate the feature extracted in batch size of 60 and label them accordingly


# In[7]:


# save the extracted cnn feature maps using mediapipe
np.save(os.path.join(os.getcwd(), 'NumClasses'), classes)
np.save(os.path.join(os.getcwd(), 'NumLandmarks'), landmark)


# In[8]:


#load the saved dataset
X = np.load('NumLandmarks.npy')
Y = np.load('NumClasses.npy')
Y = to_categorical(Y).astype(int)


# In[9]:


#split the data into train and test
print(X.shape)
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.025)
print(XTrain.shape, YTrain.shape)


# In[ ]:


DENSE_DIM = 3
NUM_HEADS = 2
CLASSES = 10
EPOCHS = 100


# In[10]:


#training the model
Numbers_model = ModelBuild(X.shape[1], X.shape[2], DENSE_DIM, NUM_HEADS, CLASSES, EPOCHS)
model = Numbers_model.fitTransform('Num', XTrain, YTrain)


# In[11]:


#Plotting the stats
Numbers_model.stats(XTest, YTest)


# In[14]:


#loading the trained model weights
Numbers_model = ModelBuild(X.shape[1], X.shape[2], DENSE_DIM, NUM_HEADS, CLASSES, EPOCHS)
model = Numbers_model.constructModel()
model.load_weights(os.path.join(os.getcwd(),'word_weights.hdf5'))


# In[15]:


#predicting the test data
res = model.predict(XTest)


# In[16]:


#neccasary variables for plotting confusion matrix
ytrue = np.argmax(YTest, axis=1).tolist()
yhat = np.argmax(res, axis=1).tolist()
iter_folder = os.listdir(os.path.join(os.getcwd(), 'Data', 'Numbers'))
actions = np.array(os.listdir(os.path.join(os.getcwd(), 'Data', 'Numbers', iter_folder[0])))


# In[18]:


#Build confusion matrix
cm = confusion_matrix(ytrue, yhat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=actions)


# In[19]:


#Plot Confusion matrix
plt.rcParams["figure.figsize"] = (20,14)
disp.plot(cmap=plt.cm.Blues)
plt.show()


# In[21]:


#Function to extract the cnn feature maps using mediapipe for external test video set
def prepare_testVid(random_test):
    sequence =[]
    cap = cv2.VideoCapture(os.path.join(os.getcwd(),'nums_vids', random_test))
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                image, results = extract_data.mp_detect(frame, holistic)
                keypoints = extract_data.get_coordinates()
                sequence.append(keypoints)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
                
    cap.release()
    cv2.destroyAllWindows()
    sequence = sequence[-60:]
    return sequence


#Predict the confidence percentage along with the perdicted class
def predictPercentage(random_test):
    res = model.predict(np.expand_dims(prepare_testVid(random_test), axis=0))[0]
    counter = 0
    for i in np.argsort(res)[::-1]:
        if counter == 0:
            print(f"  {actions[i]}: {res[i] * 100:5.2f}% : {random_test}")
            counter += 1
        


# In[22]:


#Printing the confidence percentage and predicted word along with the test video label.
for items in os.listdir(os.path.join(os.getcwd(), 'nums_vids')):
    predictPercentage(items)
    print("-------------------")


# In[ ]:




