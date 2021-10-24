#!/usr/bin/env python
# coding: utf-8

# In[6]:


#importing necessary dependencies
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from features.datacreation import CollectData
from features.dataextraction import DataExtraction
from src.ModelBuild import ModelBuild
import mediapipe as mp
import numpy as np
import cv2
import os

# In[2]:


Data = CollectData('Words', None, None) 


# In[3]:


Data.frametovideo() # convert frames to videos to for a video dataset


# In[2]:


extract_data = DataExtraction('Words', 60)


# In[3]:


extract_data.extract_coordinates() # extract features for the dataset


# In[4]:


landmark, classes = extract_data.concatenate_data_pts() #concatenate the feature extracted in batch size of 60 and label them accordingly


# In[5]:


# save the extracted cnn feature maps using mediapipe
np.save(os.path.join(os.getcwd(), 'WordClasses'), classes)
np.save(os.path.join(os.getcwd(), 'WordLandmarks'), landmark)


# In[3]:


#load the saved dataset
X = np.load('WordLandmarks.npy')
Y = np.load('WordClasses.npy')
Y = to_categorical(Y).astype(int)


# In[4]:


#split the data into train and test
print(X.shape)
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.025)
print(XTrain.shape, YTrain.shape)


# In[ ]:


DENSE_DIM = 7
NUM_HEADS = 3
CLASSES = 16
EPOCHS = 100


# In[4]:


#training the model
Transformer_model = ModelBuild(X.shape[1], X.shape[2], DENSE_DIM, NUM_HEADS, CLASSES, EPOCHS)
model = Transformer_model.fitTransform('Words', XTrain, YTrain)


# In[5]:


#Plotting the stats
Transformer_model.stats(XTest, YTest)


# In[5]:


#loading the trained model weights
Transformer_model = ModelBuild(X.shape[1], X.shape[2], DENSE_DIM, NUM_HEADS, CLASSES, EPOCHS)
model = Transformer_model.constructModel()
model.load_weights(os.path.join(os.getcwd(),'Words3_1_16_100weights.hdf5'))


# In[7]:


#predicting the test data
res = model.predict(XTest)


# In[8]:


#neccasary variables for plotting confusion matrix
ytrue = np.argmax(YTest, axis=1).tolist()
yhat = np.argmax(res, axis=1).tolist()
iter_folder = os.listdir(os.path.join(os.getcwd(), 'Data', 'Words'))
actions = np.array(os.listdir(os.path.join(os.getcwd(), 'Data', 'Words', iter_folder[0])))


# In[10]:


#Build confusion matrix
cm = confusion_matrix(ytrue, yhat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=actions)


# In[11]:


#Plot Confusion matrix
plt.rcParams["figure.figsize"] = (20,14)
disp.plot(cmap=plt.cm.Blues)
plt.show()


# In[13]:


#Function to extract the cnn feature maps using mediapipe for external test video set
def vidfeatureextract(random_test):
    sequence =[]
    cap = cv2.VideoCapture(os.path.join(os.getcwd(),'test', random_test))
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                image, results = extract_data.mp_detect(frame, holistic)
                # draw_styled_landmarks(image,results)
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
    res = model.predict(np.expand_dims(vidfeatureextract(random_test), axis=0))[0]
    counter = 0
    for i in np.argsort(res)[::-1]:
        if counter == 0:
            print(f"  {actions[i]}: {res[i] * 100:5.2f}% : {random_test}")
            counter += 1


# In[14]:


#Printing the confidence percentage and predicted word along with the test video label.
for items in os.listdir(os.path.join(os.getcwd(), 'test')):
    predictPercentage(items)
    print("-------------------")

