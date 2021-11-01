## <ins>A Gesture Recognition System for Mute People at Cafes</ins> **

## <ins>Project Introduction</ins>
The primary objective is to bridge the communication gap between non-sign language speaking café patrons and sign language speaking mute customers by making the system to understand the mute people's sign language, making it much easier for them to order in cafes than before, and which not only eases the communication and customer experience but only improves the sales of the café. In this project, using the Transformer architecture, a gesture recognition model is developed to capture the dynamic gestures made and as input for this model, the hand landmarks from the customers are extracted using Google’s open-source framework called MediaPipe. The data set is captured in a restricted environment with predefined set of parameters which in future can be scaled as per the requirement posing great potential for the gesture recognition model.

## Technologies  
1. Python  
2. Tensorflow  
3. Keras  
4. Streamlit  

## <ins>File structure</ins>

├───.gitignore  
├───requirements.txt  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
├───features  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── datacreation.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── dataextraction.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
├───src  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── FeatureExtract.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── ModelBuild.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── modelreload.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── num_weights.hdf5  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── style.css  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── Transformers.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── userinterface.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── word_weights.hdf5  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
└───utils  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── NumbersModel.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─── WordsModel.py  

## <ins>Setup and Launch</ins>  

To run this project :  

1. Extract all files and folders from zip file
2. Open terminal in extracted program folder path
3. Follow the terminal commands to open a virtual environment:  
```
$ or project folder path> python --version
$ or project folder path> python -m pip install --upgrade pip
$ or project folder path> pip install virtualenv
$ or project folder path> virtualenv <<virtual environment name>>
$ or project folder path> <<your virtual environment name>>\Scripts\activate
```
4. Install all the dependency libraries required for the project which are available in the requirements.txt file as follows:  
```
(<<virtual environment>>)$ or project folder path> python -m pip install -r requirements.txt
```
5. Now move into the src folder and run the project using streamlit as follows:
```
(<<virtual environment>>)$ or project folder path> cd src
(<<virtual environment>>)$ or project folder path> streamlit run userinterface.py
```
6. The system will be running.
