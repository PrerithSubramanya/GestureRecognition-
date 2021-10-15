import cv2
import streamlit as st
import mediapipe as mp
import ModelBuild as Mb
import FeatureExtract as Fe
import numpy as np
import os
import time

Num_frames = 60
Num_features = 258
ff_dense = 7
multi_heads = 3
words_class = 16
Num_class = 9
epochs = 50
Word_model = Mb.ModelBuild(Num_frames, Num_features, ff_dense, multi_heads, words_class, epochs)
w_model = Word_model.constructModel()
w_model.load_weights(os.path.join(os.getcwd(),'word_weights.hdf5'))


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def make_prediction(container, container1, actions, text, stage):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    feature = Fe.FeatureExtraction()
    camera = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera.isOpened():
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            container.image(frame)
            container1.header(text)
            feature.mp_detect(frame, holistic)
            keypoints = feature.get_coordinates()
            sequence.append(keypoints)
            sequence = sequence[-60:]
            if len(sequence) == 60:
                res = w_model.predict(np.expand_dims(sequence, axis=0))[0]
                predictWord = actions[np.argmax(res)]
                # print(predictWord)
                if stage == 0 and predictWord == 'Yes' or cv2.waitKey(1) & 0xFF == ord('q'):
                    stage += 1
                    break
                elif stage == 2 or cv2.waitKey(1) & 0xFF == ord('q'):
                    if predictWord in ['Hot', 'Long', 'Flat', 'Short']:
                        predictions.append(np.argmax(res))
                        if np.unique(predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) == 0:
                                    sentence.append(actions[np.argmax(res)])
                                elif len(sentence) != 2:
                                    if actions[np.argmax(res)] != sentence[0]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    predictWord = '-'.join(sentence)
                                    stage += 1
                                    break
                    # elif predictWord in ['Large', 'None', 'Regular', 'Small']:
                    #     predictions.append(np.argmax(res))
                    #     if np.unique(predictions[-10:])[0] == np.argmax(res):
                    #         if res[np.argmax(res)] > threshold:
                    #             if len(sentence) == 0:
                    #                 sentence.append(actions[np.argmax(res)])
                    #             elif len(sentence) != 2:
                    #                 if actions[np.argmax(res)] != sentence[0]:
                    #                     sentence.append(actions[np.argmax(res)])
                    #                     predictWord= sentence[1]
                    #     stage += 1
                    #     break
                    else:
                        if len(sentence) == 2:
                            predictWord = "-".join(sentence)
                        stage += 1
                        break
                elif stage == 3 and predictWord in ['Yes', 'No' ] or cv2.waitKey(1) & 0xFF == ord('q'):
                    stage += 1
                    break

        camera.release()
        cv2.destroyAllWindows()
        return stage, predictWord


def control_flow(stage, container, container1):
    words = ['Black', 'Cappuccino', 'Chocolate', 'Coffee', 'Flat', 'Hot', 'Large', 'Latte', 'Long', 'No',
             'Regular', 'Short', 'Small', 'White', 'Yes', 'None']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'None']
    print("method called again with the stage as: ", stage)
    if stage == 0:
        text = 'Would you like to begin?'
        stage, word = make_prediction(container, container1, words, text, stage)
    if stage == 1:
        container1.header('')
        text = 'Welcome, What is your Order?'
        container.header(text)
        for i in range(5, -1, -1):
            container1.subheader(f'start action in {i}')
            time.sleep(1)
            if i == 0:
                stage += 1
    if stage == 2:
        text = 'Start action to order item'
        stage, word = make_prediction(container, container1, words, text, stage)
    if stage == 3:
        container1.header('')
        container.header(f'Did you order {word}')
        for i in range(5, -1, -1):
            container1.subheader(f'To confirm gesture YES or to change order gesture NO {i}')
            time.sleep(1)
            if i == 0:
                text = ''
                stage, word = make_prediction(container, container1, words, text, stage)
    if stage == 4:
        if word == 'Yes':
            text = 'Please mention the quantity'
            stage, word = make_prediction(container, container1, words, text, stage)
        elif word == 'No':
            stage = 2
            print("After changing it: ",stage)
            container = st.empty()
            container1 = st.empty()



local_css('style.css')
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

st.title("Gesture recognition for cafe")
stage = 0
container = st.empty()
container1 = st.empty()
word = None

while True:
    control_flow(stage, container, container1)



