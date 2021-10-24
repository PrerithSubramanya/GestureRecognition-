import cv2
import streamlit as st
import mediapipe as mp
from modelreload import load_words_model, load_digits_model
from FeatureExtract import FeatureExtraction
import numpy as np

import time

Word_model = load_words_model()

Num_model = load_digits_model()


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
    predictWord = ''
    feature = FeatureExtraction()
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
            if len(sequence) == 60 and stage != 4:
                res = Word_model.predict(np.expand_dims(sequence, axis=0))[0]
                predictWord = actions[np.argmax(res)]
                print(predictWord)
                if stage == 0 and predictWord == 'Yes' or cv2.waitKey(1) & 0xFF == ord('q'):
                    stage += 1
                    break
                elif stage == 2 or cv2.waitKey(1) & 0xFF == ord('q'): #prediction of items
                    if predictWord in ['Hot', 'Long', 'Flat', 'Short']:
                        predictions.append(np.argmax(res))
                        if (
                            np.unique(predictions[-10:])[0] == np.argmax(res)
                            and res[np.argmax(res)] > threshold
                        ):
                            if not sentence:
                                sentence.append(actions[np.argmax(res)])
                            elif len(sentence) != 2:
                                if actions[np.argmax(res)] != sentence[0]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                predictWord = '-'.join(sentence)
                                stage += 1
                                break
                    else:
                        if len(sentence) == 2:
                            predictWord = "-".join(sentence)
                        stage += 1
                        break
                elif stage == 3 and predictWord in ['Yes', 'No'] or cv2.waitKey(1) & 0xFF == ord('q'):
                    stage += 1
                    break
                elif stage == 5 and predictWord in ['Yes', 'No'] or cv2.waitKey(1) & 0xFF == ord('q'):
                    stage +=1
                    break
            elif len(sequence) == 60 and stage == 4 or cv2.waitKey(1) & 0xFF == ord('q'):
                res = Num_model.predict(np.expand_dims(sequence, axis=0))[0]
                predictWord = actions[np.argmax(res)]
                stage +=1
                break

        camera.release()
        cv2.destroyAllWindows()
        return stage, predictWord


def control_flow(stage, container, container1):
    words = ['Black', 'Cappuccino', 'Chocolate', 'Coffee', 'Flat', 'Hot', 'Large', 'Latte', 'Long', 'No',
             'Regular', 'Short', 'Small', 'White', 'Yes', 'None']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'None']
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
        ordered_item = word
    if stage == 3:
        container1.header('')
        container.header(f'Did you order {word}')
        for i in range(5, -1, -1):
            container1.subheader(f'To confirm gesture YES or to change order gesture NO in {i}')
            time.sleep(1)
            if i == 0:
                text = ''
                stage, word = make_prediction(container, container1, words, text, stage)
    if stage == 4:
        if word == 'Yes':
            text = 'Please mention the quantity'
            stage, word = make_prediction(container, container1, numbers, text, stage)
            quantity = word
            container1.header('')
            container.header(f'Did you want {quantity} x {ordered_item}')
            for i in range(5, -1, -1):
                container1.subheader(f'To confirm gesture YES or to change order gesture NO in {i}')
                time.sleep(1)
                if i == 0:
                    text = ''
                    stage, word = make_prediction(container, container1, words, text, stage)
        elif word == 'No':
            stage = 2
            print("After changing it: ",stage)
            container = st.empty()
            container1 = st.empty()
    if stage == 6 and word == 'Yes':
        container1.header('Thank you have a great day')
        container.header(f'collect your order {quantity} x {ordered_item}')
        time.sleep(20)
        stage =0

local_css('style.css')
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

st.title("Gesture recognition for cafe")
stage = 0
container = st.empty()
container1 = st.empty()
word = None
ordered_item = None

while True:
    control_flow(stage, container, container1)



