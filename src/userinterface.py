from features.FeatureExtract import FeatureExtraction
from utils.modelreload import load_words_model, load_digits_model
import streamlit as st
import mediapipe as mp
import numpy as np
import cv2


def local_css(file_name):
    """
    Loads a local css file to override streamlit's builtin css
    :param file_name: css file
    :return: None
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    """
    Loads a remote css api to override the streamlit's font
    :param url: input url
    :return: None
    """
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def predict_frame(container, container1, actions, action_type ,text):
    """
    This function predicts gesture or action performed by extracting the
    cnn feature map with the help of mediapipe combined with transformer
    model classification.

    :param container: placeholder for video frame or text
    :param container1: placeholder for text
    :param actions: gesture labels
    :param action_type: gesture type, weather it belongs to word or a digit
    :param text: Texts to instruct the user to navigate the UI
    :return: predicted action (str)
    """
    feature = FeatureExtraction()
    camera = cv2.VideoCapture(0)
    word_model = load_words_model()
    digit_model = load_digits_model()
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
                if action_type == 'Words':
                    res = word_model.predict(np.expand_dims(sequence, axis=0))[0]
                    predict_action = actions[np.argmax(res)]
                    print(predict_action)
                elif action_type == 'Digits':
                    res = digit_model.predict(np.expand_dims(sequence, axis=0))[0]
                    predict_action = actions[np.argmax(res)]
                    print(predict_action)
    camera.release()
    cv2.destroyAllWindows()
    return predict_action


def navigate_pages():
    word_action = ['Black', 'Cappuccino', 'Chocolate', 'Coffee', 'Flat', 'Hot', 'Large', 'Latte', 'Long', 'No',
             'Regular', 'Short', 'Small', 'White', 'Yes', 'None']
    numbers_action = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'None']
