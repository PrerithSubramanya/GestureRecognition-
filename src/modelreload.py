from ModelBuild import ModelBuild
import os

EPOCHS = 100
NUMBER_OF_FRAMES = 60
NUMBER_OF_FEATURE = 258

WORD_DENSE_DIM = 16
WORD_CLASS = 16
NUMBER_OF_HEADS_WORDS = 7

DIGITS_DENSE_DIM = 3
NUMBER_OF_HEADS_DIGITS = 7
DIGIT_CLASS = 10


def load_words_model():
    """
    constructs custom keras model with defined parameters.

    :return: returns model for class word
    """
    Word_model = ModelBuild(NUMBER_OF_FRAMES, NUMBER_OF_FEATURE, WORD_DENSE_DIM,
                            NUMBER_OF_HEADS_WORDS, WORD_CLASS, EPOCHS)
    model = Word_model.constructModel()
    model.load_weights(os.path.join(os.getcwd(), 'word_weights.hdf5'))
    return model


def load_digits_model():
    """
    constructs custom keras model with defined parameters

    :return: returns model for class digit
    """
    digit_model = ModelBuild(NUMBER_OF_FRAMES, NUMBER_OF_FEATURE, DIGITS_DENSE_DIM,
                             NUMBER_OF_HEADS_DIGITS, DIGIT_CLASS, EPOCHS)
    model = digit_model.constructModel()
    model.load_weights(os.path.join(os.getcwd(), 'num_weights.hdf5'))
    return model
