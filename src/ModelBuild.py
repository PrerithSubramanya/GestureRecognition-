from Transformers import PositionalEmbedding, TransformerEncoder
from tensorflow.keras import layers
from tensorflow import keras
import os
import matplotlib.pyplot as plt


class ModelBuild:
    """
    This class contains attributes useful for constructing transformer model along with
    training and providing useful stats.
    """
    def __init__(self, sequence_len, no_features, dense_dim, num_heads, classes, no_epochs):
        self.sequence_length = sequence_len
        self.embed_dim = no_features
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.classes = classes
        self.epochs = no_epochs
        self.path = None
        self.history = None
        self.compiled_model = None

    def constructModel(self):
        """
        Builds transformer model consisting of input to positional embedding layer, transformer encoder layer
        and softmax layer. Also prints the summary of the model

        :return: returns constructed model object
        """
        inputs = keras.Input(shape=(None, None))
        obj = PositionalEmbedding(self.sequence_length, self.embed_dim)(inputs)
        obj = TransformerEncoder(self.embed_dim, self.dense_dim, self.num_heads)(obj)
        obj = layers.GlobalMaxPooling1D()(obj)
        obj = layers.Dropout(0.7)(obj)
        outputs = layers.Dense(self.classes, activation="softmax")(obj)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
        print(model.summary())
        return model

    def fitTransform(self, classtype, Xdata, Ydata):
        """
        Constructs the model as well as trains the constructed model for the input data.
        Also, saves the model weights in particular naming convention.
        Naming convention of weight file saved: class type + number of neurons in dense layer
        + number of attention heads + number of classes inside classtype + number of epochs

        :param classtype: type of class being trained on. Eg: Word class or Number class
        :param Xdata: training data classified as X train usually all the independent features
        :param Ydata: training data classified as Y train usually the dependent features
        :return: returns the trained model for further use.
        """
        self.path = os.path.join(os.getcwd(), classtype + str(self.dense_dim) + '_' + str(self.num_heads) + '_'
                                 +str(self.classes) + '_' + str(self.epochs) + 'weights.hdf5')
        checkpoint = keras.callbacks.ModelCheckpoint(self.path, save_weight_only=True, save_best_only=True, verbose=1)
        self.compiled_model = self.constructModel()
        self.history = self.compiled_model.fit(Xdata, Ydata, validation_split = 0.33, epochs=self.epochs,
                                               callbacks=[checkpoint])
        return self.compiled_model

    def stats(self, data, labels):
        """
        Function plots train v/s validation loss graph to understand the
        training of the model. Also, prints out the test accuracy score for the model.

        :param data: test data
        :param labels: test labels
        :return:None
        """
        plt.figure(figsize=(10, 10))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        self.compiled_model.load_weights(self.path)
        _, accuracy = self.compiled_model.evaluate(data, labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")





