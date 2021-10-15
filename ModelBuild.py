import TransformerEncoder as te
import os
import matplotlib.pyplot as plt


class ModelBuild:
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
        inputs = te.keras.Input(shape=(None, None))
        obj = te.PositionalEmbedding(self.sequence_length, self.embed_dim, name = "frame_position_embedding")(inputs)
        obj = te.TransformerEncoder(self.embed_dim, self.dense_dim, self.num_heads, name= "transformer_encode_layer")(obj)
        obj = te.layers.GlobalMaxPooling1D()(obj)
        obj = te.layers.Dropout(0.8)(obj)
        outputs = te.layers.Dense(self.classes, activation="softmax")(obj)
        model = te.keras.Model(inputs, outputs)

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
        print(model.summary())
        return model

    def fitTransform(self, Xdata, Ydata):
        curr_dir = os.getcwd()
        self.path = os.path.join(curr_dir,'word_weights.hdf5')
        checkpoint = te.keras.callbacks.ModelCheckpoint(self.path, save_weight_only=True, save_best_only=True, verbose=1)
        self.compiled_model = self.constructModel()
        self.history = self.compiled_model.fit(Xdata, Ydata, validation_split = 0.33, epochs=self.epochs,
                                               callbacks=[checkpoint])
        return self.compiled_model


    def stats(self, data, labels):
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





