from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


class PositionalEmbedding(layers.Layer):
    """
    Defining positional embedding layers from keras.embedding
    This is done as it is important to provide order of sequences
    of the video frame

        this code is based from https://github.com/keras-team/keras-io/blob/master/examples/nlp/neural_machine_translation_with_transformer.py


    """
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[1]  # The inputs are of shape: `(batch_size, frames, num_features)`
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

    def get_config(self):
        config = super().get_config().copy()
        config.update({ "sequence_length": self.sequence_length,
                         "output_dim": self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerEncoder(layers.Layer):
    """
    Defining encoder layer of the transformer, this layer
    contains the attention module along with the feed forward dense
    layers

    this code is based from https://github.com/keras-team/keras-io/blob/master/examples/nlp/neural_machine_translation_with_transformer.py

    """
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"embed_dim": self.embed_dim,
                        "dense_dim": self.dense_dim,
                        "num_heads": self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
