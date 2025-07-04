"""Basic implementation of DenseNet121 architecture."""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

from src.model.densnet.tensorflow_model_blocks import conv_block, transition_layer

def build_densenet(num_classes: int) -> Model:
    """
    Build DenseNet121 architecture from scratch.
    """
    inputs = Input(shape=(224, 224, 3), name="input_layer")

    x = conv_block(inputs, filters=64, kernel_size=(7,7), strides=2, padding="same")
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)

    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_layer(x)                          

    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_layer(x)                           

    x = dense_block(x, num_layers=24, growth_rate=32)
    x = transition_layer(x)                         

    x = dense_block(x, num_layers=16, growth_rate=32)
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="sigmoid", name="output_layer")(x)
    model = Model(inputs=inputs, outputs=outputs, name="BasicDensNet121")

    return model

def dense_block(x: tf.Tensor, num_layers: int, growth_rate: int) -> tf.Tensor:
    """
    Dense block, that consists of a concatenation of multiple convolutional blocks.
    This is a building block responsible for extracting features from the input image.
    """
    features = [x]
    for _ in range(num_layers):
        current_input = Concatenate()(features) if len(features) > 1 else x
        new_features = conv_block(current_input, growth_rate)
        features.append(new_features)
    
    return Concatenate()(features)