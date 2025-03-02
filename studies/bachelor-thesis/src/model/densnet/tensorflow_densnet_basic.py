"""Basic implementation of DenseNet121 architecture."""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

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

def conv_block(x: tf.Tensor, 
             filters: int, 
             kernel_size: Tuple[int, int] = (3,3), 
             strides: int = 1, 
             padding: str = "same") -> tf.Tensor:
    """
    Convolutional block, that consists of a convolutional layer, batch normalization and ReLU activation.
    This is a building block responsible for extracting features from the input image.
    """
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(1e-4),
        kernel_initializer='he_normal')(x)
    
    return x

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

def transition_layer(x: tf.Tensor, compression: float = 0.5) -> tf.Tensor:
    """
    Transition layer, that consists of a convolutional layer, batch normalization and ReLU activation.
    This is a building block responsible for reducing the number of channels after each dense block.
    """
    filters = int(tf.keras.backend.int_shape(x)[-1] * compression)
    x = conv_block(x, filters, kernel_size=(1,1))
    x = AveragePooling2D(pool_size=(2,2), strides=2, padding="same")(x) 
    return x