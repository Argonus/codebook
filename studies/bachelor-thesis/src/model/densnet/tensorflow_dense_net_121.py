"""DensNet 121 model implementation using TensorFlow with bottleneck layers."""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_densenet121(num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Model:
    """
    Build DenseNet121 architecture from scratch with bottleneck layers.
    
    :params num_classes: Number of output classes
    :params input_shape: Input image shape (height, width, channels)
    :return: Model: DenseNet-121 model
    """
    inputs = Input(shape=input_shape, name="input_layer")

    # Initial convolution
    x = BatchNormalization()(inputs)
    x = ReLU()(x)
    x = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(1e-4),
        kernel_initializer='he_normal'
    )(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # Dense blocks with transition layers
    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_layer(x)

    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_layer(x)

    x = dense_block(x, num_layers=24, growth_rate=32)
    x = transition_layer(x)

    x = dense_block(x, num_layers=16, growth_rate=32)

    # Classification layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation="sigmoid", name="output_layer")(x)
    model = Model(inputs=inputs, outputs=outputs, name="DensNet121")

    return model

def bottleneck_layer(x: tf.Tensor, growth_rate: int) -> tf.Tensor:
    """
    Bottleneck layer with BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3) pattern.
    :param x: Input tensor
    :param growth_rate: Growth rate for the dense block
    :return: tf.Tensor: Output tensor after bottleneck transformation
    """
    # First composite function (1x1 conv)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=4 * growth_rate,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(1e-4),
        kernel_initializer='he_normal'
    )(x)

    # Second composite function (3x3 conv)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=growth_rate,
        kernel_size=(3, 3),
        strides=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(1e-4),
        kernel_initializer='he_normal'
    )(x)

    return x

def dense_block(x: tf.Tensor, num_layers: int, growth_rate: int) -> tf.Tensor:
    """
    Dense block with bottleneck layers and dense connectivity pattern.
    
    :param x: Input tensor
    :param num_layers: Number of layers in the dense block
    :param growth_rate: Growth rate for feature maps
    :return: tf.Tensor: Concatenated feature maps
    """
    features = [x]
    for _ in range(num_layers):
        current_input = Concatenate()(features) if len(features) > 1 else x
        new_features = bottleneck_layer(current_input, growth_rate)
        features.append(new_features)

    return Concatenate()(features)

def transition_layer(x: tf.Tensor, compression: float = 0.5) -> tf.Tensor:
    """
    Transition layer with compression to reduce feature maps.
    
    :param x: Input tensor
    :param compression: Compression factor for reducing feature maps
    :return: tf.Tensor: Compressed and pooled feature maps
    """
    filters = int(tf.keras.backend.int_shape(x)[-1] * compression)
    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(1e-4),
        kernel_initializer='he_normal'
    )(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same")(x)
    
    return x