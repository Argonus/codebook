"""DensNet 121 model implementation using TensorFlow with bottleneck layers."""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

from src.model.densnet.tensorflow_model_blocks import conv_block, transition_layer, squeeze_excitation, spatial_attention


def build_densenet121(num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3), use_se: bool = True, use_sa: bool = True) -> Model:
    """
    Build DenseNet121 architecture from scratch with bottleneck layers.
    
    :params num_classes: Number of output classes
    :params input_shape: Input image shape (height, width, channels)
    :params use_se: Whether to use Squeeze-and-Excitation blocks
    :return: Model: DenseNet-121 model
    """
    inputs = Input(shape=input_shape, name="input_layer")

    # Initial convolution
    x = conv_block(inputs, filters=64, kernel_size=(7,7), strides=2, padding="same")
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

    # Dense blocks with transition layers
    x = dense_block(x, num_layers=6, growth_rate=32, use_se=use_se, use_sa=use_sa)
    x = transition_layer(x)

    x = dense_block(x, num_layers=12, growth_rate=32, use_se=use_se, use_sa=use_sa)
    x = transition_layer(x)

    x = dense_block(x, num_layers=24, growth_rate=32, use_se=use_se, use_sa=use_sa)
    x = transition_layer(x)

    x = dense_block(x, num_layers=16, growth_rate=32, use_se=use_se, use_sa=use_sa)
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation="sigmoid", name="output_layer")(x)
    model = Model(inputs=inputs, outputs=outputs, name="DensNet121")

    return model

def dense_block(x: tf.Tensor, num_layers: int, growth_rate: int, use_se: bool, use_sa: bool) -> tf.Tensor:
    """
    Dense block that implements the core feature of DenseNet architecture with bottleneck optimization.

    Operations:
    1. Feature concatenation - combines all previous layers' outputs
    2. Bottleneck layer - 1x1 convolution to reduce input channel dimensions 
    3. Core convolution - 3x3 convolution to extract new features
    4. Dense connectivity - each layer receives feature maps from all preceding layers

    The dense block serves several critical purposes:
    - Feature reuse: Maximizes information flow by connecting each layer to every other layer
    - Gradient flow: Direct connections to all previous layers mitigate the vanishing gradient problem
    - Parameter efficiency: Growth rate controls the amount of new information each layer contributes
    - Regularization effect: Dense connectivity pattern provides implicit deep supervision

    Bottleneck layers (1x1 convolutions) significantly reduce the computation by first compressing 
    the input feature maps before the more expensive 3x3 convolutions.

    :param x: Input tensor
    :param num_layers: Number of layers in the dense block
    :param growth_rate: Number of filters added by each layer (controls network width)
    :param use_se: Whether to use Squeeze-and-Excitation blocks
    :return: Concatenated feature maps from all layers in the block
    """
    features = [x]
    for _ in range(num_layers):
        current_input = Concatenate()(features) if len(features) > 1 else x
        new_features = conv_block(current_input, 4 * growth_rate, kernel_size=(1,1), strides=1)
        new_features = conv_block(new_features, growth_rate, kernel_size=(3,3))

        if use_se:
            new_features = squeeze_excitation(new_features, ratio=16)

        if use_sa:
            new_features = spatial_attention(new_features, kernel_size=(3,3))

        features.append(new_features)

    return Concatenate()(features)
