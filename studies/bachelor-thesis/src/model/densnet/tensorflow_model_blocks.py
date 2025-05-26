"""DensNet model blocks implementation using TensorFlow."""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, GlobalAveragePooling2D, Dense, Reshape, multiply, Concatenate, Lambda, Activation, Multiply
from tensorflow.keras.regularizers import l2

def conv_block(x: tf.Tensor, filters: int,  kernel_size: Tuple[int, int] = (3,3), strides: int = 1, padding: str = "same") -> tf.Tensor:
    """
    Convolutional block used as a core building block in DenseNet architecture.
    Operations:
    1. Batch Normalization - normalizes activations for better gradient flow
    2. ReLU - applies non-linearity to the normalized features
    3. Convolution - extracts features using learned filters
    
    The conv_block implements the BN-ReLU-Conv pre-activation pattern which:
    - Improves gradient flow through the network
    - Provides regularization effect
    - Enhances feature extraction capabilities
    
    :param x: Input tensor
    :param filters: Number of output filters/channels
    :param kernel_size: Size of the convolutional kernel (height, width)
    :param strides: Stride of the convolution operation
    :param padding: Padding type for the convolution ('same' or 'valid')
    :return: Tensor with extracted features
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

def transition_layer(x: tf.Tensor, compression: float = 0.5) -> tf.Tensor:
    """
    Transition layer that reduces feature map size between dense blocks in DenseNet.
    Operations:
    1. Channel reduction - uses compression factor to reduce number of channels
    2. 1x1 Convolution - applies a 1x1 conv to reduce channel dimensions efficiently
    3. Average Pooling - downsamples spatial dimensions by factor of 2
    
    The transition layer serves two important purposes:
    - Controls model growth: DenseNet architectures grow in parameters rapidly due to
      concatenation of feature maps. Transition layers apply compression to reduce this growth.
    - Reduces spatial dimensions: Halves the height and width, reducing computation in
      subsequent layers.
    
    :param x: Input tensor from a dense block
    :param compression: Factor to reduce number of feature maps (0-1), also known as theta in the paper
    :return: Tensor with reduced spatial and channel dimensions
    """
    in_channels = tf.keras.backend.int_shape(x)[-1]
    reduced_channels = int(in_channels * compression)
    x = conv_block(x, reduced_channels, kernel_size=(1,1))
    x = AveragePooling2D(pool_size=(2,2), strides=2, padding="same")(x) 
    return x

def squeeze_excitation(x: tf.Tensor, ratio: int = 16) -> tf.Tensor:
    """
    Squeeze-and-Excitation block that models channel-wise interdependencies.
    
    Operations:
    1. Global average pooling - to reduce the spatial dimensions.
    2. Dense One - reduction of the number of channels. Used to reduce the number of parameters.
    3. Dense Two - expansion of the number of channels. Used to produce channel recalibration weights.
    4. Reshape - to match the input shape.
    5. Scale - multiplication of the input with the recalibration weights.
    
    The SE block serves several important purposes:
    - Channel attention: Enables the network to focus on the most informative features by
      dynamically recalibrating channel-wise feature responses.
    - Feature interdependencies: Explicitly models relationships between channels that
      standard convolutions might not efficiently capture.
    - Adaptability: Allows the network to adjust its focus based on the input, which is
      particularly valuable for medical imaging where features of interest may vary in importance.
    
    :param x: Input tensor
    :param ratio: Reduction ratio for the bottleneck in the channel excitation
    :return: Tensor with the same shape as input but with channel-wise recalibration
    """
    channels = x.shape[-1]    
    squeeze = GlobalAveragePooling2D()(x)
    
    excitation = Dense(channels // ratio, activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, channels))(excitation)
    
    output = multiply([x, excitation])
    return output

def spatial_attention(x: tf.Tensor, kernel_size: Tuple[int, int] = (7,7)) -> tf.Tensor:
    """
    Implements spatial attention for X-ray feature maps.
    """
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    
    attention = Conv2D(filters=1,
                      kernel_size=kernel_size,
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(1e-4),
                      kernel_initializer='he_normal')(concat)
    attention = Activation('sigmoid')(attention)
    
    return multiply([x, attention])