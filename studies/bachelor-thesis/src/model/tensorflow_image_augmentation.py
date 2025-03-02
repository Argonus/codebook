"""
Module containing image augmentation functionality that may be shared between model runs
"""
import tensorflow as tf


@tf.function
def augment_xray(image: tf.Tensor, probability: dict = {}) -> tf.Tensor:
    """
    Applies medical-safe augmentations to a chest X-ray image.
    """
    tf.debugging.assert_rank(image, 3, message="Input image must be a single image with shape [height, width, channels]")
    
    # Cast to float32
    image = tf.cast(image, tf.float32)
    
    # Apply augmentations
    image = _apply_augmentation(image, _random_brightness, probability=probability.get("brightness", 0.5))
    image = _apply_augmentation(image, _random_contrast, probability=probability.get("contrast", 0.5))
    image = _apply_augmentation(image, _random_shifting, probability=probability.get("shifting", 0.1))
    image = _apply_augmentation(image, _gaussian_noise, probability=probability.get("gaussian_noise", 0.05))
    
    tf.debugging.assert_rank(image, 3, message="Output image must be a single image with shape [height, width, channels]")
    return image
@tf.function
def _apply_augmentation(image: tf.Tensor, augmentation_func: callable, probability=0.5, *args, **kwargs) -> tf.Tensor:
    """
    Applies an augmentation function to an image with a given probability.
    :param image: A 3-D tensor of shape [height, width, channels]
    :param augmentation_func: Function to apply augmentation
    :param probability: Probability of applying the augmentation
    :params *args, **kwargs: Additional arguments to pass to augmentation_func
    :return: A 3-D tensor of shape [height, width, channels]
    """    
    return tf.cond(
        tf.random.uniform([], 0, 1) < probability,
        lambda: augmentation_func(image, *args, **kwargs),
        lambda: image
    )

@tf.function
def _random_brightness(image: tf.Tensor, max_delta: float = 0.2) -> tf.Tensor:
    """
    Apply random brightness adjustment.
    Mimics: Variations in X-ray exposure settings or tissue absorption
    """
    return tf.image.random_brightness(image, max_delta)

@tf.function
def _random_contrast(image: tf.Tensor, lower: float = 0.5, upper: float = 1.5) -> tf.Tensor:
    """
    Apply random contrast adjustment.
    Mimics: Differences in scanner contrast settings or patient body composition
    """
    return tf.image.random_contrast(image, lower, upper)

@tf.function
def _random_shifting(image: tf.Tensor, minval: int = -10, maxval: int = 10) -> tf.Tensor:
    """
    Apply random shifting to a single image.
    Mimics: Differences in scanner position or patient position
    """
    
    image = tf.expand_dims(image, 0)
    dx = tf.random.uniform([], minval=minval, maxval=maxval, dtype=tf.float32)
    dy = tf.random.uniform([], minval=minval, maxval=maxval, dtype=tf.float32)
    
    transforms = tf.stack([
        tf.ones_like(dx),
        tf.zeros_like(dx),
        dx,
        tf.zeros_like(dy),
        tf.ones_like(dy),
        dy,
        tf.zeros_like(dx),
        tf.zeros_like(dx)
    ])
    transforms = tf.reshape(transforms, [1, 8])
    input_shape = tf.shape(image)[1:3]
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transforms,
        output_shape=input_shape,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0
    )
    
    return tf.squeeze(transformed, 0)

@tf.function
def _gaussian_noise(image: tf.Tensor, stddev: float = 0.2) -> tf.Tensor:
    """
    Apply Gaussian noise.
    Mimics: Noise in the X-ray image
    """
    
    shape = image.shape
    if shape.rank is None:
        shape = tf.shape(image)
    
    noise = tf.random.normal(shape=shape, mean=0., stddev=stddev)
    return tf.clip_by_value(image + noise, 0, 1)