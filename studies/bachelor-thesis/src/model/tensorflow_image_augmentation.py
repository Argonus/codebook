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
    image = _apply_augmentation(image, _random_intensity_scaling, probability=probability.get("intensity_scaling", 0.1))
    image = _apply_augmentation(image, _adaptive_histogram_equalization, probability=probability.get("adaptive_histogram", 0.1))

    image = _apply_augmentation(image, _random_zoom, probability=probability.get("zoom", 0.3))
    image = _apply_augmentation(image, _random_shifting, probability=probability.get("shifting", 0.2))
    image = _apply_augmentation(image, _random_rotation, probability=probability.get("rotation", 0.2))
    
    image = _apply_augmentation(image, _gaussian_noise, probability=probability.get("gaussian_noise", 0.1))
    image = _apply_augmentation(image, _random_cutout, probability=probability.get("cutout", 0.1))
    
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
def _random_rotation(image: tf.Tensor, max_angle: float = 5.0) -> tf.Tensor:
    """
    Apply small random rotation to simulate patient positioning variations
    Mimics: Variations in scanner angle or patient position
    """
    angle = tf.random.uniform([], -max_angle, max_angle) * (3.14159/180)
    cos_angle = tf.cos(angle)
    sin_angle = tf.sin(angle)

    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    
    cx = width / 2
    cy = height / 2
    
    tx = cx - cx * cos_angle + cy * sin_angle
    ty = cy - cx * sin_angle - cy * cos_angle
    
    transforms = tf.stack([
        cos_angle, sin_angle, tx,
        -sin_angle, cos_angle, ty,
        0.0, 0.0
    ])
    
    transforms = tf.reshape(transforms, [1, 8])
    image = tf.expand_dims(image, 0)
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transforms,
        output_shape=tf.shape(image)[1:3],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0
    )
    
    return tf.squeeze(transformed, 0)

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
    
    shape = tf.shape(image) 
    noise = tf.random.normal(shape=shape, mean=0., stddev=stddev)
    return tf.clip_by_value(image + noise, 0, 1)

@tf.function
def _random_intensity_scaling(image: tf.Tensor, scale_range=(0.75, 1.25), shift_range=(-0.1, 0.1)) -> tf.Tensor:
    """
    Apply random intensity scaling and shifting to simulate different X-ray exposures.
    Mimics: Different radiation dose, detector sensitivity, processing algorithms.
    """
    scale = tf.random.uniform([], scale_range[0], scale_range[1])
    shift = tf.random.uniform([], shift_range[0], shift_range[1])
    image = image * scale + shift
    
    return tf.clip_by_value(image, 0.0, 1.0)

@tf.function
def _adaptive_histogram_equalization(image: tf.Tensor) -> tf.Tensor:
    """
    Apply contrast stretching with more visible effect.
    Mimics: Different X-ray processing settings.
    """
    p_low = 0.02
    p_high = 0.98
    
    flat_img = tf.reshape(image, [-1])
    sorted_img = tf.sort(flat_img)
    n_pixels = tf.shape(sorted_img)[0]
    
    idx_low = tf.cast(tf.cast(n_pixels, tf.float32) * p_low, tf.int32)
    idx_high = tf.cast(tf.cast(n_pixels, tf.float32) * p_high, tf.int32)
    
    low = sorted_img[idx_low]
    high = sorted_img[idx_high]
    
    scaled = (image - low) / (high - low + 1e-5)
    
    return tf.clip_by_value(scaled, 0.0, 1.0)

@tf.function
def _random_zoom(image: tf.Tensor, zoom_range=(0.85, 1.15)) -> tf.Tensor:
    """
    Apply random zoom to simulate variations in patient distance from scanner.
    Mimics: Different X-ray focus distances.
    """
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    
    zoom = tf.random.uniform([], zoom_range[0], zoom_range[1])
    
    new_height = tf.cast(tf.cast(height, tf.float32) * zoom, tf.int32)
    new_width = tf.cast(tf.cast(width, tf.float32) * zoom, tf.int32)
    
    if zoom > 1.0:
        y_start = (new_height - height) // 2
        x_start = (new_width - width) // 2
        image = tf.image.resize(image, [new_height, new_width])
        image = tf.image.crop_to_bounding_box(image, y_start, x_start, height, width)
    else:
        image = tf.image.resize(image, [new_height, new_width])
        y_pad = (height - new_height) // 2
        x_pad = (width - new_width) // 2
        image = tf.image.pad_to_bounding_box(
            image, y_pad, x_pad, height, width)
    
    return image

@tf.function
def _random_cutout(image: tf.Tensor, mask_size_range=(10, 20), min_cutouts=1, max_cutouts=2) -> tf.Tensor:
    """
    Randomly mask out rectangular regions of the image.
    Mimics: Purely technical variation. Forces model to look at broader context instead of specific small features.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    num_cutouts = tf.random.uniform([], min_cutouts, max_cutouts, dtype=tf.int32)    
    result = tf.identity(image)
    
    for _ in range(num_cutouts):
        mask_size = tf.random.uniform([], mask_size_range[0], mask_size_range[1], dtype=tf.int32)
        
        x = tf.random.uniform([], 0, width - mask_size, dtype=tf.int32)
        y = tf.random.uniform([], 0, height - mask_size, dtype=tf.int32)
        
        mask_value = tf.reduce_mean(image)
        mask_patch = tf.ones([mask_size, mask_size, shape[2]]) * mask_value
        
        y_indices = tf.range(y, y + mask_size)
        x_indices = tf.range(x, x + mask_size)
        grid_y, grid_x = tf.meshgrid(y_indices, x_indices, indexing='ij')
        
        indices = tf.stack([grid_y, grid_x], axis=-1)
        indices = tf.reshape(indices, [-1, 2])
        
        updates = tf.reshape(mask_patch, [-1, shape[2]])
        result = tf.tensor_scatter_nd_update(result, indices, updates)
    
    return tf.clip_by_value(result, 0, 1)