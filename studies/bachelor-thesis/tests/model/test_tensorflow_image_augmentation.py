"""
Unit tests for the tensorflow_image_augmentation module.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.model.tensorflow_image_augmentation import (
    augment_xray,
    _random_brightness,
    _random_contrast,
    _random_shifting,
    _gaussian_noise
)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return tf.convert_to_tensor(np.ones((50, 50, 1), dtype=np.float32) * 0.5)


@pytest.fixture
def non_uniform_image():
    """Create a non-uniform image with a distinct feature for testing."""
    image = np.zeros((50, 50, 1), dtype=np.float32)
    image[20:30, 20:30, 0] = 1.0
    return tf.convert_to_tensor(image)


# Augment X-ray Image
def test_augment_xray_shape_preservation(sample_image):
    """Test that augment_xray preserves image shape."""
    augmented = augment_xray(sample_image)
    assert augmented.shape == sample_image.shape


def test_augment_xray_value_range(sample_image):
    """Test that augment_xray maintains reasonable pixel values."""
    augmented = augment_xray(sample_image)

    assert tf.reduce_min(augmented) >= 0.0
    assert tf.reduce_max(augmented) <= 1.0


def test_augment_xray_with_zero_probabilities(sample_image):
    """Test that augment_xray respects custom probabilities."""
    custom_probs = {
        "brightness": 0.0,
        "contrast": 0.0,
        "shifting": 0.0,
        "gaussian_noise": 0.0
    }

    augmented = augment_xray(sample_image, probability=custom_probs)
    assert tf.reduce_all(tf.equal(augmented, sample_image))
    

def test_augment_xray_with_one_probabilities(sample_image):
    """Test that augment_xray respects custom probabilities."""
    custom_probs = {
        "brightness": 0.0,
        "contrast": 0.0,
        "shifting": 0.0,
        "gaussian_noise": 1.0
    }

    augmented = augment_xray(sample_image, probability=custom_probs)
    assert not tf.reduce_all(tf.equal(augmented, sample_image))


def test_random_brightness(sample_image):
    """Test that random brightness changes pixel values."""
    brightened = _random_brightness(sample_image, max_delta=0.2)
    
    assert brightened.shape == sample_image.shape
    assert not tf.reduce_all(tf.equal(brightened, sample_image))


def test_random_contrast_sample(sample_image):
    """Test that random contrast preserves shape on uniform images."""
    contrasted = _random_contrast(sample_image, lower=0.5, upper=1.5)
    
    assert contrasted.shape == sample_image.shape
    assert tf.reduce_all(tf.equal(contrasted, sample_image))
    
def test_random_contrast_non_uniform(non_uniform_image):
    """Test that random contrast changes pixel values."""
    contrasted_non_uniform = _random_contrast(non_uniform_image, lower=0.5, upper=1.5)

    assert contrasted_non_uniform.shape == non_uniform_image.shape
    assert not tf.reduce_all(tf.equal(contrasted_non_uniform, non_uniform_image))


def test_random_shifting(non_uniform_image):
    """Test that random shifting changes pixel positions."""
    shifted = _random_shifting(non_uniform_image, minval=5, maxval=5)  # Force 5px shift
    
    assert shifted.shape == non_uniform_image.shape    
    assert not tf.reduce_all(tf.equal(shifted, non_uniform_image))


def test_gaussian_noise(sample_image):
    """Test that Gaussian noise changes pixel values."""
    noisy = _gaussian_noise(sample_image, stddev=0.1)
    
    assert noisy.shape == sample_image.shape
    assert not tf.reduce_all(tf.equal(noisy, sample_image))    
    assert tf.reduce_min(noisy) >= 0.0
    assert tf.reduce_max(noisy) <= 1.0
