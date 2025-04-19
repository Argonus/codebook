import os
import numpy as np

from src.data.image_utils import get_image_array, get_image_bytes, calculate_ssim, resize_image

def test_get_image_array():
    dirname = os.path.dirname(__file__)
    image_one = f"{dirname}/../fixtures/image_one.png"
    result = get_image_array(image_one)

    assert isinstance(result, np.ndarray)

def test_get_image_bytes():
    dirname = os.path.dirname(__file__)
    image_one = f"{dirname}/../fixtures/image_one.png"
    result = get_image_bytes(image_one)

    assert isinstance(result, bytes)

def test_calculate_ssim_different_images():
    dirname = os.path.dirname(__file__)
    image_one = f"{dirname}/../fixtures/image_one.png"
    image_two = f"{dirname}/../fixtures/image_two.png"
    result = calculate_ssim(get_image_array(image_one), get_image_array(image_two))

    assert isinstance(result, float)
    assert round(result, 2) == 0.89

def test_calculate_ssim_same_images():
    dirname = os.path.dirname(__file__)
    image_one = f"{dirname}/../fixtures/image_one.png"
    image_two = f"{dirname}/../fixtures/image_one.png"
    result = calculate_ssim(get_image_array(image_one), get_image_array(image_two))

    assert isinstance(result, float)
    assert round(result, 2) == 1.00

def test_resize_image():
    dirname = os.path.dirname(__file__)
    image_one = get_image_array(f"{dirname}/../fixtures/image_one.png")
    image_two = get_image_array(f"{dirname}/../fixtures/image_two.png")
    result = resize_image(image_one, image_two)

    assert isinstance(result, np.ndarray)
    assert result.shape == image_two.shape
