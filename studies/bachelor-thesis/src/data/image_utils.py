import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def get_image_array(image_path: str) -> np.ndarray:
    """Returns an image as a NumPy array in grayscale."""
    grayscale_img = Image.open(image_path).convert("L")

    return np.array(grayscale_img)

def get_image_bytes(image_path: str) -> bytes:
    """Returns the raw bytes of an image file."""
    with open(image_path, 'rb') as f:
        return f.read()

def calculate_ssim(original_image: np.ndarray, reference_image: np.ndarray) -> float:
    """Calculate SSIM value of image, based on reference image"""
    if reference_image.shape != original_image.shape:
        original_image = resize_image(original_image, reference_image)
    ssim_value, _ = ssim(reference_image, original_image, full=True)

    return ssim_value

def resize_image(original_image: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
    """Resize original image to reference image shape"""
    shape_zero = ref_image.shape[0]
    shape_one  = ref_image.shape[1]
    resized_image = Image.fromarray(original_image).resize((shape_one, shape_zero), Image.LANCZOS)

    return np.array(resized_image)