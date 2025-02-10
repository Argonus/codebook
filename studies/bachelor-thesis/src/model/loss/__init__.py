"""
Source package for chest_x_ray_diseases_detection.
This allows pytest to recognize the `src/model` directory as a package.
"""
import sys
import os

# Register custom loss
from tensorflow.keras.utils import get_custom_objects
from .tensorflow_no_finding_binary_crossentropy import NoFindingBinaryCrossentropy
get_custom_objects().update({'NoFindingBinaryCrossentropy': NoFindingBinaryCrossentropy})

# Ensure `src/model` is in the Python path for importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "model", "loss")))