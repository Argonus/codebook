"""
Project-wide constants and configuration values.

This module contains all the constant values used throughout the project,
including paths, model parameters, and dataset configurations.
"""

from typing import List, Tuple
import os

# Path Configuration
# -----------------
CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
DATASETS_PATH: str = os.path.join(PROJECT_ROOT, "datasets")
"""Path to the datasets directory in project root"""

# Dataset Paths
# ------------
RAW_DATASET: str = os.path.join(DATASETS_PATH, 'raw-data', 'nih-dataset')
"""Path to the original NIH dataset downloaded from Kaggle"""

CLEARED_DATASET: str = os.path.join(DATASETS_PATH, 'cleared-data', 'nih-dataset')
"""Path to the filtered and preprocessed dataset"""

TF_RECORD_DATASET: str = os.path.join(DATASETS_PATH, 'tfrecord-dataset', 'nih-dataset')
"""Path to the TFRecord format dataset ready for model training"""

# TensorFlow Configuration
# ----------------------
TF_BUFFER_SIZE: int = 262144
"""Buffer size for TFRecord reading, optimized for performance"""

# ImageNet Normalization Parameters
# ------------------------------
DENSENET_IMAGE_SIZE: Tuple[int, int] = (224, 224)
"""Required input image dimensions for DenseNet model (height, width)"""

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
"""ImageNet mean values for RGB channels, used for image normalization"""

IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
"""ImageNet standard deviation values for RGB channels, used for image normalization"""
