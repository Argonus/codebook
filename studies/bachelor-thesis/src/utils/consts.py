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
MODELS_PATH: str = os.path.join(PROJECT_ROOT, "models")
"""Path to the datasets directory in project root"""

# Dataset Paths
# ------------
RAW_DATASET: str = os.path.join(DATASETS_PATH, 'raw-data', 'nih-dataset')
"""Path to the original NIH dataset downloaded from Kaggle"""

CLEARED_DATASET: str = os.path.join(DATASETS_PATH, 'cleared-data', 'nih-dataset')
"""Path to the filtered and preprocessed dataset"""

TF_RECORD_DATASET: str = os.path.join(DATASETS_PATH, 'tfrecord-dataset', 'nih-dataset')
"""Path to the TFRecord format dataset ready for model training"""

# Dataset Metadata
# ----------------
NUM_CLASSES: int = 15
"""Number of classes in the dataset"""
DATASET_SIZE = 102697
"""Total number of samples in the dataset"""
NO_FINDING_CLASS_IDX: int = 10
"""Index of the No Finding class in the dataset"""
DROP_NO_FINDING_CLASS: bool = False
"""Whether to drop the No Finding class from the dataset"""
NUM_CLASSES_WITHOUT_NO_FINDING: int = NUM_CLASSES - 1
"""Number of classes in the dataset without the No Finding class"""

# TensorFlow Configuration
# ----------------------
TF_MAX_EPOCHS: int = 100
"""Maximum number of epochs for training, with early stopping monitoring validation metrics"""

TF_BUFFER_SIZE: int = 8 * 1024 * 1024  # 8MB
"""Buffer size for TFRecord reading, optimized for RTX 3090 Ti"""

TF_SHUFFLE_SIZE: int = 12500
"""Shuffle size for TFRecord shuffling, ~12% of dataset size for good randomization"""

TF_BATCH_SIZE: int = 128
"""Batch size optimized for RTX 3090 Ti memory and compute capability"""

# ImageNet Normalization Parameters
# ------------------------------
DENSENET_IMAGE_SIZE: Tuple[int, int] = (224, 224)
"""Required input image dimensions for DenseNet model (height, width)"""

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
"""ImageNet mean values for RGB channels, used for image normalization"""

IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
"""ImageNet standard deviation values for RGB channels, used for image normalization"""
