import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import numpy as np

from tqdm import tqdm
from typing import Union, List
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, F1Score, AUC

from src.utils.consts import DENSENET_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

from src.model.tensorflow_logger import TrainingLogger
from src.model.tensorflow_weight_monitor import WeightMonitor
from src.model.tensorflow_csv_metrics_logger import CSVMetricsLogger
from src.model.tensorflow_log_filter import LogFilter

# ------------------------------------------------------------------------------
# Tensorflow Features Utils
# ------------------------------------------------------------------------------
def to_bytes_feature(value: Union[str, bytes]) -> tf.train.Feature:
    """
    Converts a string or bytes to TensorFlow feature format.
    input: str or bytes
    output: tf.train.Feature
    """
    if isinstance(value, str):
        value = value.encode()
    elif not isinstance(value, bytes):
        raise ValueError(f"Expected str or bytes, got {type(value)}")

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_float_feature(value: Union[float, List[float]]) -> tf.train.Feature:
    """
    Converts a float or list of floats to TensorFlow feature format.
    input: float or list of floats
    output: tf.train.Feature
    """
    if isinstance(value, (float, int)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
    elif isinstance(value, list) and all(isinstance(v, (float, int)) for v in value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v) for v in value]))
    else:
        raise ValueError(f"Expected float or list of floats, got {type(value)}")

def to_int64_feature(value: Union[int, List[int]]) -> tf.train.Feature:
    """
    Converts an int or list of ints to TensorFlow feature format.
    input: int or list of ints
    output: tf.train.Feature
    """
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, list) and all(isinstance(v, int) for v in value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        raise ValueError(f"Expected int or list of ints, got {type(value)}")

# ------------------------------------------------------------------------------
# Tensorflow Dataset Utils
# ------------------------------------------------------------------------------
def parse_record(example_proto: bytes) -> (tf.Tensor, tf.Tensor):
    """
    Parses a record from the TFRecord dataset and converts labels to one-hot encoding.
    Returns:
        - image: tensor of shape (224, 224, 3)
        - labels: tensor of shape (15,) - one-hot encoded labels
    """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "encoded_finding_labels": tf.io.VarLenFeature(tf.int64)
    }

    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(parsed_features["image"], channels=1, dtype=tf.uint8)
    image = tf.image.grayscale_to_rgb(image)

    image = tf.image.resize(image, DENSENET_IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    imagenet_mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    imagenet_std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    image = (image - imagenet_mean) / imagenet_std

    label_indices = tf.sparse.to_dense(parsed_features["encoded_finding_labels"])
    one_hot = tf.reduce_max(tf.one_hot(label_indices, depth=NUM_CLASSES, dtype=tf.float32), axis=0)
    
    image = tf.ensure_shape(image, [224, 224, 3])
    one_hot = tf.ensure_shape(one_hot, [NUM_CLASSES])

    return image, one_hot

def load_and_split_dataset(record_path: str, shuffle_buffer_size: int = 1000, tfrecord_buffer_size: int = 262144, dataset_size: int = None)  -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    """
    Loads, splits, and optimizes a TFRecord dataset.
    """
    dataset = load_dataset(record_path, tfrecord_buffer_size)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.7, 0.15, shuffle_buffer_size, dataset_size)
    return train_dataset, val_dataset, test_dataset

def load_dataset(record_path: str, tfrecord_buffer_size: int) -> tf.data.TFRecordDataset:
    """
    Loads a TFRecord dataset
    """
    dataset = tf.data.TFRecordDataset([record_path], buffer_size=tfrecord_buffer_size)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def split_dataset(dataset, train_ratio: float, val_ratio: float, shuffle_buffer_size: int, dataset_size: int) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    """
    Splits the dataset into training, validation, and test sets based on given ratios.
    Datasets will be shuffled, to ensure that all splits will contain mix of samples
    """
    # Compute dataset size if not provided
    dataset_size = count_dataset_size(dataset, dataset_size)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)

    # Shuffle before splitting
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset

def count_dataset_size(dataset: tf.data.TFRecordDataset, dataset_size: int) -> int:
    """
    Counts the number of elements in the dataset with a progress bar.
    """
    if dataset_size is not None:
        return dataset_size

    dataset_size = sum(1 for _ in tqdm(dataset, desc="Counting samples", unit=" samples"))
    return dataset_size


def apply_augmentation_to_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Applies augmentation to a dataset.
    Each sample should have shape ([224, 224, 3], [NUM_CLASSES]).
    """
    return dataset.map(
        lambda x, y: (augment_xray(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

def optimize_dataset(dataset, batch_size: int):
    """
    Applies batching and prefetching to optimize dataset performance.
    Each sample should have shape ([224, 224, 3], [NUM_CLASSES]) after parse_record.
    """
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------------------
# Tensorflow Image Transform Utils
# ------------------------------------------------------------------------------
@tf.function
def apply_augmentation(image: tf.Tensor, augmentation_func: callable, probability=0.5, *args, **kwargs) -> tf.Tensor:
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
def random_brightness(image: tf.Tensor, max_delta: float = 0.2) -> tf.Tensor:
    """
    Apply random brightness adjustment.
    Mimics: Variations in X-ray exposure settings or tissue absorption
    """
    return tf.image.random_brightness(image, max_delta)

@tf.function
def random_contrast(image: tf.Tensor, lower: float = 0.5, upper: float = 1.5) -> tf.Tensor:
    """
    Apply random contrast adjustment.
    Mimics: Differences in scanner contrast settings or patient body composition
    """
    return tf.image.random_contrast(image, lower, upper)

@tf.function
def random_shifting(image: tf.Tensor, minval: int = -10, maxval: int = 10) -> tf.Tensor:
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
def gaussian_noise(image: tf.Tensor, stddev: float = 0.2) -> tf.Tensor:
    """
    Apply Gaussian noise.
    Mimics: Noise in the X-ray image
    """
    
    shape = image.shape
    if shape.rank is None:
        shape = tf.shape(image)
    
    noise = tf.random.normal(shape=shape, mean=0., stddev=stddev)
    return tf.clip_by_value(image + noise, 0, 1)

@tf.function
def augment_xray(image: tf.Tensor) -> tf.Tensor:
    """
    Applies medical-safe augmentations to a chest X-ray image.
    """
    tf.debugging.assert_rank(image, 3, message="Input image must be a single image with shape [height, width, channels]")
    
    # Cast to float32
    image = tf.cast(image, tf.float32)
    
    # Apply augmentations
    image = apply_augmentation(image, random_brightness, probability=0.5)
    image = apply_augmentation(image, random_contrast, probability=0.5)
    image = apply_augmentation(image, random_shifting, probability=0.1)
    image = apply_augmentation(image, gaussian_noise, probability=0.05)
    
    tf.debugging.assert_rank(image, 3, message="Output image must be a single image with shape [height, width, channels]")
    return image

# ------------------------------------------------------------------------------
# Statistics calculations
# ------------------------------------------------------------------------------
def calculate_class_weights(dataset: tf.data.Dataset, num_classes: int) -> dict:
    """
    Calculate balanced class weights for your dataset.
    Works with both batched and unbatched datasets.
    :param dataset: TensorFlow dataset containing (image, labels) pairs
    :param num_classes: Number of classes in the dataset
    :return: A dictionary mapping class indices to their respective weights
    """
    labels_dataset = dataset.map(lambda x, y: y)
    total_samples = 0
    class_counts = np.zeros(num_classes)

    for labels in labels_dataset:
        if len(labels.shape) == 1:
            total_samples += 1
            class_counts += labels.numpy()
        else:
            total_samples += labels.shape[0]
            class_counts += tf.reduce_sum(labels, axis=0).numpy()

    class_counts = np.maximum(class_counts, 1)
    weights = (total_samples / (num_classes * class_counts)).astype(np.float32)
    
    return dict(enumerate(weights))

def show_class_weights(class_weights: dict) -> None:
    print("Class Weights:")
    for class_idx, weight in class_weights.items():
        print(f"Class {class_idx}: {weight:.2f}")

    return None

# ------------------------------------------------------------------------------
# Visualisation Utils
# ------------------------------------------------------------------------------
def show_sample_record(dataset: tf.data.Dataset):
    """
    Display a single sample from the dataset, showing both the X-ray image
    and its corresponding disease labels.
    
    Args:
        dataset: TensorFlow dataset containing (image, labels) pairs
    """
    _image, labels = extract_first_record(dataset)

    print('\nOne-hot encoded labels:')
    print(labels.numpy())

def show_augmented_sample(dataset):
    """
    Takes one batch from the dataset, applies augmentation to a single image,
    and displays both the original and augmented images side by side.
    """
    image, _labels = extract_first_record(dataset)

    # Get Image & Agument it
    augmented_image = augment_xray(image)

    # Convert tensors to numpy for visualizatio
    original_np = image.numpy()
    augmented_np = augmented_image.numpy()

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(augmented_np)
    axes[1].set_title("Augmented Image")
    axes[1].axis("off")

    plt.show()

def extract_first_record(dataset: tf.data.Dataset) -> (tf.Tensor, tf.Tensor):
    """
    Extracts the first record from a TensorFlow dataset and returns it as a tuple
    containing the image and labels.
    :param dataset: TensorFlow dataset containing (image, labels) pairs
    :return: A tuple containing the image and labels
    """
    for images, labels in dataset.take(1):
        if len(images.shape) == 4:
            return images[0], labels[0]
        else:
            return images, labels

# ------------------------------------------------------------------------------
# Logger Utils
# ------------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addFilter(LogFilter())
    tf.get_logger().addFilter(LogFilter())
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    file_handler = logging.FileHandler("training.log")
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_training_logger(logger: logging.Logger, batch_size: int, log_interval: int=10) -> TrainingLogger:
    """
    Setup Training Logger, that will log status of model training
    :param logger: Logger instance to use for logging
    :param batch_size: Batch size used for training
    :param log_interval: Logs every N batches (default: 10)
    :return: TrainingLogger
    """
    return TrainingLogger(logger, batch_size, log_interval)

def setup_weight_monitor(logger: logging.Logger) -> WeightMonitor:
    """
    Setup Weight Monitor, that will log weights in model training
    :param logger:
    :return:
    """
    return WeightMonitor(logger)

def setup_metrics_logger(output_dir: str, output_file: str) -> CSVMetricsLogger:
    """
    Setup CSV Metrics Logger that will save training metrics to CSV files.
    This will be used to log metrics during training, and load them later for analysis.
    """
    return CSVMetricsLogger(output_dir, output_file)

def get_metrics(threshold: float=0.5):
    """
    Returns a list of metrics to be used during training.
    Configured specifically for multi-label classification.
    :param threshold: Classification threshold for binary metrics (default: 0.5)
    :return: List of metrics configured for multi-label classification
    """
    return [
        BinaryAccuracy(name='accuracy', threshold=threshold),
        Precision(name='precision', thresholds=threshold),
        Recall(name='recall', thresholds=threshold),
        AUC(name='auc', multi_label=True, num_labels=NUM_CLASSES),
        F1Score(name='f1_score', threshold=threshold, average='weighted')
    ]
