import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import numpy as np

from tqdm import tqdm
from typing import Union, List

from src.utils.consts import DENSENET_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.model.tensorflow_logger import TrainingLogger
from src.model.tensorflow_weight_monitor import WeightMonitor

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
    Parses a record from the TFRecord dataset.
    :param tensorflow example protobuf
    :return: tuple of image and labels tensors
    """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "encoded_finding_labels": tf.io.VarLenFeature(tf.int64)
    }

    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode Image
    image = tf.io.decode_png(parsed_features["image"], channels=1, dtype=tf.uint8)
    image = tf.image.grayscale_to_rgb(image)

    # Resize Normalize Image to match ImageNet requirements
    image = tf.image.resize(image, DENSENET_IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    imagenet_mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    imagenet_std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    image = (image - imagenet_mean) / imagenet_std

    # Convert Labels to Dense
    labels = tf.sparse.to_dense(parsed_features["encoded_finding_labels"])
    labels = tf.pad(labels, [[0, 14 - tf.shape(labels)[0]]], constant_values=0)
    labels = tf.reshape(labels, [14])

    return image, labels

def load_and_split_dataset(record_path: str, batch_size: int = 32, shuffle_buffer_size: int = 1000, tfrecord_buffer_size: int = 262144, dataset_size: int = None)  -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    """
    Loads, splits, and optimizes a TFRecord dataset.
    """
    dataset = load_dataset(record_path, tfrecord_buffer_size)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.7, 0.15, shuffle_buffer_size, dataset_size)

    train_dataset = optimize_dataset(train_dataset, batch_size)
    val_dataset = optimize_dataset(val_dataset, batch_size)
    test_dataset = optimize_dataset(test_dataset, batch_size)

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


def optimize_dataset(dataset, batch_size: int):
    """
    Applies padded batching and prefetching to optimize dataset performance.
    - padded batching is used, as we have multiple labels used in classification
    - prefetching is used well to prefetch data for training processes
    """
    return dataset.padded_batch(batch_size, padded_shapes=([224, 224, 3], [None])).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------------------
# Tensorflow Image Transform Utils
# ------------------------------------------------------------------------------
def apply_augmentation(image: tf.Tensor, augmentation_func: callable, probability=0.5, *args, **kwargs) -> tf.Tensor:
    """
    Applies an augmentation function to an image with a given probability.
    Supports passing additional arguments to augmentation functions.
    """
    return tf.cond(
        tf.random.uniform([], 0, 1) < probability,
        lambda: augmentation_func(image, *args, **kwargs),  # Pass additional parameters
        lambda: image
    )

def random_brightness(image: tf.Tensor, max_delta: float = 0.2) -> tf.Tensor:
    """
    Apply random brightness adjustment.
    Mimics: Variations in X-ray exposure settings or tissue absorption
    """
    return tf.image.random_brightness(image, max_delta)

def random_contrast(image: tf.Tensor, lower: float = 0.5, upper: float = 1.5) -> tf.Tensor:
    """
    Apply random contrast adjustment.
    Mimics: Differences in scanner contrast settings or patient body composition
    """
    return tf.image.random_contrast(image, lower, upper)

def random_shifting(image: tf.Tensor, minval: int = -10, maxval: int = 10) -> tf.Tensor:
    """
    Apply random shifting.
    Mimics: Differences in scanner position or patient position
    """
    shape = tf.shape(image)
    batch_size = shape[0] if tf.rank(image) == 4 else 1
    height, width = shape[-3], shape[-2]

    dx = tf.random.uniform([batch_size], minval=minval, maxval=maxval, dtype=tf.float32)
    dy = tf.random.uniform([batch_size], minval=minval, maxval=maxval, dtype=tf.float32)

    transforms = tf.stack([
        tf.ones_like(dx), tf.zeros_like(dx), dx,
        tf.zeros_like(dy), tf.ones_like(dy), dy,
        tf.zeros_like(dx), tf.zeros_like(dx)
    ], axis=1)

    if tf.rank(image) == 3:
        image = tf.expand_dims(image, axis=0)

    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transforms,
        interpolation="BILINEAR",
        output_shape=[height, width],
        fill_value=0.0
    )

    return tf.squeeze(transformed, axis=0) if batch_size == 1 else transformed

def gaussian_noise(image: tf.Tensor, stddev: float = 0.2) -> tf.Tensor:
    """
    Apply Gaussian noise.
    Mimics: Noise in the X-ray image
    """
    noise = tf.random.normal(shape=tf.shape(image), mean=0., stddev=stddev)
    return tf.clip_by_value(image + noise, 0, 1)

def augment_xray(image: tf.Tensor) -> tf.Tensor:
    """
    Applies medical-safe augmentations to a chest X-ray image or batch of images.
    """
    image = tf.cast(image, tf.float32)

    if tf.rank(image) == 3:
        image = tf.expand_dims(image, axis=0)

    image = apply_augmentation(image, random_brightness, probability=0.5)
    image = apply_augmentation(image, random_contrast, probability=0.5)
    image = apply_augmentation(image, random_shifting, probability=0.1)
    image = apply_augmentation(image, gaussian_noise, probability=0.05)

    return tf.squeeze(image, axis=0) if tf.shape(image)[0] == 1 else image

# ------------------------------------------------------------------------------
# Statistics calculations
# ------------------------------------------------------------------------------
def calculate_class_weights(dataset: tf.data.Dataset, num_classes: int) -> dict:
    total_samples = 0
    class_counts = np.zeros(num_classes)

    for _, labels in dataset:
        total_samples += labels.shape[0]
        class_counts += tf.reduce_sum(labels, axis=0)

    class_weights = total_samples / (num_classes * class_counts)
    return dict(enumerate(class_weights))

# ------------------------------------------------------------------------------
# Visualisation Utils
# ------------------------------------------------------------------------------
def show_augmented_sample(dataset):
    """
    Takes one batch from the dataset, applies augmentation to a single image,
    and displays both the original and augmented images side by side.
    """
    for images, labels in dataset.take(1):
        break  # Extract only one batch

    # Get Image & Agument it
    image = images[0]
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

# ------------------------------------------------------------------------------
# Logger Utils
# ------------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def setup_training_logger(logger: logging.Logger, log_interval: int=10) -> TrainingLogger:
    """
    Setup Training Logger, that will log status od model training
    :param log_interval:
    :param logger
    :return: TrainingLogger
    """
    return TrainingLogger(logger, log_interval)

def setup_weight_monitor(logger: logging.Logger) -> WeightMonitor:
    """
    Setup Weight Monitor, that will log weights in model training
    :param logger:
    :return:
    """
    return WeightMonitor(logger)