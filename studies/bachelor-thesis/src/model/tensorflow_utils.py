import os 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import logging

from tqdm import tqdm
from typing import Union, List
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, F1Score, AUC

from src.utils.consts import DENSENET_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, TF_RECORD_DATASET, DROP_NO_FINDING_CLASS, NO_FINDING_CLASS_IDX, NUM_CLASSES_WITHOUT_NO_FINDING

# Image Augmentation
from src.model.tensorflow_image_augmentation import augment_xray

from src.model.tensorflow_logger import TrainingLogger
from src.model.monitors.tensorflow_metrics_monitor import MetricsMonitor
from tensorflow.keras.callbacks import EarlyStopping

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
    :param example_proto: A serialized Example proto containing image and labels
    
    :return: A tuple of (image, labels) where:
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

def filter_no_finding_class(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Filters out the 'No Finding' class from dataset labels.
    
    :param dataset: TensorFlow dataset with (image, labels) pairs where labels has 15 classes
    :return: Dataset with 'No Finding' class removed from labels
    """
    
    if DROP_NO_FINDING_CLASS:
        return dataset.map(_remove_no_finding, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def _remove_no_finding(image, labels):
    labels_before = labels[:NO_FINDING_CLASS_IDX]
    labels_after = labels[NO_FINDING_CLASS_IDX+1:]
    new_labels = tf.concat([labels_before, labels_after], axis=0)
    new_labels = tf.ensure_shape(new_labels, [NUM_CLASSES - 1])

    return image, new_labels

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


def apply_augmentation_to_dataset(dataset: tf.data.Dataset, probability: dict = {}) -> tf.data.Dataset:
    """
    Applies augmentation to a dataset.
    Each sample should have shape ([224, 224, 3], [NUM_CLASSES]).
    """
    return dataset.map(
        lambda x, y: (augment_xray(x, probability), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

def optimize_dataset(dataset, batch_size: int):
    """
    Applies batching and prefetching to optimize dataset performance.
    Each sample should have shape ([224, 224, 3], [NUM_CLASSES]) after parse_record.
    """
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------------------
# Data Oversampling
# ------------------------------------------------------------------------------
def get_sampling_rates(class_weights: dict, min_rate: float = 1.0, max_rate: float = 5.0) -> tf.Tensor:
    """    
    :param class_weights: Dictionary mapping class indices to their respective weights
    :param min_rate: Minimum sampling rate (default: 1.0)
    :param max_rate: Maximum sampling rate (default: 5.0)
    :return TensorFlow tensor of sampling rates
    """
    min_weight = min(class_weights.values())
    max_weight = max(class_weights.values())
    weight_range = max_weight - min_weight
    
    sampling_rates = []
    num_classes = get_num_classes()
    
    for i in range(num_classes):
        if not DROP_NO_FINDING_CLASS and i == NO_FINDING_CLASS_IDX:
            sampling_rates.append(min_rate)
        else:
            weight = class_weights[i]
            if weight_range > 0:
                norm_weight = (weight - min_weight) / weight_range
                rate = min_rate + norm_weight * (max_rate - min_rate)
            else:
                rate = max_rate
            sampling_rates.append(rate)
    
    return tf.constant(sampling_rates, dtype=tf.float32)

def oversample_minority_classes(dataset: tf.data.Dataset, class_weights: dict, min_rate: float = 1.0, max_rate: float = 5.0) -> tf.data.Dataset:
    """
    Oversampling for minority classes with proportional sampling rates.
    :param dataset: TensorFlow dataset containing (image, labels) pairs
    :param class_weights: Dictionary mapping class indices to their respective weights
    :param min_rate: Minimum sampling rate (default: 1.0)
    :param max_rate: Maximum sampling rate (default: 5.0)   
    :returns TensorFlow dataset with oversampled minority classes
    """
    sampling_rates = get_sampling_rates(class_weights, min_rate, max_rate)
    
    def calculate_repeat_weights(x, y):
        indices = tf.where(y > 0)
        has_positive_class = tf.shape(indices)[0] > 0
    
        def process_with_positives():
            present_classes = tf.cast(indices[:, 0], tf.int32)
            class_rates = tf.gather(sampling_rates, present_classes)
            return tf.cond(
                tf.shape(present_classes)[0] > 1,
            lambda: tf.cast(tf.math.ceil(tf.reduce_max(class_rates) * 1.5), tf.int64),
            lambda: tf.cast(tf.math.ceil(tf.reduce_max(class_rates)), tf.int64)
        )

        repeat_weight = tf.cond(
            has_positive_class,
            process_with_positives,
            lambda: tf.constant(1, dtype=tf.int64)  # Use 1 as default for samples with no positive classes
        )
    
        return x, y, repeat_weight
    
    weighted_ds = dataset.map(calculate_repeat_weights, num_parallel_calls=tf.data.AUTOTUNE)
    def repeat_sample(x, y, repeat_count):
        sample_ds = tf.data.Dataset.from_tensors((x, y))
        return sample_ds.repeat(repeat_count)
    
    repeated_ds = weighted_ds.flat_map(repeat_sample)
    shuffled_ds = repeated_ds.shuffle(buffer_size=10000, seed=42)
    
    return shuffled_ds

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

def analyze_class_distribution(dataset: tf.data.Dataset, num_classes: int) -> dict:
    """
    Analyzes class distribution in a multi-label dataset.
    :param dataset: TensorFlow dataset containing (image, labels) pairs
    :param num_classes: Number of classes (default 15)
    :return class_distribution: Dictionary with class statistics
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
    
    class_percentages = (class_counts / total_samples) * 100
    class_distribution = {
        'total_samples': total_samples,
        'class_counts': class_counts,
        'class_percentages': class_percentages,
        'class_weights': (total_samples / (num_classes * class_counts)).astype(np.float32)
    }
    
    # Print statistics
    print(f"Total samples: {total_samples}")
    print("\nClass distribution:")
    for i in range(num_classes):
        print(f"Class {i}: {class_counts[i]} samples ({class_percentages[i]:.2f}%), weight: {class_distribution['class_weights'][i]:.4f}")
    
    return class_distribution

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

def setup_early_stopping(patience: int = 15, min_delta: float = 0.001) -> EarlyStopping:
    """
    Setup early stopping callback to prevent overfitting.
    
    :param patience: Number of epochs with no improvement after which training will stop
    :param min_delta: Minimum change in monitored quantity to qualify as an improvement
    :return: Configured EarlyStopping callback
    """
    return EarlyStopping(
        monitor='val_f1_score',  # Monitor validation F1 score
        mode='max',         # We want to maximize the F1 score
        patience=patience,  # Number of epochs with no improvement
        min_delta=min_delta,  # Minimum change to qualify as an improvement
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
        verbose=1  # Print message when early stopping is triggered
    )

def setup_metrics_monitor(output_dir: str, model_name: str, logger: logging.Logger, resume_training: bool = False, initial_epoch: int = 0) -> MetricsMonitor:
    """
    Setup CSV Metrics Logger that will save training metrics to CSV files.
    This will be used to log metrics during training, and load them later for analysis.
    
    :param output_dir: Directory to save metrics
    :param model_name: Name of the model
    :param logger: Logger instance
    :param resume_training: Whether this is a resumed training run (if True, preserves existing files)
    :param initial_epoch: Epoch to start from when resuming (used to clean up data from this epoch and beyond)
    :return: MetricsMonitor instance
    """
    return MetricsMonitor(output_dir, model_name, logger, resume_training=resume_training, initial_epoch=initial_epoch)

def get_metrics(threshold: float=0.5, as_dict: bool=False):
    """
    Returns metrics to be used during training.
    Configured specifically for multi-label classification.
    
    :param threshold: Classification threshold for binary metrics (default: 0.5)
    :param as_dict: If True, returns a dictionary mapping metric names to metric objects
                    This is useful for custom_objects in model loading
    :return: List or dict of metrics configured for multi-label classification
    """
    num_classes = get_num_classes()
    metrics = [
        BinaryAccuracy(name='accuracy', threshold=threshold),
        Precision(name='precision', thresholds=threshold),
        Recall(name='recall', thresholds=threshold),
        AUC(name='auc', multi_label=True, num_labels=num_classes),
        F1Score(name='f1_score', threshold=threshold, average='weighted')
    ]
    
    if as_dict:
        # Convert to a dictionary for easy lookup when loading models
        metrics_dict = {}
        for metric in metrics:
            metrics_dict[metric.name] = metric
        return metrics_dict
    
    return metrics

# ------------------------------------------------------------------------------
# Tensorflow Model Utils
# ------------------------------------------------------------------------------
def load_model(models_path: str, model_name: str) -> tf.keras.Model:
    """
    Loads a TensorFlow model from a given path.
    :param model_path: Path to the saved models 
    :param model_name: Name of the model
    :return: Loaded TensorFlow model
    """
    return tf.keras.models.load_model(os.path.join(models_path, f"{model_name}.keras"), safe_mode=False)


def start_or_resume_training(model, compile_kwargs, train_ds, val_ds, epochs, steps_per_epoch, validation_steps, 
                        class_weights=None, callbacks=None, checkpoint_path=None, initial_epoch=0,
                        output_dir=None, model_name=None, logger=None):
    """
    Start training from scratch or resume from a checkpoint
    
    :param model: TensorFlow model instance
    :param compile_kwargs: Dictionary with model compilation arguments
    :param train_ds: Training dataset
    :param val_ds: Validation dataset
    :param epochs: Total number of epochs to train
    :param steps_per_epoch: Number of steps per epoch
    :param validation_steps: Number of validation steps
    :param class_weights: Optional dictionary of class weights
    :param callbacks: Optional list of callbacks
    :param checkpoint_path: Optional path to checkpoint to resume from
    :param initial_epoch: Epoch to start from (for resuming training)
    :param output_dir: Optional output directory for saving models
    :param model_name: Optional model name
    :param logger: Optional logger instance
    :return: Tuple of (training history, trained model)
    """
    is_resuming = checkpoint_path is not None and os.path.exists(checkpoint_path)
    
    if is_resuming:
        if logger:
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        history, trained_model = resume_training(checkpoint_path, train_ds, val_ds, 
                                    epochs, initial_epoch, steps_per_epoch, validation_steps, 
                                    class_weights, callbacks, output_dir, model_name, logger)
        return history, trained_model
    else:
        history, model = start_training(model, compile_kwargs, train_ds, val_ds, epochs, 
                              callbacks, steps_per_epoch, validation_steps, class_weights)
        return history, model

def resume_training(checkpoint_path: str, 
                   train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, 
                   epochs: int, initial_epoch: int, steps_per_epoch: int, validation_steps: int,
                   class_weights: dict = None, callbacks: list = None,
                   output_dir: str = None, model_name: str = None, logger = None):
    """
    Resume training from a checkpoint with a specified initial epoch
    
    :param checkpoint_path: Path to the checkpoint file
    :param train_ds: Training dataset
    :param val_ds: Validation dataset
    :param epochs: Total number of epochs to train (including previously trained epochs)
    :param initial_epoch: Epoch to start from (for logging and scheduling purposes)
    :param steps_per_epoch: Number of steps per epoch
    :param validation_steps: Number of validation steps
    :param class_weights: Optional dictionary of class weights
    :param callbacks: List of callbacks
    :param output_dir: Optional output directory for saving models
    :param model_name: Optional model name
    :param logger: Optional logger instance
    :return: Tuple of (training history, trained model)
    """
    # Load the full model with its optimizer state
    loaded_model = tf.keras.models.load_model(checkpoint_path, safe_mode=False)
    
    # Set up metrics monitor if needed
    if output_dir and model_name and logger:
        if callbacks is None:
            callbacks = []
        
        metrics_monitor_found = False
        for i, callback in enumerate(callbacks):
            if isinstance(callback, MetricsMonitor):
                callbacks[i] = setup_metrics_monitor(output_dir, model_name, logger, resume_training=True, initial_epoch=initial_epoch)
                metrics_monitor_found = True
        
        if not metrics_monitor_found:
            callbacks.append(setup_metrics_monitor(output_dir, model_name, logger, resume_training=True, initial_epoch=initial_epoch))
    
    history = loaded_model.fit(
        train_ds.repeat(),
        validation_data=val_ds.repeat(),
        class_weight=class_weights,
        epochs=epochs,
        initial_epoch=initial_epoch-1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    return history, loaded_model

def start_training(model: tf.keras.Model, compile_kwargs: dict, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, 
                epochs: int, callbacks: list, steps_per_epoch: int, validation_steps: int, class_weights: dict = None):
    """
    Start training a model from scratch
    
    :param model: TensorFlow model instance
    :param compile_kwargs: Dictionary with model compilation arguments
    :param train_ds: Training dataset
    :param val_ds: Validation dataset
    :param epochs: Number of epochs to train
    :param callbacks: List of callbacks
    :param steps_per_epoch: Number of steps per epoch
    :param validation_steps: Number of validation steps
    :param class_weights: Optional dictionary of class weights
    :return: Training history
    """
    model.compile(**compile_kwargs)
    history = model.fit(
        train_ds.repeat(),
        validation_data=val_ds.repeat(),
        class_weight=class_weights,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    return history, model

def get_num_classes() -> int:
    """Returns the number of classes based on whether No Finding is dropped"""    
    return NUM_CLASSES_WITHOUT_NO_FINDING if DROP_NO_FINDING_CLASS else NUM_CLASSES