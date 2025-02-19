import tensorflow as tf
import numpy as np
import logging

from tensorflow.keras.callbacks import Callback

class WeightMonitor(Callback):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_epoch_start(self, epoch, logs=None):
        weights = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                w = layer.get_weights()[0]
                weights.extend(w.flatten())

        weights = np.array(weights)
        self.logger.info(f"\nEpoch {epoch} Weight Stats:")
        self.logger.info(f"Mean: {np.mean(weights):.3f}")
        self.logger.info(f"Std: {np.std(weights):.3f}")
        self.logger.info(f"Max: {np.max(np.abs(weights)):.3f}")

    def on_epoch_end(self, epoch, logs=None):
        weights = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                w = layer.get_weights()[0]
                weights.extend(w.flatten())

        weights = np.array(weights)
        self.logger.info(f"\nEpoch {epoch} Weight Stats:")
        self.logger.info(f"Mean: {np.mean(weights):.3f}")
        self.logger.info(f"Std: {np.std(weights):.3f}")
        self.logger.info(f"Max: {np.max(np.abs(weights)):.3f}")
