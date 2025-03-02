import gc
import tensorflow as tf
from logging import Logger

class GarbageCollector(tf.keras.callbacks.Callback):
    def __init__(self, logger: Logger):
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()
        self.logger.info("\n=======Garbage Collector======")
        self.logger.info("Memory cleaned at end of epoch %d", epoch+1)