import logging
from tensorflow.keras.callbacks import Callback

class TrainingLogger(Callback):
    def __init__(self, logger: logging.Logger, log_interval: int=10):
        """
        Custom callback to log training details at each epoch and batch level.
        :param log_interval: Logs every N batches (default: 10)
        """
        super().__init__()
        self.logger = logger
        self.log_interval = log_interval

    def on_epoch_begin(self, epoch, logs=None):
        self.logger.info(f"Starting Epoch {epoch+1}...")

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(f"Finished Epoch {epoch+1} - Loss: {logs.get('loss'):.4f}, Accuracy: {logs.get('accuracy'):.4f}")

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_interval == 0:
            self.logger.info(f"Batch {batch} - Loss: {logs.get('loss'):.4f}, Accuracy: {logs.get('accuracy'):.4f}")

    def on_train_begin(self, logs=None):
        self.logger.info("Training Started...")

    def on_train_end(self, logs=None):
        self.logger.info("Training Completed!")