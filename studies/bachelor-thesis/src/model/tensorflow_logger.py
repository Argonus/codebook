import logging
from tensorflow.keras.callbacks import Callback

import time

class TrainingLogger(Callback):
    def __init__(self, logger: logging.Logger, batch_size: int, log_interval: int=10):
        """
        Custom callback to log training details at each epoch and batch level.
        :param logger: Logger instance to use for logging
        :param batch_size: Batch size used for training
        :param log_interval: Logs every N batches (default: 10)
        """
        super().__init__()
        self.logger = logger
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.epoch_start_time = None
        self.batch_start_time = None
        self.last_batch_time = None
        self.batch_times = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.batch_times = []
        self.logger.info(f"\n=== Starting Epoch {epoch+1} ===\n")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        
        metrics = []
        for k, v in logs.items():
            if 'val_' in k:
                continue
            metrics.append(f"{k}: {v:.4f}")
        
        val_metrics = []
        for k, v in logs.items():
            if 'val_' in k:
                val_metrics.append(f"{k[4:]}: {v:.4f}")
        
        self.logger.info(f"\n=== Epoch {epoch+1} Summary ===")
        self.logger.info(f"Time: {epoch_time:.2f}s")
        self.logger.info(f"Training   - {' - '.join(metrics)}")
        if val_metrics:
            self.logger.info(f"Validation - {' - '.join(val_metrics)}")
        self.logger.info("="*50 + "\n")

    def on_train_begin(self, logs=None):
        self.logger.info("\n=== Training Started ===\n")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Optimizer: {self.model.optimizer.__class__.__name__}")
        self.logger.info("\n" + "="*50 + "\n")

    def on_train_end(self, logs=None):
        self.logger.info("\n=== Training Completed! ===\n")
        if logs:
            metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in logs.items()])
            self.logger.info(f"Final Metrics: {metrics_str}\n")