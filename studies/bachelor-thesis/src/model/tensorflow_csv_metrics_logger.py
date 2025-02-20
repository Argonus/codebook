
import csv
from datetime import datetime
import os
import tensorflow as tf

class CSVMetricsLogger(tf.keras.callbacks.Callback):
    """
    Callback that logs training metrics to CSV file after each batch.
    Creates a single csv file that stores metrics for each epoch.
    """
    def __init__(self, output_dir: str, model_name: str):
        super().__init__()
        self.output_dir = f"{output_dir}/{model_name}"
        self.metrics_file = os.path.join(output_dir, "train_metrics.csv")
        self.validation_file = os.path.join(output_dir, "val_metrics.csv")

        os.makedirs(output_dir, exist_ok=True)
        self.init_metrics_file()
        self.init_validation_file()

        self.batch_count = 0
        self.epoch_count = 0
        
    def init_metrics_file(self):
        """Initialize CSV file with headers."""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 
                'batch', 
                'timestamp',
                # Training metrics
                'loss',
                'accuracy', 
                'precision',
                'recall',
                'f1_score',
                'auc',
                # Other metrics
                'learning_rate',
            ])
    
    def init_validation_file(self):
        """Initialize CSV file with headers."""
        with open(self.validation_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 
                'batch', 
                'timestamp',
                # Validation metrics
                'val_loss',
                'val_accuracy', 
                'val_precision',
                'val_recall',
                'val_f1_score',
                'val_auc',
                # Other metrics
                'learning_rate',
            ])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count += 1

    def on_train_batch_end(self, batch, logs=None):
        """Log metrics after each batch."""
        try:
            logs = logs or {}
            self.batch_count += 1
            
            if hasattr(self.model.optimizer.learning_rate, '__call__'):
                current_lr = float(self.model.optimizer.learning_rate(self.model.optimizer.iterations))
            else:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            
            metrics = [
                self.epoch_count,
                self.batch_count,
                datetime.now().isoformat(),
                # Training metrics
                self._get_metric('loss', logs),
                self._get_metric('accuracy', logs),
                self._get_metric('precision', logs),
                self._get_metric('recall', logs),
                self._get_metric('f1_score', logs),
                self._get_metric('auc', logs),
                # Other metrics
                current_lr
            ]
            
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(metrics)
                
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        try:
            logs = logs or {}            
            if hasattr(self.model.optimizer.learning_rate, '__call__'):
                current_lr = float(self.model.optimizer.learning_rate(self.model.optimizer.iterations))
            else:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            
            metrics = [
                self.epoch_count,
                self.batch_count,
                datetime.now().isoformat(),
                # Training metrics
                self._get_metric('val_loss', logs),
                self._get_metric('val_accuracy', logs),
                self._get_metric('val_precision', logs),
                self._get_metric('val_recall', logs),
                self._get_metric('val_f1_score', logs),
                self._get_metric('val_auc', logs),
                # Other metrics
                current_lr
            ]
            
            with open(self.validation_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(metrics)
                
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")

    def _get_metric(self, key, logs):
        """Get metric value from logs and convert to float if needed."""
        metric = logs.get(key, 0.0)
        if isinstance(metric, tf.Tensor):
            metric = float(metric.numpy())

        return metric