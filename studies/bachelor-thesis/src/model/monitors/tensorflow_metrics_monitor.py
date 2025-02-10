
import csv
from datetime import datetime
import os
import tensorflow as tf
import logging

class MetricsMonitor(tf.keras.callbacks.Callback):
    """
    Callback that logs training metrics to CSV file after each batch.
    Creates a single csv file that stores metrics for each epoch.
    """
    def __init__(self, output_dir: str, model_name: str, logger: logging.Logger, resume_training: bool = False, initial_epoch: int = 0):
        super().__init__()
        self.logger = logger
        self.output_dir = f"{output_dir}/{model_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.output_dir, "train_metrics.csv")
        self.validation_file = os.path.join(self.output_dir, "val_metrics.csv")

        # If resuming training, don't overwrite existing files
        self.resume_training = resume_training
        self.initial_epoch = initial_epoch
        
        # Initialize files (preserving existing data if resuming)
        self.init_metrics_file()
        self.init_validation_file()

        # If resuming, clean up metrics from initial_epoch and beyond, then count records
        self.batch_count = 0
        self.epoch_count = 0
        
        if self.resume_training:
            if self.initial_epoch > 0:
                self._cleanup_existing_records()
            self._count_existing_records()
        
    def init_metrics_file(self):
        """Initialize CSV file with headers."""
        # If we're resuming training and the file exists, don't reinitialize it
        if self.resume_training and os.path.exists(self.metrics_file):
            self.logger.info(f"Resuming from existing metrics file: {self.metrics_file}")
            return
            
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
        # If we're resuming training and the file exists, don't reinitialize it
        if self.resume_training and os.path.exists(self.validation_file):
            self.logger.info(f"Resuming from existing validation metrics file: {self.validation_file}")
            return
            
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
            
    def _cleanup_existing_records(self):
        """Clean up metrics data from initial_epoch and beyond when resuming training."""
        try:
            # Clean up the training metrics file
            if os.path.exists(self.metrics_file):
                temp_file = self.metrics_file + '.temp'
                kept_records = 0
                
                with open(self.metrics_file, 'r') as infile, open(temp_file, 'w', newline='') as outfile:
                    reader = csv.reader(infile)
                    writer = csv.writer(outfile)
                    
                    # Copy the header row
                    header = next(reader)
                    writer.writerow(header)
                    
                    # Only keep rows where epoch < initial_epoch
                    for row in reader:
                        if int(row[0]) < self.initial_epoch:  # epoch is the first column
                            writer.writerow(row)
                            kept_records += 1
                
                # Replace the original file with the cleaned version
                os.replace(temp_file, self.metrics_file)
                self.logger.info(f"Cleaned training metrics file, kept {kept_records} records before epoch {self.initial_epoch}")
            
            # Clean up the validation metrics file
            if os.path.exists(self.validation_file):
                temp_file = self.validation_file + '.temp'
                kept_records = 0
                
                with open(self.validation_file, 'r') as infile, open(temp_file, 'w', newline='') as outfile:
                    reader = csv.reader(infile)
                    writer = csv.writer(outfile)
                    
                    # Copy the header row
                    header = next(reader)
                    writer.writerow(header)
                    
                    # Only keep rows where epoch < initial_epoch
                    for row in reader:
                        if int(row[0]) < self.initial_epoch:  # epoch is the first column
                            writer.writerow(row)
                            kept_records += 1
                
                # Replace the original file with the cleaned version
                os.replace(temp_file, self.validation_file)
                self.logger.info(f"Cleaned validation metrics file, kept {kept_records} records before epoch {self.initial_epoch}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up existing records: {str(e)}")
            
    def _count_existing_records(self):
        """Count the existing records in metrics files to continue from the last point."""
        try:
            # Count batch records
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    # Count lines minus header
                    batch_lines = sum(1 for _ in f) - 1
                    if batch_lines > 0:
                        self.batch_count = batch_lines
                        self.logger.info(f"Found {self.batch_count} existing training records")
            
            # Count epoch records
            if os.path.exists(self.validation_file):
                with open(self.validation_file, 'r') as f:
                    # Count lines minus header
                    epoch_lines = sum(1 for _ in f) - 1
                    if epoch_lines > 0:
                        self.epoch_count = epoch_lines
                        self.logger.info(f"Found {self.epoch_count} existing validation records")
        
        except Exception as e:
            self.logger.error(f"Error counting existing records: {str(e)}")

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
            self.logger.error(f"Error logging metrics: {str(e)}")

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
            self.logger.error(f"Error logging metrics: {str(e)}")

    def _get_metric(self, key, logs):
        """Get metric value from logs and convert to float if needed."""
        metric = logs.get(key, 0.0)
        if isinstance(metric, tf.Tensor):
            metric = float(metric.numpy())

        return metric