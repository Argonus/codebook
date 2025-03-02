import os
import csv
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import logging


class LossAnalysisMonitor(Callback):
    def __init__(self, output_dir: str, model_name: str, logger: logging.Logger, validation_data: tf.data.Dataset, 
                 class_names: list[str], resume_training: bool = False, initial_epoch: int = 0):
        """
        :param validation_data: Validation dataset to analyze
        :param class_names: List of class names
        :param output_dir: Directory to save metrics
        :param model_name: Name of the model (used for file naming)
        :param logger: Logger instance for real-time logging
        :param resume_training: Whether this is a resumed training run (if True, preserves existing files)
        :param initial_epoch: Epoch to start from when resuming (used to clean up data from this epoch and beyond)
        """
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.logger = logger
        self.resume_training = resume_training
        self.initial_epoch = initial_epoch

        self.output_dir = f"{output_dir}/{model_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.output_dir, "loss_analysis_metrics.csv")
        
        # Initialize metrics file (and clean up if resuming)
        if self.resume_training and self.initial_epoch > 0:
            self._cleanup_existing_records()
        self.init_metrics_file()
        
    def _cleanup_existing_records(self):
        """Clean up metrics data from initial_epoch and beyond when resuming training."""
        try:
            if os.path.exists(self.metrics_file):
                temp_file = self.metrics_file + '.temp'
                kept_records = 0
                
                with open(self.metrics_file, 'r') as infile, open(temp_file, 'w', newline='') as outfile:
                    reader = csv.reader(infile)
                    writer = csv.writer(outfile)
                    
                    # Copy the header row
                    header = next(reader)
                    writer.writerow(header)
                    
                    # Process rows - group them by epoch for loss analysis (since we have multiple rows per epoch)
                    rows_by_epoch = {}
                    for row in reader:
                        epoch = int(row[0])  # epoch is the first column
                        if epoch not in rows_by_epoch:
                            rows_by_epoch[epoch] = []
                        rows_by_epoch[epoch].append(row)
                    
                    # Only keep rows where epoch < initial_epoch
                    for epoch, rows in sorted(rows_by_epoch.items()):
                        if epoch < self.initial_epoch:
                            for row in rows:
                                writer.writerow(row)
                                kept_records += 1
                
                # Replace the original file with the cleaned version
                os.replace(temp_file, self.metrics_file)
                self.logger.info(f"Cleaned loss analysis metrics file, kept {kept_records} records before epoch {self.initial_epoch}")
        except Exception as e:
            self.logger.error(f"Error cleaning up existing records: {str(e)}")
            
    def init_metrics_file(self):
        """Initialize CSV file with headers."""
        # If resuming training and the file exists, don't reinitialize
        if self.resume_training and os.path.exists(self.metrics_file):
            self.logger.info(f"Resuming from existing loss analysis file: {self.metrics_file}")
            return
            
        headers = [
            'epoch', 
            'timestamp', 
            'class_name',
            'high_confidence_ratio',    # > 0.9
            'medium_confidence_ratio',  # 0.6-0.9
            'uncertain_ratio',          # 0.4-0.6
            'low_confidence_ratio',     # < 0.4
            'true_positives',
            'false_positives',
            'loss_contribution',
            'avg_confidence_correct',
            'avg_confidence_incorrect'
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def on_epoch_end(self, epoch, logs={}):
        """Analyze and log metrics at the end of each epoch."""
        # Get predictions
        y_pred = np.concatenate([self.model.predict(batch[0])
                                 for batch in self.validation_data], axis=0)
        y_true = np.concatenate([y for _, y in self.validation_data], axis=0)

        timestamp = datetime.now().isoformat()
        metrics_rows = []

        # Per-class analysis
        for i, class_name in enumerate(self.class_names):
            class_pred = y_pred[:, i]
            class_true = y_true[:, i]

            # Get prediction analysis
            pred_analysis = self._analyze_predictions(class_pred, class_true)
            loss_contrib = self._calculate_loss_contribution(class_pred, class_true)

            # Prepare row for CSV
            metrics_rows.append([
                epoch,
                timestamp,
                class_name,
                float(pred_analysis['high_conf']),
                float(pred_analysis['med_conf']),
                float(pred_analysis['uncertain']),
                float(pred_analysis['low_conf']),
                int(pred_analysis['true_pos']),
                int(pred_analysis['false_pos']),
                float(loss_contrib),
                float(pred_analysis['avg_conf_correct']),
                float(pred_analysis['avg_conf_incorrect'])
            ])

            # Print current stats
            self.logger.info(f"\nLoss Analysis - {class_name}")
            self.logger.info(f"Confidence Distribution:")
            self.logger.info(f"-- High (>0.9): {pred_analysis['high_conf']:.2%}")
            self.logger.info(f"-- Medium (0.6-0.9): {pred_analysis['med_conf']:.2%}")
            self.logger.info(f"-- Uncertain (0.4-0.6): {pred_analysis['uncertain']:.2%}")
            self.logger.info(f"-- Low (<0.4): {pred_analysis['low_conf']:.2%}")
            self.logger.info(f"Performance:")
            self.logger.info(f"-- True Positives: {pred_analysis['true_pos']}")
            self.logger.info(f"-- False Positives: {pred_analysis['false_pos']}")
            self.logger.info(f"-- Loss Contribution: {loss_contrib:.4f}")
            self.logger.info(f"Average Confidence:")
            self.logger.info(f"-- Correct Predictions: {pred_analysis['avg_conf_correct']:.2%}")
            self.logger.info(f"-- Incorrect Predictions: {pred_analysis['avg_conf_incorrect']:.2%}")

        # Write all rows at once
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_rows)


    def _analyze_predictions(self, class_pred, class_true):
        """
        Analyze predictions for a single class.
        """
        high_conf = np.mean(class_pred > 0.9)
        med_conf = np.mean((class_pred > 0.6) & (class_pred <= 0.9))
        uncertain = np.mean((class_pred > 0.4) & (class_pred <= 0.6))
        low_conf = np.mean(class_pred <= 0.4)
        
        true_pos = np.sum((class_pred > 0.5) & (class_true == 1))
        false_pos = np.sum((class_pred > 0.5) & (class_true == 0))
        
        correct_mask = ((class_pred > 0.5) & (class_true == 1)) | ((class_pred <= 0.5) & (class_true == 0))
        incorrect_mask = ~correct_mask
        
        avg_conf_correct = np.mean(class_pred[correct_mask]) if np.any(correct_mask) else 0
        avg_conf_incorrect = np.mean(class_pred[incorrect_mask]) if np.any(incorrect_mask) else 0
        
        return {
            'high_conf': high_conf,
            'med_conf': med_conf,
            'uncertain': uncertain,
            'low_conf': low_conf,
            'true_pos': true_pos,
            'false_pos': false_pos,
            'avg_conf_correct': avg_conf_correct,
            'avg_conf_incorrect': avg_conf_incorrect
        }
    
    def _calculate_loss_contribution(self, class_pred, class_true):
        """
        Calculate loss contribution based on the model's loss function.
        """
        if isinstance(self.model.loss, tf.keras.losses.BinaryFocalCrossentropy):
            focal_loss = self._calculate_focal_loss(class_pred, class_true)
            return focal_loss
        else:
            bce_loss = self._calculate_bce_loss(class_pred, class_true)
            return bce_loss

    def _calculate_focal_loss(self, class_pred, class_true):
        pt = np.where(class_true == 1, class_pred, 1 - class_pred)
        gamma = self.model.loss.gamma
        alpha = self.model.loss.alpha

        focal_factor = np.power(1 - pt, gamma)
        if alpha is not None:
            focal_factor *= np.where(class_true == 1, alpha, 1 - alpha)

        return -np.mean(focal_factor * np.log(pt + 1e-7))

    def _calculate_bce_loss(self, class_pred, class_true):
        return -np.mean(
            class_true * np.log(class_pred + 1e-7) +
            (1 - class_true) * np.log(1 - class_pred + 1e-7)
        )