import os
import csv
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import logging


class LossAnalysisMonitor(Callback):
    def __init__(self, output_dir: str, model_name: str, logger: logging.Logger, validation_data: tf.data.Dataset, class_names: list[str]):
        """
        :param validation_data: Validation dataset to analyze
        :param class_names: List of class names
        :param output_dir: Directory to save metrics
        :param model_name: Name of the model (used for file naming)
        :param logger: Logger instance for real-time logging
        """
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.logger = logger

        self.output_dir = f"{output_dir}/{model_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.output_dir, "loss_analysis_metrics.csv")
        self.init_metrics_file()
        
    def init_metrics_file(self):
        """Initialize CSV file with headers."""
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