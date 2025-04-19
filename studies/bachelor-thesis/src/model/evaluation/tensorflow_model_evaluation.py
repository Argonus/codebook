import os
from typing import Tuple, Dict, List, Optional

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils.consts import TF_RECORD_DATASET, NO_FINDING_CLASS_IDX, TF_BUFFER_SIZE, DROP_NO_FINDING_CLASS, TF_BATCH_SIZE
from src.model.tensorflow_utils import load_dataset, optimize_dataset, filter_no_finding_class

class ModelEvaluation:
    """Class for evaluating TensorFlow models on chest X-ray classification tasks.
    - Handles evaluation of multi-label classification models, including    
    """
    
    def __init__(self, model: tf.keras.Model, model_name: str, label_mappings_path: str, output_dir: str):
        """Initialize the model evaluation environment.
        :param model: TensorFlow model to evaluate
        :param model_name: Name of the model (used for file naming)
        :param label_mappings_path: Path to the label mappings CSV file
        :param output_dir: Directory to save evaluation results
        """
        self.output_dir = f"{output_dir}/{model_name}"
        os.makedirs(self.output_dir, exist_ok=True)
            
        self.model = model
        self.model_name = model_name
        self.has_no_finding_output = not DROP_NO_FINDING_CLASS
        self.no_finding_idx = NO_FINDING_CLASS_IDX
        self.test_dataset = self._load_dataset()
        self.label_names = self._load_label_mappings(label_mappings_path)

        # Output files
        self.test_report_path = os.path.join(self.output_dir, "classification_report.txt")
        self.metrics_file = os.path.join(self.output_dir, "model_metrics.csv")
        self.sample_prediction_file = os.path.join(self.output_dir, "sample_predictions.png")

        # Evaluation results
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        self.y_pred_proba: Optional[np.ndarray] = None
        self.metrics: Optional[Dict[int, Dict[str, float]]] = None

    
    def get_test_dataset(self) -> tf.data.Dataset:
        """
        Get the test dataset used for evaluation.
        :return: The test dataset with appropriate preprocessing
        """
        return self.test_dataset
    
    def evaluate(self) -> Dict[int, Dict[str, float]]:
        """Run complete model evaluation on the test dataset and save results.
        :return: Dictionary of metrics per class with structure
        """
        print("\nGenerating predictions...")
        self.y_true, self.y_pred, self.y_pred_proba = self._get_predictions()
        
        print("Calculating metrics...")
        self.metrics = self._calculate_metrics_per_class(self.y_true, self.y_pred, self.y_pred_proba)
        
        print("Saving metrics to CSV...")
        self._save_metrics_to_csv(self.metrics)
        
        print("\nEvaluation complete! You can now use:")
        print("- generate_classification_report() to see the classification report")
        print("- plot_confusion_matrices() to generate confusion matrices")
        print("- visualize_prediction() to examine individual predictions")
        
        return self.metrics
        
    def generate_classification_report(self) -> None:
        """
        Generate and save classification report with performance metrics.
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No predictions available. Run evaluate() first.")
            
        print("\nClassification Report:")
        
        class_report = classification_report(
            self.y_true, 
            self.y_pred,
            target_names=self.label_names,
            zero_division=0
        )
        
        # Add note for models without No Finding class
        if not self.has_no_finding_output:
            class_report_with_note = (
                "Note: 'No Finding' class is excluded as this model doesn't predict it.\n\n" +
                class_report
            )
            print(class_report_with_note)
            
            with open(self.test_report_path, 'w') as f:
                f.write(class_report_with_note)
        else:
            print(class_report)
            
            with open(self.test_report_path, 'w') as f:
                f.write(class_report)
                
        print(f"\nReport saved to: {self.test_report_path}")

    def plot_confusion_matrices(self, combined: bool = True) -> None:
        """Generate and save confusion matrices.
        :param combined: 
            - If True, generates a single combined confusion matrix for all labels.
            - If False, generates separate confusion matrices for each class.        
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No predictions available. Run evaluate() first.")
            
        if combined:
            self._plot_combined_confusion_matrix()
        else:
            self._plot_per_class_confusion_matrices()
    
    def _plot_combined_confusion_matrix(self) -> None:
        """
        Generate a single combined confusion matrix for all labels.
        """
        print("\nGenerating combined confusion matrix...")
        
        n_classes = len(self.label_names)
        cm = np.zeros((n_classes, n_classes))

        for i in range(len(self.y_true)):
            true_labels = np.where(self.y_true[i] == 1)[0]
            pred_labels = np.where(self.y_pred[i] == 1)[0]
            
            for true_label in true_labels:
                for pred_label in pred_labels:
                    cm[true_label][pred_label] += 1
        
        plt.figure(figsize=(15, 12))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(int(cm[i, j]), 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        if not self.has_no_finding_output:
            nf_color = 'lightyellow'
            plt.axhspan(self.no_finding_idx - 0.5, self.no_finding_idx + 0.5, color=nf_color, alpha=0.3)
            plt.axvspan(self.no_finding_idx - 0.5, self.no_finding_idx + 0.5, color=nf_color, alpha=0.3)
            
            no_finding_note = "* 'No Finding' class: Model does not predict this class" 
            plt.figtext(0.5, 0.01, no_finding_note, wrap=True, horizontalalignment='center', fontsize=10)
        
        shortened_labels = [label[:15] + '...' if len(label) > 15 else label for label in self.label_names]
        plt.xticks(range(n_classes), shortened_labels, rotation=45, ha='right')
        plt.yticks(range(n_classes), shortened_labels)
        
        plt.title('Combined Confusion Matrix - All Labels')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'combined_confusion_matrix.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved combined confusion matrix to: {save_path}")
    
    def _plot_per_class_confusion_matrices(self) -> None:
        """Generate separate confusion matrices for each class."""
        print("\nGenerating per-class confusion matrices...")
        for class_idx, label in enumerate(self.label_names):
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(self.y_true[:, class_idx], self.y_pred[:, class_idx])
            
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            
            # Add numeric values to cells
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.xticks([0, 1], ['Negative', 'Positive'])
            plt.yticks([0, 1], ['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {label}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            save_path = os.path.join(self.output_dir, f'{label}_confusion_matrix.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved confusion matrix for {label} to: {save_path}")

    def visualize_prediction(self, image: tf.Tensor, true_labels: tf.Tensor, threshold: float = 0.5) -> None:
        """Visualize model prediction for a single image with ground truth comparison.
        :param image: Input image tensor
        :param true_labels: True labels tensor
        :param threshold: Threshold for positive prediction (default: 0.5)
        """
        pred_probs = self.model.predict(tf.expand_dims(image, 0))[0]
        predictions = (pred_probs > threshold).astype(int)

        display_image = image.numpy()
        display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
        
        plt.figure(figsize=(15, 6))        
        plt.subplot(1, 2, 1)
        plt.imshow(display_image, cmap='gray')
        plt.axis('off')
        plt.title('X-Ray Image')
        plt.subplot(1, 2, 2)
        
        cell_text = []
        cell_colors = []
        for i, (true, pred, prob) in enumerate(zip(true_labels, predictions, pred_probs)):
            color = 'lightgreen' if true == pred else 'salmon'
            cell_text.append([self.label_names[i], f'{true}', f'{pred}', f'{prob:.3f}'])
            cell_colors.append([color] * 4)
        
        table = plt.table(
            cellText=cell_text,
            colLabels=['Finding', 'True', 'Pred', 'Probability'],
            loc='center',
            cellLoc='center',
            cellColours=cell_colors
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.axis('off')
        plt.title('Predictions vs True Labels')
        plt.tight_layout()
        
        if self.sample_prediction_file:
            plt.savefig(self.sample_prediction_file, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _load_label_mappings(self, mappings_path: str) -> List[str]:
        df = pd.read_csv(mappings_path)
        df = df.sort_values('Index')

        if not self.has_no_finding_output:
            df = df[df['Index'] != self.no_finding_idx]

        return df['Label'].tolist()

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_true_list = []
        y_pred_list = []
        
        for x, y in self.test_dataset:
            y_true_list.append(y.numpy())
            y_pred_list.append(self.model.predict(x))
        
        y_true = np.vstack(y_true_list)
        y_pred_proba = np.vstack(y_pred_list)
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        return y_true, y_pred, y_pred_proba

    def _calculate_metrics_per_class(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[int, Dict[str, float]]:
        metrics_per_class = {}
        
        for i in range(y_true.shape[1]):
            try:
                # Calculate confusion matrix values
                tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i]).ravel()
                total_samples = len(y_true[:, i])
                
                metrics_per_class[i] = {
                    'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
                    'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                    'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                    'f1_score': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                    'auc': roc_auc_score(y_true[:, i], y_pred_proba[:, i]) if len(np.unique(y_true[:, i])) > 1 else 0.0,
                    'true_positives': int(tp),
                    'true_positives_rate': float(tp) / total_samples if total_samples > 0 else 0.0,
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_negatives': int(tn)
                }
            except Exception as e:
                print(f"Warning: Error calculating metrics for class {self.label_names[i]}: {str(e)}")
                metrics_per_class[i] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'auc': 0.0,
                    'true_positives': 0,
                    'true_positives_rate': 0.0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'true_negatives': 0
                }
        
        return metrics_per_class

    def _save_metrics_to_csv(self, metrics_per_class: Dict[int, Dict[str, float]]) -> None:
        try:
            data = []
            for class_idx, metrics in metrics_per_class.items():
                row = {
                    'model_name': self.model_name,
                    'class_name': self.label_names[class_idx],
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    **metrics
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            if os.path.exists(self.metrics_file):
                existing_df = pd.read_csv(self.metrics_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(self.metrics_file, index=False)
            print(f"\nMetrics saved to: {self.metrics_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save metrics to CSV: {str(e)}")

    def _load_dataset(self) -> tf.data.Dataset:
        dataset = load_dataset(f"{TF_RECORD_DATASET}/test.tfrecord", TF_BUFFER_SIZE)
        if not self.has_no_finding_output:
            dataset = filter_no_finding_class(dataset)
        dataset = optimize_dataset(dataset, TF_BATCH_SIZE)
        
        return dataset