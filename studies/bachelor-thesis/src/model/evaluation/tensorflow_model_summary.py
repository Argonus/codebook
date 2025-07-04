import os

import pandas as pd
import numpy as np

from typing import List, Dict
from src.model.tensorflow_utils import calculate_class_weights, load_dataset
from src.utils.consts import TF_RECORD_DATASET, TF_BUFFER_SIZE, NUM_CLASSES, NO_FINDING_CLASS_IDX

class ModelSummary:
    def __init__(self, models_path: str, model_names: List[str], models_without_no_finding: List[str]) -> None:
        self.models_path = models_path
        self.model_names = model_names
        self.models_without_no_finding = models_without_no_finding

        self.train_columns = ['epoch', 'step', 'timestamp', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'auc', 'learning_rate']
        self.val_columns = ['epoch', 'step', 'timestamp', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score', 'val_auc', 'learning_rate']

    def calculate_model_metric(self, metric: str) -> pd.DataFrame:
        """
        Calculate performance metrics for multiple models.
        :param metric (str): Metrics name that should be calculated
        :returns pd.DataFrame: DataFrame with model metrics
        """

        results = []
        val_metric = f"val_{metric}"

        for model_name in self.model_names:
            model_dir = os.path.join(self.models_path, model_name)
            val_metrics = pd.read_csv(os.path.join(model_dir, "val_metrics.csv"))
            train_metrics = pd.read_csv(os.path.join(model_dir, "train_metrics.csv"))
            
            best_validation = val_metrics[val_metric].max()
            avarage_validation = val_metrics[val_metric].mean()
            
            # Early stopping threshold
            threshold = best_validation * 0.999
            best_epochs = val_metrics[val_metrics[val_metric] >= threshold].index
            best_epoch = best_epochs[0]
            epochs_to_best = best_epoch + 1

            train_metrics = train_metrics.groupby('epoch')[metric].mean()
            best_training = train_metrics.max()
    
            results.append({
                'Model': model_name,
                'Metric': metric,
                'Best Val': f"{best_validation:.4f}",
                'Avg Val': f"{avarage_validation:.4f}",
                'Training': f"{best_training:.4f}",
                'Epochs to Best': epochs_to_best,
            })
        
        return pd.DataFrame(results)


    def calculate_loss_metrics(self) -> pd.DataFrame:
        """
        Calculate loss metrics for multiple models.
        :returns pd.DataFrame: DataFrame with model loss metrics
        """
        results = []

        for model_name in self.model_names:
            model_dir = os.path.join(self.models_path, model_name)
            train_file = os.path.join(model_dir, 'train_metrics.csv')
        
            metrics_df = pd.read_csv(train_file)
            metrics_df = metrics_df.groupby('epoch')['loss'].mean().reset_index()
        
            initial_loss = metrics_df['loss'].iloc[0] if not metrics_df.empty else None
            final_loss = metrics_df['loss'].iloc[-1] if not metrics_df.empty else None
        
            num_epochs = len(metrics_df)
            if num_epochs > 1 and initial_loss is not None and final_loss is not None:
                convergence_rate = (initial_loss - final_loss) / (num_epochs - 1)
            else:
                convergence_rate = None
            
            loss_volatility = metrics_df['loss'].std() if not metrics_df.empty else None
        
            results.append({
                'Model': model_name,
                'Initial Loss': round(initial_loss, 4) if initial_loss is not None else None,
                'Final Loss': round(final_loss, 4) if final_loss is not None else None,
                'Rate of Convergence': round(convergence_rate, 4) if convergence_rate is not None else None,
                'Loss Volatility': round(loss_volatility, 4) if loss_volatility is not None else None
            })
    
        return pd.DataFrame(results)

    def calculate_convergence_metrics(self) -> pd.DataFrame:
        """
        Calculate learning convergence pattern metrics for each model using various metrics files.
        :returns pd.DataFrame: DataFrame with convergence metrics
        """
        results = []
        
        for model_name in self.model_names:
            model_dir = os.path.join(self.models_path, model_name)
            val_file = os.path.join(model_dir, 'val_metrics.csv')
            train_file = os.path.join(model_dir, 'train_metrics.csv')
                    
            val_df = pd.read_csv(val_file)
            train_df = pd.read_csv(train_file)

            convergence = self._model_convergence(val_df, train_df)
            epochs_to_stabilize = self._model_stability(val_df)
            oscillation_level = self._model_oscillation(val_df, epochs_to_stabilize)
            final_vs_best = self._model_final_vs_best(val_df)
            
            results.append({
                'Model': model_name,
                'Converged': 'Yes' if convergence else 'No',
                'Epochs to Stabilize': epochs_to_stabilize,
                'Oscillation After Convergence': oscillation_level,
                'Final vs. Best Epoch': final_vs_best
            })

        return pd.DataFrame(results)

    def _model_convergence(self, val_df, train_df):
        initial_loss = train_df['loss'].iloc[0]
        final_loss = train_df['loss'].iloc[-1]
        relative_loss_decrease = (initial_loss - final_loss) / initial_loss
        
        num_epochs = len(val_df)
        last_third_idx = int(num_epochs * 2/3)
        last_third_f1 = val_df['val_f1_score'].iloc[last_third_idx:]
        
        f1_mean = last_third_f1.mean()
        f1_std = last_third_f1.std()
        f1_relative_std = f1_std / f1_mean if f1_mean > 0 else float('inf')
        
        best_f1 = val_df['val_f1_score'].max()
        final_f1 = val_df['val_f1_score'].iloc[-1]
        maintains_performance = (final_f1 / best_f1 >= 0.5) if best_f1 > 0 else False
        
        f1_reasonably_stable = f1_relative_std < 0.3
        significant_progress = relative_loss_decrease > 0.3
        
        return significant_progress and f1_reasonably_stable and maintains_performance
    
    def _model_stability(self, val_df):
        num_epochs = len(val_df)
        epochs_to_stabilize = num_epochs  # default value
        window_size = min(4, max(2, int(num_epochs * 0.12)))

        stability_threshold = 0.15
        required_stable_windows = 2
        stable_window_count = 0
        
        for i in range(window_size, num_epochs - window_size):
            window1 = val_df['val_f1_score'].iloc[i-window_size:i].mean()
            window2 = val_df['val_f1_score'].iloc[i:i+window_size].mean()
            
            if window1 < 0.001:
                continue
                
            relative_change = abs(window2 - window1) / window1
            if relative_change < stability_threshold:
                stable_window_count += 1
                if stable_window_count >= required_stable_windows:
                    epochs_to_stabilize = i - required_stable_windows + 1
                    break
            else:
                stable_window_count = 0
        
        return epochs_to_stabilize

    def _model_oscillation(self, val_df, epochs_to_stabilize):
        post_stable_values = val_df['val_f1_score'].iloc[epochs_to_stabilize:]
        
        if len(post_stable_values) < 3:
            return "Insufficient Data"
        
        mean_f1 = post_stable_values.mean()
        if mean_f1 <= 0.001:
            return "Unstable (Low F1)"
        
        oscillation = post_stable_values.std() / mean_f1
        
        if oscillation < 0.08:
            oscillation_level = "Low"
        elif oscillation < 0.20:
            oscillation_level = "Medium"
        elif oscillation < 0.5:
            oscillation_level = "High"
        else:
            oscillation_level = "Very High"
        
        return oscillation_level

    def _model_final_vs_best(self, val_df):
        best_epoch_idx = val_df['val_f1_score'].idxmax()
        best_f1 = val_df['val_f1_score'].iloc[best_epoch_idx]
        final_f1 = val_df['val_f1_score'].iloc[-1]
        
        if best_f1 < 0.001:
            return "Inconclusive (Low F1)"
        
        final_vs_best_pct = (final_f1 / best_f1) * 100
        best_epoch_num = best_epoch_idx + 1
        total_epochs = len(val_df)
        
        final_vs_best = f"{round(final_vs_best_pct, 1)}%"
        
        if final_f1 >= 0.95 * best_f1:
            descriptor = "Close to Best"
        elif final_f1 >= 0.8 * best_f1:
            descriptor = "Moderate Drop"
        else:
            descriptor = "Significant Drop"
        
        epoch_context = ""
        if best_epoch_num == total_epochs:
            epoch_context = " (Best at Final)"
        elif best_epoch_num > 0.8 * total_epochs:
            epoch_context = " (Best Near End)"
        elif best_epoch_num < 0.3 * total_epochs:
            epoch_context = " (Best at Early Stage)"

        return f"{final_vs_best} ({descriptor}{epoch_context})"

    def _load_label_mappings(self, has_no_finding_output: bool) -> List[str]:
        mappings_path = f"{TF_RECORD_DATASET}/label_mappings.csv"
        df = pd.read_csv(mappings_path)
        df = df.sort_values('Index')

        if not has_no_finding_output:
            df = df[df['Index'] != NO_FINDING_CLASS_IDX]

        return df['Label'].tolist()
            
    def get_metric_table(self, metric_name: str) -> pd.DataFrame:
        """Get a specific metric table for all classes and models.
        :param metric_name: Name of the metric to display (f1_score, auc, precision, recall, true_positives_rate, etc.)
        :returns: DataFrame with classes as rows and models as columns
        """
        all_classes = set()
        data = []
        
        for model_name in self.model_names:
            df = pd.read_csv(self._metrics_file(model_name))
            all_classes.update(df['class_name'].tolist())
            
        for class_name in sorted(all_classes):
            row = {'Class': class_name}
            for model_name in self.model_names:
                df = pd.read_csv(self._metrics_file(model_name))
                class_metrics = df[df['class_name'] == class_name]
                if not class_metrics.empty:
                    if metric_name in class_metrics.columns:
                        value = class_metrics[metric_name].iloc[0]
                        if isinstance(value, (int, float)):
                            row[model_name] = f"{value:.3f}"
                        else:
                            row[model_name] = str(value)
                    else:
                        if metric_name == 'true_positives_rate':
                            tp = class_metrics['true_positives'].iloc[0]
                            total = tp + class_metrics['false_negatives'].iloc[0]
                            row[model_name] = f"{(tp / total if total > 0 else 0):.3f}"
                        elif metric_name == 'false_positives_rate':
                            fp = class_metrics['false_positives'].iloc[0]
                            total = fp + class_metrics['true_negatives'].iloc[0]
                            row[model_name] = f"{(fp / total if total > 0 else 0):.3f}"
                        elif metric_name == 'false_negatives_rate':
                            fn = class_metrics['false_negatives'].iloc[0]
                            total = fn + class_metrics['true_positives'].iloc[0]
                            row[model_name] = f"{(fn / total if total > 0 else 0):.3f}"
                        elif metric_name == 'true_negatives_rate':
                            tn = class_metrics['true_negatives'].iloc[0]
                            total = tn + class_metrics['false_positives'].iloc[0]
                            row[model_name] = f"{(tn / total if total > 0 else 0):.3f}"
                else:
                    row[model_name] = 'N/A'
            data.append(row)
            
        return pd.DataFrame(data)

    def calculate_confusion_rates(self) -> Dict[str, pd.DataFrame]:
        """Generate confusion matrix rates per class for all models."""
        rates = {
            'true_positives_rate': pd.DataFrame(),
            'false_positives_rate': pd.DataFrame(),
            'false_negatives_rate': pd.DataFrame(),
            'true_negatives_rate': pd.DataFrame()
        }
        
        all_classes = set()
        for model_name in self.model_names:
            df = pd.read_csv(self._metrics_file(model_name))
            all_classes.update(df['class_name'].tolist())
        
        for rate_name in rates.keys():
            data = []
            for class_name in sorted(all_classes):
                row = {'Class': class_name}
                for model_name in self.model_names:
                    df = pd.read_csv(self._metrics_file(model_name))
                    class_metrics = df[df['class_name'] == class_name]
                    if not class_metrics.empty:
                        if rate_name == 'true_positives_rate':
                            tp = class_metrics['true_positives'].iloc[0]
                            total = tp + class_metrics['false_negatives'].iloc[0]
                            rate = tp / total if total > 0 else 0
                        elif rate_name == 'false_positives_rate':
                            fp = class_metrics['false_positives'].iloc[0]
                            total = fp + class_metrics['true_negatives'].iloc[0]
                            rate = fp / total if total > 0 else 0
                        elif rate_name == 'false_negatives_rate':
                            fn = class_metrics['false_negatives'].iloc[0]
                            total = fn + class_metrics['true_positives'].iloc[0]
                            rate = fn / total if total > 0 else 0
                        elif rate_name == 'true_negatives_rate':
                            tn = class_metrics['true_negatives'].iloc[0]
                            total = tn + class_metrics['false_positives'].iloc[0]
                            rate = tn / total if total > 0 else 0
                        row[model_name] = f"{rate:.3f}"
                    else:
                        row[model_name] = 'N/A'
                data.append(row)
            rates[rate_name] = pd.DataFrame(data)
        
        return rates

    def _metrics_file(self, model_name: str) -> str:
        return os.path.join(self._model_dir(model_name), f"model_metrics.csv")
    
    def _model_dir(self, model_name: str) -> str:
        return os.path.join(self.models_path, model_name)