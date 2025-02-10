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
            
            best_validation    = val_metrics[val_metric].max()
            avarage_validation = val_metrics[val_metric].mean()
            epochs_to_best     = val_metrics[val_metric].idxmax() + 1

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

    def calculate_tp_count_evolution(self) -> pd.DataFrame:
        """
        Calculate the evolution of True Positive Count across training
        using class-specific metrics from loss_analysis_metrics.csv files.
        :returns pd.DataFrame: DataFrame with TP count evolution metrics
        """
        results = []
        
        for model_name in self.model_names:
            model_dir = os.path.join(self.models_path, model_name)
            metrics_file = os.path.join(model_dir, 'loss_analysis_metrics.csv')
            
            loss_df = pd.read_csv(metrics_file)
            num_epochs = int(loss_df['epoch'].max()) + 1
            
            if num_epochs < 3:
                print(f"Not enough epochs in metrics for {model_name}")
                continue
                    
            early_end = max(1, num_epochs // 3)
            mid_start = early_end
            mid_end = 2 * (num_epochs // 3)
            late_start = mid_end
            
            df_filtered = loss_df[loss_df['class_name'] != 'No Finding']
            tp_by_epoch = df_filtered.groupby('epoch')['true_positives'].sum().reset_index()

            early_tp = tp_by_epoch[tp_by_epoch['epoch'] < early_end]['true_positives'].mean()
            mid_tp = tp_by_epoch[(tp_by_epoch['epoch'] >= mid_start) & (tp_by_epoch['epoch'] < mid_end)]['true_positives'].mean()
            final_tp = tp_by_epoch[tp_by_epoch['epoch'] >= late_start]['true_positives'].mean()

            results.append({
                'Model': model_name,
                'Early TP Count': f"{early_tp:.1f}",
                'Mid TP Count': f"{mid_tp:.1f}",
                'Final TP Count': f"{final_tp:.1f}"
            })
            
        return pd.DataFrame(results)

    def analyze_class_performance(self):
        """
        Analyze class performance for multiple models.
        """
        
        results = []
        test_ds = load_dataset(f"{TF_RECORD_DATASET}/test.tfrecord", TF_BUFFER_SIZE)
        class_weights = calculate_class_weights(test_ds, NUM_CLASSES)
                
        for model_name in self.model_names:
            model_dir = os.path.join(self.models_path, model_name)
            model_metrics = pd.read_csv(os.path.join(model_dir, "model_metrics.csv"))

            has_no_finding_output = not model_name in self.models_without_no_finding
            class_names = self._load_label_mappings(has_no_finding_output)
            
            most_improved = self._get_most_improved_data(model_metrics, class_names, class_weights)
            most_improved_str = ", ".join(most_improved)
            
            problematic = self._get_problematic_data(model_metrics, class_names, class_weights)
            problematic_str = ", ".join(problematic)
            
            results.append({
                "Model": model_name,
                "Most Improved Classes": most_improved_str if most_improved_str else "None above threshold",
                "Problematic Classes": problematic_str if problematic_str else "None identified",
            })
        
        return pd.DataFrame(results)

    def _load_label_mappings(self, has_no_finding_output: bool) -> List[str]:
        mappings_path = f"{TF_RECORD_DATASET}/label_mappings.csv"
        df = pd.read_csv(mappings_path)
        df = df.sort_values('Index')

        if not has_no_finding_output:
            df = df[df['Index'] != NO_FINDING_CLASS_IDX]

        return df['Label'].tolist()

    def _get_most_improved_data(self, model_metrics: pd.DataFrame, class_names: List[str], class_weights: List[float]) -> List[str]:
        most_improved_with_weights = []     
        most_improved = model_metrics.sort_values('f1_score', ascending=False).head(3)
        
        for _, row in most_improved.iterrows():
            class_name = row['class_name']
            class_idx = [i for i, name in enumerate(class_names) if name == class_name]
            
            if class_idx:
                weight = float(class_weights[class_idx[0]])
                most_improved_with_weights.append(f"{class_name} (AUC={row['auc']:.2f}, F1={row['f1_score']:.2f}, wt={weight:.1f})")
            else:
                most_improved_with_weights.append(f"{class_name} (AUC={row['auc']:.2f}, F1={row['f1_score']:.2f})")

        return most_improved_with_weights

    def _get_problematic_data(self, model_metrics: pd.DataFrame, class_names: List[str], class_weights: List[float]) -> List[str]:
        problematic_with_weights = []        
        problematic = model_metrics[(model_metrics['auc'] > 0.65) & (model_metrics['f1_score'] < 0.05)].sort_values('auc', ascending=False).head(3)
        for _, row in problematic.iterrows():
            class_name = row['class_name']
            class_idx = [i for i, name in enumerate(class_names) if name == class_name]
            
            if class_idx:
                weight = float(class_weights[class_idx[0]])
                problematic_with_weights.append(f"{class_name} (AUC={row['auc']:.2f}, F1={row['f1_score']:.2f}, wt={weight:.1f})")
            else:
                problematic_with_weights.append(f"{class_name} (AUC={row['auc']:.2f}, F1={row['f1_score']:.2f})")
        
        return problematic_with_weights
            
    def _metrics_file(self, model_name: str) -> str:
        return os.path.join(self._model_dir(model_name), f"model_metrics.csv")
    
    def _model_dir(self, model_name: str) -> str:
        return os.path.join(self.models_path, model_name)