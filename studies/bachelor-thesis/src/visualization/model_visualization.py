import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def _get_csv(metrics_path: str, model_name: str, file_name: str) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame"""
    file_path = f"{metrics_path}/{model_name}/{file_name}"
    return pd.read_csv(file_path)

def _loss_progression(ax: plt.Axes, df: pd.DataFrame, title: str, xlabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Value')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.plot(df.index, df[f'loss'], label='Loss', color='blue')
    return None

def _loss_contribution(ax: plt.Axes, df: pd.DataFrame, top_classes: int) -> None:
    top_classes = df.groupby('class_name')['loss_contribution'].mean().nlargest(top_classes).index

    for class_name in top_classes:
        class_data = df[df['class_name'] == class_name]
        ax.plot(class_data['epoch'], class_data['loss_contribution'], marker='o', label=class_name)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Contribution')
    ax.set_title('Top 5 Classes Loss Contribution Over Time')
    ax.grid(True)
    ax.legend()
    return None


def _confidence_distribution(ax: plt.Axes, df: pd.DataFrame, top_classes: int, epoch: int=None) -> None:
    latest_epoch = epoch or df['epoch'].max()
    latest_data = df[df['epoch'] == latest_epoch]

    confidence_cols = ['high_confidence_ratio', 'medium_confidence_ratio', 'uncertain_ratio', 'low_confidence_ratio']
    latest_data = latest_data.copy()
    latest_data['total_confidence'] = latest_data['high_confidence_ratio'] + latest_data['medium_confidence_ratio']

    top_confident_classes = latest_data.nlargest(top_classes, 'total_confidence')

    x = np.arange(len(top_confident_classes))
    bottom = np.zeros(len(top_confident_classes))

    for col in confidence_cols:
        ax.bar(x, top_confident_classes[col], bottom=bottom, label=col.replace('_', ' ').replace('ratio', ''))
        bottom += top_confident_classes[col]

    ax.set_xticks(x)
    ax.set_xticklabels(top_confident_classes['class_name'], rotation=45, ha='right')
    ax.set_ylabel('Ratio')
    ax.set_title('Confidence Distribution for Top 5 Classes (Latest Epoch)')
    ax.legend()
    return None

def _confidence_comparison(ax: plt.Axes, df: pd.DataFrame, epoch: int=None) -> None:
    latest_epoch = epoch or df['epoch'].max()
    latest_data = df[df['epoch'] == latest_epoch]
    ax.scatter(latest_data['avg_confidence_correct'], latest_data['avg_confidence_incorrect'], alpha=0.6)
    min_val = min(latest_data['avg_confidence_correct'].min(), latest_data['avg_confidence_incorrect'].min())
    max_val = max(latest_data['avg_confidence_correct'].max(), latest_data['avg_confidence_incorrect'].max())

    ax.scatter(latest_data['avg_confidence_correct'], latest_data['avg_confidence_incorrect'], alpha=0.6)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    ax.set_xlabel('Average Confidence (Correct Predictions)')
    ax.set_ylabel('Average Confidence (Incorrect Predictions)')
    ax.set_title('Confidence Comparison: Correct vs Incorrect Predictions')
    ax.grid(True)

def _positives_comparison(ax: plt.Axes, df: pd.DataFrame, top_classes: int, epoch: int=None) -> None:
    latest_epoch = epoch or df['epoch'].max()
    latest_data = df[df['epoch'] == latest_epoch]
    top_tp_classes = latest_data.nlargest(top_classes, 'true_positives')

    x = np.arange(len(top_tp_classes))

    width = 0.35
    ax.bar(x - width/2, top_tp_classes['true_positives'], width, label='True Positives')
    ax.bar(x + width/2, top_tp_classes['false_positives'], width, label='False Positives')

    ax.set_xticks(x)
    ax.set_xticklabels(top_tp_classes['class_name'], rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('True Positives vs False Positives (Latest Epoch)')
    return None

def _performance_metrics(ax: plt.Axes, df: pd.DataFrame, metric: str) -> None:
    ax.bar(df['class_name'], df[metric], color='purple')
    ticks = np.arange(len(df['class_name']))
    ax.set_xticks(ticks)
    ax.set_xticklabels(df['class_name'], rotation=45, ha='right')
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"{metric}")
    ax.grid(True, alpha=0.3)
    return None

def _performance_metric_distribution(ax: plt.Axes, df: pd.DataFrame, top_classes: int) -> None:
    top_classes = df.nlargest(top_classes, 'f1_score')
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for idx, row in top_classes.iterrows():
        values = row[metrics].values.flatten()
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=row['class_name'])
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Top 5 Classes Performance Overview')
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))
    return None

def _performance_metric_by_class(ax: plt.Axes, df: pd.DataFrame) -> None:
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(df['class_name']))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df[metric], width, label=metric.replace('_', ' ').title())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['class_name'], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return None

def plot_combined_analysis(metrics_path: str, model_name: str) -> None:
    """
    Create a combined visualization of model training process
    :params metrics_path: Path to the metrics directory
    :params model_name: Name of the model (used for file naming)
    """
    # Get Data Frames of files
    training_df = _get_csv(metrics_path, model_name, 'train_metrics.csv')

    validation_df = _get_csv(metrics_path, model_name, 'val_metrics.csv')
    validation_df["loss"] = validation_df["val_loss"]

    loss_analysis_df = _get_csv(metrics_path, model_name, 'loss_analysis_metrics.csv')
    model_metrics_df = _get_csv(metrics_path, model_name, 'model_metrics.csv')

    fig = plt.figure(figsize=(30, 30))
    plt.rcParams.update({'font.size': 18})
    gs = fig.add_gridspec(4, 3)

    # Loss Analysis
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    _loss_progression(ax1, training_df, 'Training Loss Progression', 'Batch')
    _loss_progression(ax2, validation_df, 'Validation Loss Progression', 'Epoch')
    _loss_contribution(ax3, loss_analysis_df, 5)

    # Confidance Analysis
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    _confidence_distribution(ax4, loss_analysis_df, 5)
    _confidence_comparison(ax5, loss_analysis_df)
    _positives_comparison(ax6, loss_analysis_df, 5)

    # Model Performance
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    _performance_metrics(ax7, model_metrics_df, 'f1_score')
    _performance_metrics(ax8, model_metrics_df, 'recall')
    _performance_metrics(ax9, model_metrics_df, 'precision')

    # Model Analysis
    ax10 = fig.add_subplot(gs[3, 0], projection='polar')
    ax11 = fig.add_subplot(gs[3, 1:3])
    _performance_metric_distribution(ax10, model_metrics_df, 5)
    _performance_metric_by_class(ax11, model_metrics_df)

    plt.tight_layout()
    plt.show()