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
    model_metrics_df = _get_csv(metrics_path, model_name, 'model_metrics.csv')

    fig = plt.figure(figsize=(25, 20))
    plt.rcParams.update({'font.size': 14})
    gs = fig.add_gridspec(3, 3)

    # Training and Validation Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    _loss_progression(ax1, training_df, 'Training Loss Progression', 'Batch')
    _loss_progression(ax2, validation_df, 'Validation Loss Progression', 'Epoch')
    
    # Plot F1 score progression
    ax3.set_title('Validation F1 Score Progression')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.plot(validation_df.index, validation_df['val_f1_score'], label='Validation F1 Score', color='green')
    ax3.legend()

    # Model Performance Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    _performance_metrics(ax4, model_metrics_df, 'f1_score')
    _performance_metrics(ax5, model_metrics_df, 'recall')
    _performance_metrics(ax6, model_metrics_df, 'precision')

    # Model Analysis
    ax7 = fig.add_subplot(gs[2, 0], projection='polar')
    ax8 = fig.add_subplot(gs[2, 1:3])
    _performance_metric_distribution(ax7, model_metrics_df, 5)
    _performance_metric_by_class(ax8, model_metrics_df)

    plt.tight_layout()
    plt.show()

def plot_model_class_comparison(metrics_path: str, model_names: list[str], metric: str, models_without_no_finding: list[str]) -> None:
    """
    Create a comparison visualization of different models' performance for a specific metric across all classes
    
    :param metrics_path: Path to the metrics directory
    :param model_names: List of model names to compare
    :param metric: Metric to compare (e.g., 'f1_score', 'precision', 'recall', 'accuracy')
    :param models_without_no_finding: List of model names that don't include the No Finding class
    """
    # Setup the plot
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    # Load data for each model
    dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{metrics_path}/{model_name}/model_metrics.csv")
        df['model'] = model_name
        if model_name in models_without_no_finding:
            # Skip the No Finding class for models that don't include it
            df = df[df['class_name'] != 'No Finding']
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Get unique classes (excluding No Finding for models that don't have it)
    classes = sorted(combined_df['class_name'].unique())
    
    # Calculate positions for bars
    num_models = len(model_names)
    bar_width = 0.8 / num_models
    r = np.arange(len(classes))
    
    # Plot bars for each model
    for idx, model_name in enumerate(model_names):
        model_data = combined_df[combined_df['model'] == model_name]
        # Ensure data is in the same order as classes
        model_data = model_data.set_index('class_name').reindex(classes).reset_index()
        position = r + idx * bar_width
        plt.bar(position, model_data[metric], bar_width, label=model_name)
    
    # Customize the plot
    plt.xlabel('Classes')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Comparison Across Models')
    plt.xticks(r + bar_width * (num_models-1)/2, classes, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()