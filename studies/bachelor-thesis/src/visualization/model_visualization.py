import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def _get_csv(metrics_path: str, model_name: str, file_name: str) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame"""
    file_path = f"{metrics_path}/{model_name}/{file_name}"
    return pd.read_csv(file_path)

def _performance_metrics(ax: plt.Axes, df: pd.DataFrame, metric: str) -> None:
    """
    Plot performance metrics as a bar chart, sorted by metric value.
    Highlights the best class in bright green and worst class in bright red.
    """
    sorted_df = df.sort_values(by=metric, ascending=False).copy()
    
    default_color = 'purple'
    colors = [default_color] * len(sorted_df)
    
    if len(sorted_df) > 0:
        colors[0] = '#00c853'
        if len(sorted_df) > 1:
            colors[-1] = '#d50000'
    
    ax.bar(sorted_df['class_name'], sorted_df[metric], color=colors)
        
    ticks = np.arange(len(sorted_df['class_name']))
    ax.set_xticks(ticks)
    ax.set_xticklabels(sorted_df['class_name'], rotation=45, ha='right')
    
    for i, v in enumerate(sorted_df[metric]):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)
    
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
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
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
    
    # Find the epoch with best validation F1 score
    best_f1_idx = validation_df['val_f1_score'].idxmax()
    best_f1_epoch = validation_df.loc[best_f1_idx].name if isinstance(best_f1_idx, (int, np.integer)) else best_f1_idx

    fig = plt.figure(figsize=(25, 25))
    plt.rcParams.update({'font.size': 14})
    # Add main title for the entire figure
    fig.suptitle(f"Model Metrics {model_name}", fontsize=20, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(4, 3)
    
    # First row - training metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    _loss_progression(ax1, training_df, 'Training Loss Progression', 'Batch')    
    _metric_progression(ax2, training_df, 'Training F1 Score Progression', 'Batch', 'f1_score', 'F1 Score')
    _learning_rate_progression(ax3, training_df, 'Training Learning Rate Progression', 'Batch')    

    # Second row - validation metrics with best epoch marker
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    _loss_progression(ax4, validation_df, 'Validation Loss Progression', 'Epoch')    
    _metric_progression(ax5, validation_df, 'Validation F1 Score Progression', 'Epoch', 'val_f1_score', 'F1 Score')
    _metric_progression(ax6, validation_df, 'Validation AUC Progression', 'Epoch', 'val_auc', 'AUC')
    _mark_best_epoch(ax4, validation_df, best_f1_epoch, 'loss')
    _mark_best_epoch(ax5, validation_df, best_f1_epoch, 'val_f1_score')
    _mark_best_epoch(ax6, validation_df, best_f1_epoch, 'val_auc')

    # Third row - class performance metrics
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    _performance_metrics(ax7, model_metrics_df, 'f1_score')
    _performance_metrics(ax8, model_metrics_df, 'recall')
    _performance_metrics(ax9, model_metrics_df, 'precision')
    
    # Fourth row - additional analysis
    ax10 = fig.add_subplot(gs[3, 0], projection='polar')
    ax11 = fig.add_subplot(gs[3, 1:3])
    _performance_metric_distribution(ax10, model_metrics_df, 5)
    _performance_metric_by_class(ax11, model_metrics_df)

    plt.tight_layout()
    
    output_path = f"{metrics_path}/{model_name}/combined_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def _loss_progression(ax: plt.Axes, df: pd.DataFrame, title: str, xlabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.plot(df.index, df[f'loss'], label='Loss', color='blue')
    
    return None

def _metric_progression(ax: plt.Axes, df: pd.DataFrame, title: str, xlabel: str, metric: str, metric_title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.plot(df.index, df[metric], label=f'{metric_title}', color='green')
    return None

def _learning_rate_progression(ax: plt.Axes, df: pd.DataFrame, title: str, xlabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Learning Rate')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.plot(df.index, df['learning_rate'], label='Learning Rate', color='blue')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    return None

def plot_model_class_comparison(models_path: str, model_names: list[str], metric: str, models_without_no_finding: list[str], output_path: str) -> None:
    """
    Create a comparison visualization of different models' performance for a specific metric across all classes
    
    :param models_path: Path to the models directory
    :param model_names: List of model names to compare
    :param metric: Metric to compare (e.g., 'f1_score', 'precision', 'recall', 'accuracy')
    :param models_without_no_finding: List of model names that don't include the No Finding class
    :param output_path: Subdirectory name to save the visualization
    """
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{models_path}/{model_name}/model_metrics.csv")
        df['model'] = model_name
        if model_name in models_without_no_finding:
            df = df[df['class_name'] != 'No Finding']
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    class_avg_performance = combined_df.groupby('class_name')[metric].mean().reset_index()
    sorted_classes = class_avg_performance.sort_values(by=metric, ascending=False)['class_name'].tolist()
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    num_models = len(model_names)
    bar_width = 0.8 / num_models
    r = np.arange(len(sorted_classes))
    
    best_model_idx = {}
    for class_name in sorted_classes:
        class_data = combined_df[combined_df['class_name'] == class_name]
        if not class_data.empty:
            best_model = class_data.loc[class_data[metric].idxmax()]
            best_model_idx[(class_name, best_model['model'])] = True
    
    for idx, model_name in enumerate(model_names):
        model_data = combined_df[combined_df['model'] == model_name]
        model_data = model_data.set_index('class_name').reindex(sorted_classes).reset_index()
        model_data = model_data.fillna(0)
        
        position = r + idx * bar_width
        bars = plt.bar(position, model_data[metric], bar_width, 
                 label=model_name, color=model_colors[idx], alpha=0.8)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)                
                class_name = sorted_classes[i]
                if (class_name, model_name) in best_model_idx:
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.06,
                            '★', ha='center', va='bottom', fontsize=12, color='gold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Comparison Across Models')
    plt.xticks(r + bar_width * (num_models-1)/2, sorted_classes, rotation=45, ha='right')
    plt.legend(loc='upper right')

    for idx, model_name in enumerate(model_names):
        model_mean = combined_df[combined_df['model'] == model_name][metric].mean()
        plt.axhline(y=model_mean, color=model_colors[idx], linestyle='--', alpha=0.5)
        plt.text(len(sorted_classes)-1, model_mean, f'{model_name} avg: {model_mean:.3f}', 
                 ha='right', va='bottom', color=model_colors[idx], fontsize=9)
    
    plt.tight_layout()
    
    output_dir = os.path.join(models_path, output_path)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'model_class_comparison_{metric}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Class comparison visualization saved to: {file_path}")

def _mark_best_epoch(ax: plt.Axes, df: pd.DataFrame, best_epoch: int, metric: str) -> None:
    """
    Mark the best F1 score epoch on any validation chart
    :param ax: The matplotlib axes to mark
    :param df: The dataframe containing the data
    :param best_epoch: The epoch with the best F1 score
    :param metric: The metric to use for marking the value
    """
    if best_epoch in df.index and metric in df.columns:
        value_at_best_epoch = df.loc[best_epoch, metric]
        
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.plot(best_epoch, value_at_best_epoch, 'ro', markersize=8)
        ax.annotate(f'Best F1 Epoch: {best_epoch}', 
                    xy=(best_epoch, value_at_best_epoch),
                    xytext=(best_epoch, value_at_best_epoch + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1),
                    ha='center',  # Horizontal alignment: center 
                    va='baseline',  # Default vertical alignment
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    return None