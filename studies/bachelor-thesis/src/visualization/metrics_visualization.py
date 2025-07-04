import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List

def plot_metrics(metrics_path: str, metrics_file: str, prefix: str='') -> None:
    df = pd.read_csv(os.path.join(metrics_path, metrics_file))

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(df.index, df[f'{prefix}loss'], label='Loss', color='blue')
    ax1.plot(df.index, df[f'{prefix}accuracy'], label='Accuracy', color='green')
    ax1.set_title('Training Progression')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Value')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.plot(df.index, df[f'{prefix}precision'], label='Precision', color='purple')
    ax2.plot(df.index, df[f'{prefix}recall'], label='Recall', color='orange')
    ax2.plot(df.index, df[f'{prefix}f1_score'], label='F1 Score', color='red')
    ax2.plot(df.index, df[f'{prefix}auc'], label='AUC', color='brown')
    ax2.set_title('Classification Performance')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Value')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    for ax in [ax1, ax2]:
        epoch_boundaries = df[df['epoch'] != df['epoch'].shift()].index
        for boundary in epoch_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_metric_comparison(models_path: str, model_names: List[str], metric: str, output_path: str) -> None:
    """
    Create a line chart comparing training and validation metrics across different models
    
    :param models_path: Path to the directory containing model metrics
    :param model_names: List of model names to compare
    :param metric: Name of the metric to compare (e.g., 'f1_score') without 'val_' prefix
    :param output_path: Subdirectory name to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Prepare data structures to track max values for annotations
    train_max_values = {}
    train_max_epochs = {}
    train_overall_max_model = None
    train_overall_max_value = -float('inf')
    
    val_max_values = {}
    val_max_epochs = {}
    val_overall_max_model = None
    val_overall_max_value = -float('inf')
    
    for idx, model_name in enumerate(model_names):
        train_file = os.path.join(models_path, model_name, 'train_metrics.csv')
        if os.path.exists(train_file):
            train_df = pd.read_csv(train_file)
            train_epochs = train_df['epoch'].unique()
            train_values = train_df.groupby('epoch')[metric].mean().values
            
            if len(train_values) > 0:
                max_idx = np.argmax(train_values)
                max_value = train_values[max_idx]
                max_epoch = train_epochs[max_idx]
                train_max_values[model_name] = max_value
                train_max_epochs[model_name] = max_epoch
                
                if max_value > train_overall_max_value:
                    train_overall_max_value = max_value
                    train_overall_max_model = model_name
                    train_overall_max_epoch = max_epoch
            
            ax1.plot(
                train_epochs, 
                train_values,
                marker='o', 
                label=model_name,
                linewidth=2,
                markersize=5,
                color=colors[idx],
                alpha=0.8
            )
        
        val_file = os.path.join(models_path, model_name, 'val_metrics.csv')
        if os.path.exists(val_file):
            val_df = pd.read_csv(val_file)
            val_epochs = val_df['epoch'].unique()
            val_metric_name = f'val_{metric}'
            if val_metric_name in val_df.columns:
                val_values = val_df.groupby('epoch')[val_metric_name].mean().values
                
                if len(val_values) > 0:
                    max_idx = np.argmax(val_values)
                    max_value = val_values[max_idx]
                    max_epoch = val_epochs[max_idx]
                    val_max_values[model_name] = max_value
                    val_max_epochs[model_name] = max_epoch
                    
                    if max_value > val_overall_max_value:
                        val_overall_max_value = max_value
                        val_overall_max_model = model_name
                        val_overall_max_epoch = max_epoch
                
                # Plot validation data
                ax2.plot(
                    val_epochs, 
                    val_values,
                    marker='o', 
                    label=model_name,
                    linewidth=2,
                    markersize=5,
                    color=colors[idx],
                    alpha=0.8
                )
    
    if train_overall_max_model:
        for idx, model_name in enumerate(model_names):
            if model_name == train_overall_max_model and model_name in train_max_epochs:
                ax1.plot(
                    train_max_epochs[model_name],
                    train_max_values[model_name],
                    marker='*',
                    markersize=15,
                    color=colors[idx],
                    markeredgecolor='black',
                    markeredgewidth=1
                )
                
                ax1.annotate(
                    f'Best: {train_max_values[model_name]:.4f}',
                    xy=(train_max_epochs[model_name], train_max_values[model_name]),
                    xytext=(train_max_epochs[model_name] + 1, train_max_values[model_name] + 0.02),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
                )
    
    for idx, model_name in enumerate(model_names):
        if model_name != train_overall_max_model and model_name in train_max_epochs:
            ax1.annotate(
                f'{train_max_values[model_name]:.4f}',
                xy=(train_max_epochs[model_name], train_max_values[model_name]),
                xytext=(train_max_epochs[model_name], train_max_values[model_name] + 0.01),
                fontsize=8,
                ha='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
            )
    
    if val_overall_max_model:
        for idx, model_name in enumerate(model_names):
            if model_name == val_overall_max_model and model_name in val_max_epochs:
                ax2.plot(
                    val_max_epochs[model_name],
                    val_max_values[model_name],
                    marker='*',
                    markersize=15,
                    color=colors[idx],
                    markeredgecolor='black',
                    markeredgewidth=1
                )
                
                ax2.annotate(
                    f'Best: {val_max_values[model_name]:.4f}',
                    xy=(val_max_epochs[model_name], val_max_values[model_name]),
                    xytext=(val_max_epochs[model_name] + 1, val_max_values[model_name] + 0.02),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
                )
    
    for idx, model_name in enumerate(model_names):
        if model_name != val_overall_max_model and model_name in val_max_epochs:
            ax2.annotate(
                f'{val_max_values[model_name]:.4f}',
                xy=(val_max_epochs[model_name], val_max_values[model_name]),
                xytext=(val_max_epochs[model_name], val_max_values[model_name] + 0.01),
                fontsize=8,
                ha='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
            )
    
    metric_title = metric.replace('_', ' ').title()
    ax1.set_title(f'Training {metric_title} per Epoch', fontsize=14)
    ax2.set_title(f'Validation {metric_title} per Epoch', fontsize=14)
    
    ax1.set_ylabel(f'Training {metric_title}', fontsize=12)
    ax2.set_ylabel(f'Validation {metric_title}', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    
    # Adjust y-axis limits to provide more space between title and annotations
    if train_overall_max_value > 0:
        # Add 20% additional space above the highest value
        ax1.set_ylim(top=train_overall_max_value * 1.15)
        
    if val_overall_max_value > 0:
        # Add 20% additional space above the highest value
        ax2.set_ylim(top=val_overall_max_value * 1.15)
    
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)
        ax.minorticks_on()
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Models', 
               title_fontsize=12, fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9, hspace=0.3)
    
    output_dir = os.path.join(models_path, output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f'model_comparison_{metric}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Combined training and validation comparison saved to: {file_path}")