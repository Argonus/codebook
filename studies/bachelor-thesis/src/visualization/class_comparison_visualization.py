"""
Visualization tools for comparing model performance across different classes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union


def plot_per_class_metric_comparison(
    metrics_path: str,
    model_names: List[str],
    metric: str = "f1_score",
    title: Optional[str] = None,
    figsize: tuple = (20, 12),
    save_path: Optional[str] = None
) -> None:
    """
    Create a grid of plots comparing the same metric across different models for each class.
    
    Args:
        metrics_path: Path to metrics directory containing model_metrics.csv files for each model
        model_names: List of model names to compare
        metric: Metric to compare (e.g., 'accuracy', 'precision', 'recall', 'f1_score', 'auc')
        title: Optional title for the figure
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    # Set up colors and markers for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', '*', 'x', 'P']
    
    # Load data for each model
    model_data = {}
    class_names = set()
    
    for model_name in model_names:
        metrics_file = os.path.join(metrics_path, model_name, "model_metrics.csv")
        if not os.path.exists(metrics_file):
            print(f"Warning: Metrics file not found for model {model_name}: {metrics_file}")
            continue
            
        df = pd.read_csv(metrics_file)
        model_data[model_name] = df
        class_names.update(df['class_name'].values)
    
    if not model_data:
        print("No valid model data found.")
        return
    
    # Sort class names for consistent ordering
    class_names = sorted(list(class_names))
    
    # Calculate grid dimensions
    n_classes = len(class_names)
    cols = min(5, n_classes)
    rows = (n_classes + cols - 1) // cols  # Ceiling division
    
    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    else:
        fig.suptitle(f'Per-Class {metric.replace("_", " ").title()} Comparison', fontsize=16, y=0.98)
    
    # Plot data for each class
    for i, class_name in enumerate(class_names):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Plot data for each model
        for j, (model_name, df) in enumerate(model_data.items()):
            class_data = df[df['class_name'] == class_name]
            if not class_data.empty:
                ax.bar(j, class_data[metric].values[0], 
                       label=model_name if i == 0 else "", 
                       color=colors[j % len(colors)],
                       alpha=0.7)
                
                # Add value text on top of each bar
                ax.text(j, class_data[metric].values[0] + 0.01, 
                        f"{class_data[metric].values[0]:.3f}", 
                        ha='center', va='bottom', 
                        fontsize=8)
        
        # Set title and format axes
        ax.set_title(class_name, fontsize=10)
        ax.set_xticks(range(len(model_data)))
        ax.set_xticklabels([m.replace('Simplified_DensNet_', 'v') for m in model_data.keys()], 
                           rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Only set y label for leftmost plots
        if col == 0:
            ax.set_ylabel(metric.replace('_', ' ').title())
    
    # Hide empty subplots
    for i in range(n_classes, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)
    
    # Add legend outside the plots
    handles, labels = [], []
    for j, model_name in enumerate(model_data.keys()):
        patch = plt.Rectangle((0, 0), 1, 1, color=colors[j % len(colors)], alpha=0.7)
        handles.append(patch)
        # Shorten model names for the legend
        labels.append(model_name.replace('Simplified_DensNet_', 'v'))
    
    fig.legend(handles, labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.95), ncol=len(model_data), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_metric_evolution(
    metrics_path: str,
    model_names: List[str],
    class_name: str,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot the evolution of multiple metrics for a specific class across different epochs.
    
    Args:
        metrics_path: Path to metrics directory containing training metrics files
        model_names: List of model names to compare
        class_name: Name of the class to analyze
        metrics: List of metrics to plot (default: ['loss_contribution', 'accuracy', 'precision', 'recall', 'f1_score'])
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    if metrics is None:
        metrics = ['loss_contribution', 'accuracy', 'precision', 'recall', 'f1_score']
    
    # Set up colors and markers for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', '*']
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, model_name in enumerate(model_names):
            # For loss analysis
            if metric == 'loss_contribution':
                file_path = os.path.join(metrics_path, model_name, "loss_analysis_metrics.csv")
            else:
                # For other metrics during training
                file_path = os.path.join(metrics_path, model_name, "training_metrics.csv")
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            
            # Filter for the specific class if available
            if 'class_name' in df.columns:
                class_data = df[df['class_name'] == class_name]
            else:
                # For training metrics that might not have class-specific data
                class_data = df
                
            if class_data.empty:
                print(f"Warning: No data found for class {class_name} in {file_path}")
                continue
                
            # Get epochs
            if 'epoch' in class_data.columns:
                x_values = class_data['epoch']
            else:
                x_values = np.arange(len(class_data))
                
            # Check if the metric column exists
            if metric in class_data.columns:
                ax.plot(x_values, class_data[metric], 
                        marker=markers[j % len(markers)],
                        color=colors[j % len(colors)],
                        label=f"{model_name.replace('Simplified_DensNet_', 'v')}")
            else:
                print(f"Warning: Metric {metric} not found in data for {model_name}")
        
        # Set title and format axes
        ax.set_title(f"{metric.replace('_', ' ').title()} for {class_name}")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend(loc='best')
        
    # Set common x-label for the figure
    axes[-1].set_xlabel('Epoch')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    metrics_path = "path/to/metrics"
    model_names = ["Simplified_DensNet_v1", "Simplified_DensNet_v2", "Simplified_DensNet_v3"]
    
    # Plot F1 score comparison for all classes
    plot_per_class_metric_comparison(
        metrics_path=metrics_path,
        model_names=model_names,
        metric="f1_score",
        title="F1 Score Comparison Across Models"
    )
    
    # Plot evolution of metrics for a specific class
    plot_metric_evolution(
        metrics_path=metrics_path,
        model_names=model_names,
        class_name="Pneumonia",  # Replace with an actual class name
        metrics=['accuracy', 'precision', 'recall', 'f1_score']
    )
