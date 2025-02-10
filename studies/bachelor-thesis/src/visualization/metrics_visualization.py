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

def plot_metric_comparison(models_path: str, model_names: List[str], file: str, metric: str) -> None:
    plt.figure(figsize=(12, 8))
    
    for model_name in model_names:
        metrics_file = os.path.join(models_path, model_name, file)
        if not os.path.exists(metrics_file):
            continue
            
        df = pd.read_csv(metrics_file)
        plt.plot(
            df['epoch'].unique(), 
            df.groupby('epoch')[metric].mean(),
            marker='o', 
            label=model_name,
            linewidth=2,
            markersize=6
        )
    
    plt.title(f'Model Comparison: {metric} per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(f'{metric}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Models', title_fontsize=12, fontsize=10)
    
    # Add minor gridlines for better readability
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.minorticks_on()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()