import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_model_metrics(metrics_path: str, metrics_file: str) -> None:
    df = pd.read_csv(os.path.join(metrics_path, metrics_file))
    fig = plt.figure(figsize=(20, 15))

    ax1 = plt.subplot(2, 2, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(df['class_name']))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width, df[metric], width, label=metric.replace('_', ' ').title())

    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(df['class_name'], rotation=45, ha='right')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics by Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(df['class_name'], df['f1_score'], color='purple')
    ax2.set_xticklabels(df['class_name'], rotation=45, ha='right')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Scores by Class')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, (3, 4), projection='polar')

    top_classes = df.nlargest(5, 'f1_score')
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)

    angles = np.concatenate((angles, [angles[0]]))

    for idx, row in top_classes.iterrows():
        values = row[metrics].values.flatten()
        values = np.concatenate((values, [values[0]]))
        ax3.plot(angles, values, 'o-', linewidth=2, label=row['class_name'])
        ax3.fill(angles, values, alpha=0.25)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics)
    ax3.set_title('Top 5 Classes Performance Overview')
    ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))

    plt.tight_layout()
    plt.show()
