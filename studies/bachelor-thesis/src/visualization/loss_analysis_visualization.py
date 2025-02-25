import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss_analysis(metrics_path: str, metrics_file: str) -> None:
    df = pd.read_csv(os.path.join(metrics_path, metrics_file))
    top_classes = df.groupby('class_name')['loss_contribution'].mean().nlargest(5).index
    
    fig = plt.figure(figsize=(20, 15))
    
    ax1 = plt.subplot(2, 2, 1)
    for class_name in top_classes:
        class_data = df[df['class_name'] == class_name]
        ax1.plot(class_data['epoch'], class_data['loss_contribution'], 
                marker='o', label=class_name)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Contribution')
    ax1.set_title('Top 5 Classes Loss Contribution Over Time')
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    latest_epoch = df['epoch'].max()
    latest_data = df[df['epoch'] == latest_epoch]
    confidence_cols = ['high_confidence_ratio', 'medium_confidence_ratio', 
                      'uncertain_ratio', 'low_confidence_ratio']
    
    latest_data['total_confidence'] = latest_data['high_confidence_ratio'] + latest_data['medium_confidence_ratio']
    top_confident_classes = latest_data.nlargest(5, 'total_confidence')
    x = np.arange(len(top_confident_classes))
    bottom = np.zeros(len(top_confident_classes))
    
    for col in confidence_cols:
        ax2.bar(x, top_confident_classes[col], bottom=bottom, 
                label=col.replace('_', ' ').replace('ratio', ''))
        bottom += top_confident_classes[col]
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_confident_classes['class_name'], rotation=45, ha='right')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Confidence Distribution for Top 5 Classes (Latest Epoch)')
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    scatter = ax3.scatter(latest_data['avg_confidence_correct'],
                         latest_data['avg_confidence_incorrect'],
                         alpha=0.6)
    
    min_val = min(latest_data['avg_confidence_correct'].min(),
                 latest_data['avg_confidence_incorrect'].min())
    max_val = max(latest_data['avg_confidence_correct'].max(),
                 latest_data['avg_confidence_incorrect'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    ax3.set_xlabel('Average Confidence (Correct Predictions)')
    ax3.set_ylabel('Average Confidence (Incorrect Predictions)')
    ax3.set_title('Confidence Comparison: Correct vs Incorrect Predictions')
    ax3.grid(True)

    ax4 = plt.subplot(2, 2, 4)
    top_tp_classes = latest_data.nlargest(5, 'true_positives')
    
    x = np.arange(len(top_tp_classes))
    width = 0.35
    
    ax4.bar(x - width/2, top_tp_classes['true_positives'], width, 
            label='True Positives')
    ax4.bar(x + width/2, top_tp_classes['false_positives'], width, 
            label='False Positives')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_tp_classes['class_name'], rotation=45, ha='right')
    ax4.set_ylabel('Count')
    ax4.set_title('True Positives vs False Positives (Latest Epoch)')
    ax4.legend()

    plt.tight_layout()
    plt.show()
