# Model Evolution Summary

This document tracks the evolution of DenseNet architectures developed during this research, including performance metrics, architectural changes, and observed behaviors.

## Performance Comparison

### Overall Metrics Comparison
This metrics are collected from train and validation datasets logs.

#### F1 Score
F1 Score is metric that shows balance between precision and recall, in this case it is used to measure how well the model is able to identify positive cases (diseases) in the validation and training datasets. This is a main metric of a project and can be calculated using following algorithm

```latex
F1 = 2 * (precision * recall) / (precision + recall)
```

| Model                | Best Val F1 | Avg Val F1 | Best Training F1 | Epochs to Best |
|----------------------|-------------|------------|------------------|----------------|
|SimplifiedDenseNetV1_1|   0.3969    | 0.3447     | 0.9221           | 33             |

#### AUC ROC Score

AUC ROC is a metrics that shows how well the model is able to separate positive and negative cases. This is a secondary metric.

| Model                | Best Val AUC | Avg Val AUC | Training AUC | Epochs to Best |
|----------------------|--------------|-------------|--------------|----------------|
|SimplifiedDenseNetV1_1| 0.7612       | 0.7009      | 0.9871       | 11             |

#### Precision

Precision is a metrics that shows how many of the predicted positive cases are actually positive. This is a secondary metric that can be calculated using following algorithm

```latex
Precision = True Positives / (True Positives + False Positives)
```

| Model                | Best Val Precision | Avg Val Precision | Training Precision | Epochs to Best |
|----------------------|--------------------|-------------------|--------------------|----------------|
|SimplifiedDenseNetV1_1| 0.7902             | 0.5435            | 0.9647             | 4              |

#### Recall

Recall is a metrics that shows how many of the actual positive cases are predicted as positive. This is a secondary metric that can be calculated using following algorithm

```latex
Recall = True Positives / (True Positives + False Negatives)
```

| Model                | Best Val Recall | Avg Val Recall | Training Recall | Epochs to Best |
|----------------------|-----------------|----------------|-----------------|----------------|
|SimplifiedDenseNetV1_1| 0.3824          | 0.3214         | 0.8993          | 43             |

### Detailed Class Performance Comparison
This metrics are collected from test dataset logs.

#### F1 Score per Class

| Class              | SDV1_1  |
|--------------------|---------|
| Atelectasis        | 0.207   |
| Cardiomegaly       | 0.238   |
| Consolidation      | 0.120   |
| Edema              | 0.108   |
| Effusion           | 0.428   |
| Emphysema          | 0.160   |
| Fibrosis           | 0.021   |
| Hernia             | 0.000   |
| Infiltration       | 0.278   |
| Mass               | 0.181   |
| No Finding         | 0.649   |
| Nodule             | 0.064   |
| Pleural_Thickening | 0.079   |
| Pneumonia          | 0.000   |
| Pneumothorax       | 0.283   |

#### Precision per Class

| Class              | SDV1_1  |
|--------------------|---------|
| Atelectasis        | 0.312   |
| Cardiomegaly       | 0.365   |
| Consolidation      | 0.136   |
| Edema              | 0.205   |
| Effusion           | 0.430   |
| Emphysema          | 0.230   |
| Fibrosis           | 0.088   |
| Hernia             | 0.000   |
| Infiltration       | 0.295   |
| Mass               | 0.248   |
| No Finding         | 0.646   |
| Nodule             | 0.159   |
| Pleural_Thickening | 0.127   |
| Pneumonia          | 0.000   |
| Pneumothorax       | 0.420   |

#### Recall per Class

| Class              | SDV1_1  |
|--------------------|---------|
| Atelectasis        | 0.155   |
| Cardiomegaly       | 0.176   |
| Consolidation      | 0.108   |
| Edema              | 0.073   |
| Effusion           | 0.427   |
| Emphysema          | 0.122   |
| Fibrosis           | 0.012   |
| Hernia             | 0.000   |
| Infiltration       | 0.262   |
| Mass               | 0.143   |
| No Finding         | 0.652   |
| Nodule             | 0.040   |
| Pleural_Thickening | 0.058   |
| Pneumonia          | 0.000   |
| Pneumothorax       | 0.213   |

### Loss Metrics Evolution

**Loss** metric allows to track how well our model is learning over time. We can see how many mistakes, our model is doing over time. In summary, we can see that our model learned something based on diff between initial and final loss. And we aim to have a low initial loss and a low final loss.

- **Initial Loss** is a metric that shows the initial loss of the model. In summary it shows how well our model was learning at the beginning. 
- **Final Loss** is a metric that shows the final loss of the model. In summary its a loss at last epoch. In summary it shows how well our model was learning at the end.
- **Rate of Convergence** is a metric that shows the rate of convergence of the model. Convergence shows us how quickly model moves to end state. Bigger value, in theory means model is learning faster.
- **Loss Volatility** is a metric that shows the volatility of the loss of the model. Volatility is a how much the loss changes over time. Lower value means more stable learning process, bigger value means more unstable learning process.

| Model                | Initial Loss | Final Loss | Rate of Convergence | Loss Volatility |
|----------------------|--------------|------------|---------------------|-----------------|
|SimplifiedDenseNetV1_1| 0.7782       | 0.0784     | 0.0149              | 0.1459          |

### Training Dynamics Comparison

#### Learning Convergence Patterns

This section shows the learning convergence patterns of the models. It shows how well the models have converged and how they have stabilized.

- **Converged** is a metric that shows if the model has converged. In summary it shows if the model has reached a stable state.
- **Epochs to Stabilize** is a metric that shows the number of epochs it took for the model to stabilize. In summary it shows how many epochs it took for the model to reach a stable state.
- **Oscillation After Convergence** is a metric that shows the oscillation of the model after convergence. In summary it shows how much the model oscillates after it has reached a stable state.
- **Final vs. Best Epoch** is a metric that shows the final vs. best epoch of the model. In summary it shows how much the model has improved over time.

| Model                | Converged | Epochs to Stabilize | Oscillation After Convergence | Final vs. Best Epoch |
|----------------------|-----------|---------------------|-------------------------------|----------------------|
|SimplifiedDenseNetV1_1| Yes       | 5                   | Medium                        | 99.5% (Close to Best (Best Near End)) |

### Test Samples Rate Evolution

#### Test True Positive Rate Evolution

| Class              | SDV1_1  |
|--------------------|---------|
| Atelectasis        | 0.155   |
| Cardiomegaly       | 0.176   |
| Consolidation      | 0.108   |
| Edema              | 0.073   |
| Effusion           | 0.427   |
| Emphysema          | 0.122   |
| Fibrosis           | 0.012   |
| Hernia             | 0.000   |
| Infiltration       | 0.262   |
| Mass               | 0.143   |
| No Finding         | 0.652   |
| Nodule             | 0.040   |
| Pleural_Thickening | 0.058   |
| Pneumonia          | 0.000   |
| Pneumothorax       | 0.213   |

#### Test False Positive Rate Evolution

| Class              | SDV1_1  |
|--------------------|---------|
| Atelectasis        | 0.043   |
| Cardiomegaly       | 0.008   |
| Consolidation      | 0.032   |
| Edema              | 0.006   |
| Effusion           | 0.084   |
| Emphysema          | 0.010   |
| Fibrosis           | 0.002   |
| Hernia             | 0.000   |
| Infiltration       | 0.148   |
| Mass               | 0.026   |
| No Finding         | 0.359   |
| Nodule             | 0.014   |
| Pleural_Thickening | 0.013   |
| Pneumonia          | 0.000   |
| Pneumothorax       | 0.016   |

#### Test False Negative Rate Evolution

| Class              | SDV1_1  |
|--------------------|---------|
| Atelectasis        | 0.845   |
| Cardiomegaly       | 0.824   |
| Consolidation      | 0.892   |
| Edema              | 0.927   |
| Effusion           | 0.573   |
| Emphysema          | 0.878   |
| Fibrosis           | 0.988   |
| Hernia             | 1.000   |
| Infiltration       | 0.738   |
| Mass               | 0.857   |
| No Finding         | 0.348   |
| Nodule             | 0.960   |
| Pleural_Thickening | 0.942   |
| Pneumonia          | 1.000   |
| Pneumothorax       | 0.787   |

#### Test True Negative Rate Evolution

| Class              | SDV1_1  |
|--------------------|---------|
| Atelectasis        | 0.957   |
| Cardiomegaly       | 0.992   |
| Consolidation      | 0.968   |
| Edema              | 0.994   |
| Effusion           | 0.916   |
| Emphysema          | 0.990   |
| Fibrosis           | 0.998   |
| Hernia             | 1.000   |
| Infiltration       | 0.852   |
| Mass               | 0.974   |
| No Finding         | 0.641   |
| Nodule             | 0.986   |
| Pleural_Thickening | 0.987   |
| Pneumonia          | 1.000   |
| Pneumothorax       | 0.984   |

## Architectural Changes and Impacts

### SimplifiedDenseNetV1_1 (Baseline)
Baseline model with default configuration, no changes applied only Early Stopping was enabled to reduce training time.

#### Configuration
- Epochs: 33(100)
- Batch size: 128
- Optimizer: Adam(0.0001)
- Loss: Binary Crossentropy
- Image Augmentation: None
- Architecture: SimplifiedDenseNet
- Learning rate schedule: ReduceLROnPlateau

### SimplifiedDenseNetV1_2

#### Configuration
- **Changes from v1:**
- **Hypothesis:**
- **Impact:**