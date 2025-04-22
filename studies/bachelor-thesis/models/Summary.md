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
|SimplifiedDenseNetV1_2|   0.4119    | 0.3614     | 0.6581           | 54             |
|SimplifiedDenseNetV1_3|   0.1285    | 0.0529     | 0.1331           | 28             |

#### AUC ROC Score

AUC ROC is a metrics that shows how well the model is able to separate positive and negative cases. This is a secondary metric.

| Model                | Best Val AUC | Avg Val AUC | Training AUC | Epochs to Best |
|----------------------|--------------|-------------|--------------|----------------|
|SimplifiedDenseNetV1_1| 0.7612       | 0.7009      | 0.9871       | 11             |
|SimplifiedDenseNetV1_2| 0.7793       | 0.7411      | 0.9226       | 25             |
|SimplifiedDenseNetV1_3| 0.7477       | 0.7114      | 0.7920       | 30             |

#### Precision

Precision is a metrics that shows how many of the predicted positive cases are actually positive. This is a secondary metric that can be calculated using following algorithm

```latex
Precision = True Positives / (True Positives + False Positives)
```

| Model                | Best Val Precision | Avg Val Precision | Training Precision | Epochs to Best |
|----------------------|--------------------|-------------------|--------------------|----------------|
|SimplifiedDenseNetV1_1| 0.7902             | 0.5435            | 0.9647             | 4              |
|SimplifiedDenseNetV1_2| 0.7427             | 0.6048            | 0.8321             | 5              |
|SimplifiedDenseNetV1_3| 0.4483             | 0.3341            | 0.4968             | 26             |

#### Recall

Recall is a metrics that shows how many of the actual positive cases are predicted as positive. This is a secondary metric that can be calculated using following algorithm

```latex
Recall = True Positives / (True Positives + False Negatives)
```

| Model                | Best Val Recall | Avg Val Recall | Training Recall | Epochs to Best |
|----------------------|-----------------|----------------|-----------------|----------------|
|SimplifiedDenseNetV1_1| 0.3824          | 0.3214         | 0.8993          | 43             |
|SimplifiedDenseNetV1_2| 0.3933          | 0.3376         | 0.5970          | 39             |
|SimplifiedDenseNetV1_3| 0.0851          | 0.0378         | 0.0945          | 28             |

### Detailed Class Performance Comparison
This metrics are collected from test dataset logs.

#### F1 Score per Class

| Class              | SDV1_1  | SDV1_2  | SDV1_3  |
|--------------------|---------|---------|---------|
| Atelectasis        | 0.207   | 0.216   | 0.074   |
| Cardiomegaly       | 0.238   | 0.315   | 0.309   |
| Consolidation      | 0.120   | 0.022   | 0.000   |
| Edema              | 0.108   | 0.154   | 0.217   |
| Effusion           | 0.428   | 0.473   | 0.324   |
| Emphysema          | 0.160   | 0.193   | 0.144   |
| Fibrosis           | 0.021   | 0.000   | 0.000   |
| Hernia             | 0.000   | 0.057   | 0.208   |
| Infiltration       | 0.278   | 0.294   | 0.102   |
| Mass               | 0.181   | 0.227   | 0.000   |
| No Finding         | 0.649   | 0.658   | 0.151   |
| Nodule             | 0.064   | 0.057   | 0.000   |
| Pleural_Thickening | 0.079   | 0.025   | 0.000   |
| Pneumonia          | 0.000   | 0.000   | 0.000   |
| Pneumothorax       | 0.283   | 0.345   | 0.112   |

#### Precision per Class

| Class              | SDV1_1  | SDV1_2  | SDV1_3  |
|--------------------|---------|---------|---------|
| Atelectasis        | 0.312   | 0.343   | 0.439   |
| Cardiomegaly       | 0.365   | 0.381   | 0.234   |
| Consolidation      | 0.136   | 0.320   | 0.000   |
| Edema              | 0.205   | 0.226   | 0.163   |
| Effusion           | 0.430   | 0.493   | 0.588   |
| Emphysema          | 0.230   | 0.368   | 0.253   |
| Fibrosis           | 0.088   | 0.000   | 0.000   |
| Hernia             | 0.000   | 1.000   | 0.186   |
| Infiltration       | 0.295   | 0.339   | 0.426   |
| Mass               | 0.248   | 0.378   | 0.000   |
| No Finding         | 0.646   | 0.685   | 0.797   |
| Nodule             | 0.159   | 0.265   | 0.000   |
| Pleural_Thickening | 0.127   | 0.117   | 0.000   |
| Pneumonia          | 0.000   | 0.000   | 0.000   |
| Pneumothorax       | 0.420   | 0.407   | 0.485   |

#### Recall per Class

| Class              | SDV1_1  | SDV1_2  | SDV1_3  |
|--------------------|---------|---------|---------|
| Atelectasis        | 0.155   | 0.158   | 0.040   |
| Cardiomegaly       | 0.176   | 0.268   | 0.454   |
| Consolidation      | 0.108   | 0.012   | 0.000   |
| Edema              | 0.073   | 0.117   | 0.322   |
| Effusion           | 0.427   | 0.454   | 0.223   |
| Emphysema          | 0.122   | 0.130   | 0.101   |
| Fibrosis           | 0.012   | 0.000   | 0.000   |
| Hernia             | 0.000   | 0.029   | 0.235   |
| Infiltration       | 0.262   | 0.259   | 0.058   |
| Mass               | 0.143   | 0.163   | 0.000   |
| No Finding         | 0.652   | 0.634   | 0.084   |
| Nodule             | 0.040   | 0.032   | 0.000   |
| Pleural_Thickening | 0.058   | 0.014   | 0.000   |
| Pneumonia          | 0.000   | 0.000   | 0.000   |
| Pneumothorax       | 0.213   | 0.300   | 0.063   |

### Loss Metrics Evolution

**Loss** metric allows to track how well our model is learning over time. We can see how many mistakes, our model is doing over time. In summary, we can see that our model learned something based on diff between initial and final loss. And we aim to have a low initial loss and a low final loss.

- **Initial Loss** is a metric that shows the initial loss of the model. In summary it shows how well our model was learning at the beginning. 
- **Final Loss** is a metric that shows the final loss of the model. In summary its a loss at last epoch. In summary it shows how well our model was learning at the end.
- **Rate of Convergence** is a metric that shows the rate of convergence of the model. Convergence shows us how quickly model moves to end state. Bigger value, in theory means model is learning faster.
- **Loss Volatility** is a metric that shows the volatility of the loss of the model. Volatility is a how much the loss changes over time. Lower value means more stable learning process, bigger value means more unstable learning process.

| Model                | Initial Loss | Final Loss | Rate of Convergence | Loss Volatility |
|----------------------|--------------|------------|---------------------|-----------------|
|SimplifiedDenseNetV1_1| 0.7782       | 0.0784     | 0.0149              | 0.1459          |
|SimplifiedDenseNetV1_2| 0.7842       | 0.1581     | 0.0092              | 0.1073          |
|SimplifiedDenseNetV1_3| 0.7668       | 0.2010     | 0.0135              | 0.1218          |

### Training Dynamics Comparison

#### Learning Convergence Patterns

This section shows the learning convergence patterns of the models. It shows how well the models have converged and how they have stabilized.

- **Converged** is a metric that shows if the model has converged. In summary it shows if the model has reached a stable state.
- **Epochs to Stabilize** is a metric that shows the number of epochs it took for the model to stabilize. In summary it shows how many epochs it took for the model to reach a stable state.
- **Oscillation After Convergence** is a metric that shows the oscillation of the model after convergence. In summary it shows how much the model oscillates after it has reached a stable state.
- **Final vs. Best Epoch** is a metric that shows the final vs. best epoch of the model. In summary it shows how much the model has improved over time.

| Model                | Converged | Epochs to Stabilize | Oscillation After Convergence | Final vs. Best Epoch                  |
|----------------------|-----------|---------------------|-------------------------------|---------------------------------------|
|SimplifiedDenseNetV1_1| Yes       | 5                   | Medium                        | 99.5% (Close to Best (Best Near End)) |
|SimplifiedDenseNetV1_2| Yes       | 4                   | Medium                        | 99.2% (Close to Best (Best Near End)) |
|SimplifiedDenseNetV1_3| Yes       | 19                  | High                          | 84.8% (Moderate Drop)                 |

### Test Samples Rate Evolution

#### Test True Positive Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3  |
|--------------------|--------|--------|---------|
| Atelectasis        | 0.155  | 0.158  | 0.040   |
| Cardiomegaly       | 0.176  | 0.268  | 0.454   |
| Consolidation      | 0.108  | 0.012  | 0.000   |
| Edema              | 0.073  | 0.117  | 0.322   |
| Effusion           | 0.427  | 0.454  | 0.223   |
| Emphysema          | 0.122  | 0.130  | 0.101   |
| Fibrosis           | 0.012  | 0.000  | 0.000   |
| Hernia             | 0.000  | 0.029  | 0.235   |
| Infiltration       | 0.262  | 0.259  | 0.058   |
| Mass               | 0.143  | 0.163  | 0.000   |
| No Finding         | 0.652  | 0.634  | 0.084   |
| Nodule             | 0.040  | 0.032  | 0.000   |
| Pleural_Thickening | 0.058  | 0.014  | 0.000   |
| Pneumonia          | 0.000  | 0.000  | 0.000   |
| Pneumothorax       | 0.213  | 0.300  | 0.063   |

#### Test False Positive Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3  |
|--------------------|--------|--------|---------|
| Atelectasis        | 0.043  | 0.038  | 0.006   |
| Cardiomegaly       | 0.008  | 0.012  | 0.041   |
| Consolidation      | 0.032  | 0.001  | 0.000   |
| Edema              | 0.006  | 0.009  | 0.037   |
| Effusion           | 0.084  | 0.069  | 0.023   |
| Emphysema          | 0.010  | 0.006  | 0.007   |
| Fibrosis           | 0.002  | 0.000  | 0.000   |
| Hernia             | 0.000  | 0.000  | 0.002   |
| Infiltration       | 0.148  | 0.119  | 0.018   |
| Mass               | 0.026  | 0.016  | 0.000   |
| No Finding         | 0.359  | 0.292  | 0.021   |
| Nodule             | 0.014  | 0.006  | 0.000   |
| Pleural_Thickening | 0.013  | 0.004  | 0.000   |
| Pneumonia          | 0.000  | 0.000  | 0.000   |
| Pneumothorax       | 0.016  | 0.024  | 0.004   |

#### Test False Negative Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3  |
|--------------------|--------|--------|---------|
| Atelectasis        | 0.845  | 0.842  | 0.960   |
| Cardiomegaly       | 0.824  | 0.732  | 0.546   |
| Consolidation      | 0.892  | 0.988  | 1.000   |
| Edema              | 0.927  | 0.883  | 0.678   |
| Effusion           | 0.573  | 0.546  | 0.777   |
| Emphysema          | 0.878  | 0.870  | 0.899   |
| Fibrosis           | 0.988  | 1.000  | 1.000   |
| Hernia             | 1.000  | 0.971  | 0.765   |
| Infiltration       | 0.738  | 0.741  | 0.942   |
| Mass               | 0.857  | 0.837  | 1.000   |
| No Finding         | 0.348  | 0.366  | 0.916   |
| Nodule             | 0.960  | 0.968  | 1.000   |
| Pleural_Thickening | 0.942  | 0.986  | 1.000   |
| Pneumonia          | 1.000  | 1.000  | 1.000   |
| Pneumothorax       | 0.787  | 0.700  | 0.937   |

#### Test True Negative Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 |
|--------------------|--------|--------|--------|
| Atelectasis        | 0.957  | 0.962  | 0.994  |
| Cardiomegaly       | 0.992  | 0.988  | 0.959  |
| Consolidation      | 0.968  | 0.999  | 1.000  |
| Edema              | 0.994  | 0.991  | 0.963  |
| Effusion           | 0.916  | 0.931  | 0.977  |
| Emphysema          | 0.990  | 0.994  | 0.993  |
| Fibrosis           | 0.998  | 1.000  | 1.000  |
| Hernia             | 1.000  | 1.000  | 0.998  |
| Infiltration       | 0.852  | 0.881  | 0.982  |
| Mass               | 0.974  | 0.984  | 1.000  |
| No Finding         | 0.641  | 0.708  | 0.979  |
| Nodule             | 0.986  | 0.994  | 1.000  |
| Pleural_Thickening | 0.987  | 0.996  | 1.000  |
| Pneumonia          | 1.000  | 1.000  | 1.000  |

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
- Class weights: None

### SimplifiedDenseNetV1_2

#### Configuration
- **Changes from v1:** 
  - Added Basic Image Augmentation
- **Hypothesis:** 
  - Image augmentation will improve model generalization by 
    exposing it to more varied training samples and reducing overfitting
- **Impact:** 
  - Improved model robustness to variations in input images
  - Better generalization on test data
  - Reduced overfitting compared to v1

## SimplifiedDenseNetV1_3

#### Configuration
- **Changes from v2:** 
  - Added class weights to model training
- **Hypothesis:** 
  - Class weights will help address class imbalance 
    issues and improve performance on minority classes
- **Impact:**
  - Better performance on minority classes
  - Most classes showed decreased performance 
  - Significantly decreased overall F1 score (from 0.4119 to 0.1285)
