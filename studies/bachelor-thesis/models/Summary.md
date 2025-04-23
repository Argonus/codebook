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
|DenseNetV1_1          |   0.4244	   | 0.3491	    | 0.7167	         | 32             |
|DenseNetV1_2          |   0.1338	   | 0.0917	    | 0.1483	         | 98             |

#### AUC ROC Score

AUC ROC is a metrics that shows how well the model is able to separate positive and negative cases. This is a secondary metric.

| Model                | Best Val AUC | Avg Val AUC | Training AUC | Epochs to Best |
|----------------------|--------------|-------------|--------------|----------------|
|SimplifiedDenseNetV1_1| 0.7612       | 0.7009      | 0.9871       | 11             |
|SimplifiedDenseNetV1_2| 0.7793       | 0.7411      | 0.9226       | 25             |
|SimplifiedDenseNetV1_3| 0.7477       | 0.7114      | 0.7920       | 30             |
|DenseNetV1_1          | 0.7814       | 0.7438      | 0.9369       | 18             |
|DenseNetV1_2          | 0.7501       | 0.7375      | 0.8020       | 37             |

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
|DenseNetV1_1          | 0.7880             | 0.6161            | 0.8561             | 9              |
|DenseNetV1_2          | 0.5104             | 0.4121            | 0.5094             | 9              |

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
|DenseNetV1_1          | 0.4047          | 0.3233         | 0.6574          | 33             |
|DenseNetV1_2          | 0.0898          | 0.0631         | 0.1070          | 9              |

### Detailed Class Performance Comparison
This metrics are collected from test dataset logs.

#### F1 Score per Class

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 |
|--------------------|--------|--------|--------|-------|-------|
| Atelectasis        | 0.207  | 0.216  | 0.074  | 0.281 | 0.049 |
| Cardiomegaly       | 0.238  | 0.315  | 0.309  | 0.255 | 0.315 |
| Consolidation      | 0.120  | 0.022  | 0.000  | 0.037 | 0.000 |
| Edema              | 0.108  | 0.154  | 0.217  | 0.097 | 0.222 |
| Effusion           | 0.428  | 0.473  | 0.324  | 0.487 | 0.380 |
| Emphysema          | 0.160  | 0.193  | 0.144  | 0.185 | 0.192 |
| Fibrosis           | 0.021  | 0.000  | 0.000  | 0.008 | 0.077 |
| Hernia             | 0.000  | 0.057  | 0.208  | 0.000 | 0.250 |
| Infiltration       | 0.278  | 0.294  | 0.102  | 0.296 | 0.071 |
| Mass               | 0.181  | 0.227  | 0.000  | 0.264 | 0.021 |
| No Finding         | 0.649  | 0.658  | 0.151  | 0.665 | 0.142 |
| Nodule             | 0.064  | 0.057  | 0.000  | 0.052 | 0.000 |
| Pleural_Thickening | 0.079  | 0.025  | 0.000  | 0.030 | 0.000 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 |
| Pneumothorax       | 0.283  | 0.345  | 0.112  | 0.367 | 0.172 |

#### Precision per Class

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 |
|--------------------|--------|--------|--------|-------|-------|
| Atelectasis        | 0.312  | 0.343  | 0.439  | 0.316 | 0.377 |
| Cardiomegaly       | 0.365  | 0.381  | 0.234  | 0.443 | 0.277 |
| Consolidation      | 0.136  | 0.320  | 0.000  | 0.209 | 0.000 |
| Edema              | 0.205  | 0.226  | 0.163  | 0.286 | 0.186 |
| Effusion           | 0.430  | 0.493  | 0.588  | 0.507 | 0.550 |
| Emphysema          | 0.230  | 0.368  | 0.253  | 0.359 | 0.290 |
| Fibrosis           | 0.088  | 0.000  | 0.000  | 0.333 | 0.083 |
| Hernia             | 0.000  | 1.000  | 0.186  | 0.000 | 0.318 |
| Infiltration       | 0.295  | 0.339  | 0.426  | 0.360 | 0.415 |
| Mass               | 0.248  | 0.378  | 0.000  | 0.436 | 0.529 |
| No Finding         | 0.646  | 0.685  | 0.797  | 0.703 | 0.822 |
| Nodule             | 0.159  | 0.265  | 0.000  | 0.388 | 0.000 |
| Pleural_Thickening | 0.127  | 0.117  | 0.000  | 0.250 | 0.000 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 |
| Pneumothorax       | 0.420  | 0.407  | 0.485  | 0.357 | 0.444 |

#### Recall per Class

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 |
|--------------------|--------|--------|--------|-------|-------|
| Atelectasis        | 0.155  | 0.158  | 0.040  | 0.253 | 0.026 |
| Cardiomegaly       | 0.176  | 0.268  | 0.454  | 0.179 | 0.365 |
| Consolidation      | 0.108  | 0.012  | 0.000  | 0.020 | 0.000 |
| Edema              | 0.073  | 0.117  | 0.322  | 0.058 | 0.275 |
| Effusion           | 0.427  | 0.454  | 0.223  | 0.468 | 0.290 |
| Emphysema          | 0.122  | 0.130  | 0.101  | 0.125 | 0.144 |
| Fibrosis           | 0.012  | 0.000  | 0.000  | 0.004 | 0.071 |
| Hernia             | 0.000  | 0.029  | 0.235  | 0.000 | 0.206 |
| Infiltration       | 0.262  | 0.259  | 0.058  | 0.251 | 0.039 |
| Mass               | 0.143  | 0.163  | 0.000  | 0.189 | 0.010 |
| No Finding         | 0.652  | 0.634  | 0.084  | 0.632 | 0.078 |
| Nodule             | 0.040  | 0.032  | 0.000  | 0.028 | 0.000 |
| Pleural_Thickening | 0.058  | 0.014  | 0.000  | 0.016 | 0.000 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 |
| Pneumothorax       | 0.213  | 0.300  | 0.063  | 0.378 | 0.106 |

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
|DenseNetV1_1          | 2.1351       | 0.1413     | 0.0433              | 0.3716          |
|DenseNetV1_2          | 2.1788       | 0.4187     | 0.0178              | 0.2473          |

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
|DenseNetV1_1          | Yes       | 4                   | Medium                        | 98.4% (Close to Best (Best Near End)) |
|DenseNetV1_2          | Yes       | 19                  | Medium                        | 88.6% (Moderate Drop (Best Near End)) |

### Test Samples Rate Evolution

#### Test True Positive Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 |
|--------------------|--------|--------|--------|-------|-------|
| Atelectasis        | 0.155  | 0.158  | 0.040  | 0.253 | 0.026 |
| Cardiomegaly       | 0.176  | 0.268  | 0.454  | 0.179 | 0.365 |
| Consolidation      | 0.108  | 0.012  | 0.000  | 0.020 | 0.000 |
| Edema              | 0.073  | 0.117  | 0.322  | 0.058 | 0.275 |
| Effusion           | 0.427  | 0.454  | 0.223  | 0.468 | 0.290 |
| Emphysema          | 0.122  | 0.130  | 0.101  | 0.125 | 0.144 |
| Fibrosis           | 0.012  | 0.000  | 0.000  | 0.004 | 0.071 |
| Hernia             | 0.000  | 0.029  | 0.235  | 0.000 | 0.206 |
| Infiltration       | 0.262  | 0.259  | 0.058  | 0.251 | 0.039 |
| Mass               | 0.143  | 0.163  | 0.000  | 0.189 | 0.010 |
| No Finding         | 0.652  | 0.634  | 0.084  | 0.632 | 0.078 |
| Nodule             | 0.040  | 0.032  | 0.000  | 0.028 | 0.000 |
| Pleural_Thickening | 0.058  | 0.014  | 0.000  | 0.016 | 0.000 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 |
| Pneumothorax       | 0.213  | 0.300  | 0.063  | 0.378 | 0.106 |

#### Test False Positive Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 |
|--------------------|--------|--------|--------|-------|-------|
| Atelectasis        | 0.043  | 0.038  | 0.006  | 0.069 | 0.005 |
| Cardiomegaly       | 0.008  | 0.012  | 0.041  | 0.006 | 0.026 |
| Consolidation      | 0.032  | 0.001  | 0.000  | 0.004 | 0.000 |
| Edema              | 0.006  | 0.009  | 0.037  | 0.003 | 0.027 |
| Effusion           | 0.084  | 0.069  | 0.023  | 0.067 | 0.035 |
| Emphysema          | 0.010  | 0.006  | 0.007  | 0.006 | 0.009 |
| Fibrosis           | 0.002  | 0.000  | 0.000  | 0.004 | 0.013 |
| Hernia             | 0.000  | 0.000  | 0.002  | 0.000 | 0.001 |
| Infiltration       | 0.148  | 0.119  | 0.018  | 0.105 | 0.013 |
| Mass               | 0.026  | 0.016  | 0.000  | 0.015 | 0.001 |
| No Finding         | 0.359  | 0.292  | 0.021  | 0.268 | 0.017 |
| Nodule             | 0.014  | 0.006  | 0.000  | 0.003 | 0.000 |
| Pleural_Thickening | 0.013  | 0.004  | 0.000  | 0.002 | 0.000 |
| Pneumonia          | 0.000  | 0.000  | 0.000  | 0.000 | 0.000 |
| Pneumothorax       | 0.016  | 0.024  | 0.004  | 0.037 | 0.007 |

#### Test False Negative Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 |
|--------------------|--------|--------|--------|-------|-------|
| Atelectasis        | 0.845  | 0.842  | 0.960  | 0.747 | 0.974 |
| Cardiomegaly       | 0.824  | 0.732  | 0.546  | 0.821 | 0.635 |
| Consolidation      | 0.892  | 0.988  | 1.000  | 0.980 | 1.000 |
| Edema              | 0.927  | 0.883  | 0.678  | 0.942 | 0.725 |
| Effusion           | 0.573  | 0.546  | 0.777  | 0.532 | 0.710 |
| Emphysema          | 0.878  | 0.870  | 0.899  | 0.875 | 0.856 |
| Fibrosis           | 0.988  | 1.000  | 1.000  | 0.996 | 0.929 |
| Hernia             | 1.000  | 0.971  | 0.765  | 1.000 | 0.794 |
| Infiltration       | 0.738  | 0.741  | 0.942  | 0.749 | 0.961 |
| Mass               | 0.857  | 0.837  | 1.000  | 0.811 | 0.990 |
| No Finding         | 0.348  | 0.366  | 0.916  | 0.368 | 0.922 |
| Nodule             | 0.960  | 0.968  | 1.000  | 0.972 | 1.000 |
| Pleural_Thickening | 0.942  | 0.986  | 1.000  | 0.984 | 1.000 |
| Pneumonia          | 1.000  | 1.000  | 1.000  | 1.000 | 1.000 |
| Pneumothorax       | 0.787  | 0.700  | 0.937  | 0.622 | 0.894 |

#### Test True Negative Rate Evolution

| Class              | SDV1_1 | SDV1_2 | SDV1_3 | DV1_1 | DV1_2 |
|--------------------|--------|--------|--------|-------|-------|
| Atelectasis        | 0.957  | 0.962  | 0.994  | 0.931 | 0.995 |
| Cardiomegaly       | 0.992  | 0.988  | 0.959  | 0.994 | 0.974 |
| Consolidation      | 0.968  | 0.999  | 1.000  | 0.996 | 1.000 |
| Edema              | 0.994  | 0.991  | 0.963  | 0.997 | 0.973 |
| Effusion           | 0.916  | 0.931  | 0.977  | 0.933 | 0.965 |
| Emphysema          | 0.990  | 0.994  | 0.993  | 0.994 | 0.991 |
| Fibrosis           | 0.998  | 1.000  | 1.000  | 0.999 | 0.987 |
| Hernia             | 1.000  | 1.000  | 0.998  | 1.000 | 0.999 |
| Infiltration       | 0.852  | 0.881  | 0.982  | 0.895 | 0.987 |
| Mass               | 0.974  | 0.984  | 1.000  | 0.985 | 0.999 |
| No Finding         | 0.641  | 0.708  | 0.979  | 0.732 | 0.983 |
| Nodule             | 0.986  | 0.994  | 1.000  | 0.997 | 1.000 |
| Pleural_Thickening | 0.987  | 0.996  | 1.000  | 0.998 | 1.000 |
| Pneumonia          | 1.000  | 1.000  | 1.000  | 1.000 | 1.000 |
| Pneumothorax       | 0.984  | 0.976  | 0.996  | 0.963 | 0.993 |

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

### DenseNetV1_1

#### Configuration
- **Changes from v3:**
  - Full DenseNet121 architecture with bottleneck layers
  - Basic Image Augmentation from previous versions
  - Removed class weights
- **Hypothesis:**
  - Full DenseNet121 architecture with all its layers and features
    - Overall model performance
    - Feature extraction capabilities
- **Impact:**
  - Improved overall metrics compared to SimplifiedDenseNetV1_2

### DenseNetV1_2

#### Configuration
- **Changes from v1:**
  - Added class weights to model training
- **Hypothesis:**
  - Class weights will help address class imbalance 
    issues and improve performance on minority classes
- **Impact:**
  - Better performance on minority classes
  - Most classes showed decreased performance 