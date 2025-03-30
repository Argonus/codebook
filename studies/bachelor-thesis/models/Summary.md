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

| Model | Best Val F1 | Avg Val F1 | Best Training F1 | Epochs to Best |
|-------|------------|------------|------------------|----------------|
| SimplifiedDenseNet_v1 | 0.0584 | 0.0216 | 0.0465 | 20 |
| SimplifiedDenseNet_v2 | 0.0540 | 0.0095 | 0.0654 | 17 |
| SimplifiedDenseNet_v3 | 0.0574 | 0.0117 | 0.0588 | 10 |
| SimplifiedDenseNet_v4 | 0.0558 | 0.0131 | 0.0670 | 26 |
| SimplifiedDenseNet_v5 | 0.1041 | 0.0361 | 0.1166 | 27 |
| DenseNet121_v1        | 0.1341 | 0.0510 | 0.1582 | 26 |
| DenseNet121_v2        | 0.1642 | 0.0526 | 0.1519 | 30 |
| DenseNet121_v2_1      | 0.1285 | 0.0449 | 0.1238 | 11 |
| DenseNet121_v3        | 0.3554 | 0.2452 | 0.3038 | 27 |
| DenseNet121_v3_1      | 0.3650 | 0.2692 | 0.3393 | 22 |
| DenseNet121_v3_2      | 0.3958 | 0.3062 | 0.4441 | 28 |


#### AUC ROC Score

AUC ROC is a metrics that shows how well the model is able to separate positive and negative cases. This is a secondary metric.

| Model | Best Val AUC | Avg Val AUC | Training AUC | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.7144 | 0.6730 | 0.7149 | 29 |
| SimplifiedDenseNet_v2 | 0.6904 | 0.6418 | 0.6716 | 15 |
| SimplifiedDenseNet_v3 | 0.7127 | 0.6527 | 0.7235 | 28 |
| SimplifiedDenseNet_v4 | 0.7216 | 0.6736 | 0.7445 | 24 |
| SimplifiedDenseNet_v5 | 0.7510 | 0.6988 | 0.7685 | 26 |
| DenseNet121_v1        | 0.7552 | 0.7131 | 0.7962 | 26 |
| DenseNet121_v2        | 0.7590 | 0.7170 | 0.7987 | 26 |
| DenseNet121_v2_1      | 0.7403 | 0.7049 | 0.7788 | 24 |
| DenseNet121_v3        | 0.7798 | 0.7496 | 0.8228 | 23 |
| DenseNet121_v3_1      | 0.7825 | 0.7608 | 0.8313 | 19 |
| DenseNet121_v3_2      | 0.7831 | 0.7621 | 0.8629 | 11 |

#### Precision

Precision is a metrics that shows how many of the predicted positive cases are actually positive. This is a secondary metric that can be calculated using following algorithm

```latex
Precision = True Positives / (True Positives + False Positives)
```

| Model | Best Val Precision | Avg Val Precision | Training Precision | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.5000 | 0.2310 | 0.3834 | 10 |
| SimplifiedDenseNet_v2 | 0.2438 | 0.1020 | 0.2249 | 27 |
| SimplifiedDenseNet_v3 | 0.4087 | 0.1745 | 0.4487 | 16 |
| SimplifiedDenseNet_v4 | 0.4456 | 0.2005 | 0.4557 | 30 |
| SimplifiedDenseNet_v5 | 0.6667 | 0.2925 | 0.5004 | 2  |
| DenseNet121_v1        | 0.5537 | 0.2777 | 0.5266 | 12 |
| DenseNet121_v2        | 0.5214 | 0.3083 | 0.5300 | 21 |
| DenseNet121_v2_1      | 0.4936 | 0.3015 | 0.5099 | 28 |
| DenseNet121_v3        | 1.0000 | 0.7124 | 0.7269 | 1  |
| DenseNet121_v3_1      | 0.8141 | 0.6489 | 0.7003 | 2  |
| DenseNet121_v3_2      | 0.7304 | 0.6276 | 0.7127 | 1  |

#### Recall

Recall is a metrics that shows how many of the actual positive cases are predicted as positive. This is a secondary metric that can be calculated using following algorithm

```latex
Recall = True Positives / (True Positives + False Negatives)
```

| Model | Best Val Recall | Avg Val Recall | Training Recall | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.0536 | 0.0171 | 0.0316 | 4  |
| SimplifiedDenseNet_v2 | 0.0907 | 0.0088 | 0.0474 | 17 |
| SimplifiedDenseNet_v3 | 0.1047 | 0.0121 | 0.0424 | 9  |
| SimplifiedDenseNet_v4 | 0.0494 | 0.0098 | 0.0469 | 23 |
| SimplifiedDenseNet_v5 | 0.1043 | 0.0278 | 0.0851 | 27 |
| DenseNet121_v1        | 0.0960 | 0.0392 | 0.1179 | 26 |
| DenseNet121_v2        | 0.1177 | 0.0399 | 0.1132 | 30 |
| DenseNet121_v2_1      | 0.0810 | 0.0323 | 0.0911 | 30 |
| DenseNet121_v3        | 0.3314 | 0.1871 | 0.2295 | 8  |
| DenseNet121_v3_1      | 0.3139 | 0.2109 | 0.2603 | 13 |
| DenseNet121_v3_2      | 0.3436 | 0.2409 | 0.3580 | 23 |

### Detailed Class Performance Comparison
This metrics are collected from test dataset logs.

#### Top Performing Classes by Model (F1 Score)

| Model | 1st Best Class | 2nd Best Class | 3rd Best Class |
|-------|---------------|---------------|---------------|
| SimplifiedDenseNet_v1 | Cardiomegaly (F1=0.34) | Effusion (F1=0.25)      | Edema (F1=0.18)        |
| SimplifiedDenseNet_v2 | Infiltration (F1=0.27) | Consolidation (F1=0.08) | Mass (F1=0.08)         |
| SimplifiedDenseNet_v3 | Effusion (F1=0.35)     | Cardiomegaly (F1=0.20)  | Atelectasis (F1=0.02)  |
| SimplifiedDenseNet_v4 | Cardiomegaly (F1=0.22) | Hernia (F1=0.11)        | Pneumothorax (F1=0.01) |
| SimplifiedDenseNet_v5 | Effusion (F1=0.33)     | Infiltration (F1=0.32)  | Cardiomegaly (F1=0.28) |
| DenseNet121_v1        | Effusion (F1=0.41)     | Cardiomegaly (F1=0.32)  | EdHerniaema (F1=0.30)  |
| DenseNet121_v2        | Effusion (F1=0.33)     | No Finding (F1=0.28)    | Hernia (F1=0.25)       |
| DenseNet121_v2_1      | Cardiomegaly (F1=0.30) | Emphysema (F1=0.22)     | Effusion (F1=0.20)     |
| DenseNet121_v3        | No Finding (F1=0.50)   | Effusion (F1=0.43)      | Cardiomegaly (F1=0.35) |
| DenseNet121_v3_1      | No Finding (F1=0.63)   | Effusion (F1=0.37)      | Cardiomegaly (F1=0.32) |
| DenseNet121_v3_2      | No Finding (F1=0.64)   | Effusion (F1=0.45)      | Pneumothorax (F1=0.32) |

#### Loss Metrics Evolution

**Loss** metric allows to track how well our model is learning over time. We can see how many mistakes, our model is doing over time. In summary, we can see that our model learned something based on diff between initial and final loss. And we aim to have a low initial loss and a low final loss.

- **Initial Loss** is a metric that shows the initial loss of the model. In summary it shows how well our model was learning at the beginning. 
- **Final Loss** is a metric that shows the final loss of the model. In summary its a loss at last epoch. In summary it shows how well our model was learning at the end.
- **Rate of Convergence** is a metric that shows the rate of convergence of the model. Convergence shows us how quickly model moves to end state. Bigger value, in theory means model is learning faster.
- **Loss Volatility** is a metric that shows the volatility of the loss of the model. Volatility is a how much the loss changes over time. Lower value means more stable learning process, bigger value means more unstable learning process.

| Model | Initial Loss | Final Loss | Rate of Convergence | Loss Volatility |
|-------|-------------|------------|---------------------|----------------|
| Simplified_DenseNet_v1 | 0.7445 | 0.1967 | 0.0189 | 0.1114 |
| Simplified_DenseNet_v2 | 0.5757 | 0.0530 | 0.0180 | 0.1034 |
| Simplified_DenseNet_v3 | 1.4972 | 0.0495 | 0.0499 | 0.2637 |
| Simplified_DenseNet_v4 | 0.5338 | 0.0533 | 0.0166 | 0.0907 |
| Simplified_DenseNet_v5 | 0.5338 | 0.0533 | 0.0166 | 0.0907 |
| DenseNet121_v1         | 1.8001 | 0.1750 | 0.0560 | 0.2990 |
| DenseNet121_v2         | 1.6527 | 0.1819 | 0.0507 | 0.2689 |
| DenseNet121_v2_1       | 1.6972 | 0.1804 | 0.0523 | 0.2777 |
| DenseNet121_v3         | 1.5953 | 0.2051 | 0.0479 | 0.2552 |
| DenseNet121_v3_1       | 1.5264 | 0.1748 | 0.0466 | 0.2456 |
| DenseNet121_v3_2       | 1.5107 | 0.2203 | 0.0445 | 0.2348 |

### Training Dynamics Comparison

#### Learning Convergence Patterns

This section shows the learning convergence patterns of the models. It shows how well the models have converged and how they have stabilized.

- **Converged** is a metric that shows if the model has converged. In summary it shows if the model has reached a stable state.
- **Epochs to Stabilize** is a metric that shows the number of epochs it took for the model to stabilize. In summary it shows how many epochs it took for the model to reach a stable state.
- **Oscillation After Convergence** is a metric that shows the oscillation of the model after convergence. In summary it shows how much the model oscillates after it has reached a stable state.
- **Final vs. Best Epoch** is a metric that shows the final vs. best epoch of the model. In summary it shows how much the model has improved over time.

| Model | Converged | Epochs to Stabilize | Oscillation After Convergence | Final vs. Best Epoch |
|-------|-----------|---------------------|------------------------------|---------------------|
| Simplified_DenseNet_v1 | No  | 30 | Insufficient Data | 75.3% (Significant Drop)                 |
| Simplified_DenseNet_v2 | No  | 30 | Insufficient Data | 26.9% (Significant Drop)                 |
| Simplified_DenseNet_v3 | No  | 30 | Insufficient Data | 41.4% (Significant Drop)                 |
| Simplified_DenseNet_v4 | No  | 30 | Insufficient Data | 12.0% (Significant Drop (Best Near End)) |
| Simplified_DenseNet_v5 | Yes | 13 | High              | 70.6% (Significant Drop (Best Near End)) |
| DenseNet121_v1         | Yes | 12 | High              | 75.9% (Significant Drop (Best Near End)) |
| DenseNet121_v2         | No  | 12 | High              | 100.0% (Close to Best (Best at Final))   |
| DenseNet121_v2_1       | Yes | 30 | Insufficient Data | 78.2% (Significant Drop (Best Near End)) |
| DenseNet121_v3         | No  | 7  | High              | 46.1% (Significant Drop (Best Near End)) |
| DenseNet121_v3_1       | Yes | 9  | Medium            | 98.2% (Close to Best)                    |
| DenseNet121_v3_2       | Yes | 7  | Medium            | 93.5% (Moderate Drop (Best Near End))    |


#### True Positive Rate Evolution

| Model | Early TP Count | Mid TP Count | Final TP Count |
|-------|---------------|-------------|---------------|
| Simplified_DenseNet_v1 | 156.9 | 513.3 | 338.4  |
| Simplified_DenseNet_v2 | 30.2  | 385.3 | 102.6  |
| Simplified_DenseNet_v3 | 393.8 | 45.2  | 258.6  |
| Simplified_DenseNet_v4 | 29.4  | 152.9 | 398.0  |
| Simplified_DenseNet_v5 | 101.3 | 393.7 | 1084.5 |
| DenseNet121_v1         | 193.3 | 734.5 | 1263.3 |
| DenseNet121_v2         | 246.1 | 710.1 | 1070.1 |
| DenseNet121_v2_1       | 93.2  | 487.3 | 1092.1 |
| DenseNet121_v3         | 248.6 | 754.0 | 1120.0 |
| DenseNet121_v3_1       | 564.7 | 1299.5| 1559.3 |
| DenseNet121_v3_2       | 828.1 | 1526.8| 2168.2 |

## Class-wise Performance

| Model | Most Improved Classes | Problematic Classes | Class Balance Impact |
|-------|----------------------|---------------------|----------------------|
| SimplifiedDenseNet_v1 | Cardiomegaly (AUC=0.88, F1=0.34, wt=2.5) <br/> Effusion (AUC=0.80, F1=0.25, wt=0.5) <br/> Edema (AUC=0.84, F1=0.18, wt=3.0) | Pneumothorax (AUC=0.74, F1=0.03, wt=1.3) <br/> Consolidation (AUC=0.73, F1=0.00, wt=1.5) <br/> Atelectasis (AUC=0.72, F1=0.01, wt=0.6) | ? |
| SimplifiedDenseNet_v2 | Infiltration (AUC=0.51, F1=0.27, wt=0.3) <br/> Consolidation (AUC=0.53, F1=0.08, wt=1.5) <br/> Mass (AUC=0.51, F1=0.08, wt=1.2) | None identified | ? |
| SimplifiedDenseNet_v3 | Effusion (AUC=0.78, F1=0.35, wt=0.5) </br> Cardiomegaly (AUC=0.83, F1=0.20, wt=2.5) <br/> Atelectasis (AUC=0.69, F1=0.02, wt=0.6) | Edema (AUC=0.83, F1=0.00, wt=3.0) <br/> Consolidation (AUC=0.71, F1=0.00, wt=1.5) <br/> No Finding (AUC=0.70, F1=0.00, wt=0.1) | ? |
| SimplifiedDenseNet_v4 | Cardiomegaly (AUC=0.85, F1=0.22, wt=2.5) <br/> Hernia (AUC=0.73, F1=0.11, wt=30.2) <br/> Pneumothorax (AUC=0.75, F1=0.01, wt=1.3) | Edema (AUC=0.84, F1=0.00, wt=3.0) <br/> Effusion (AUC=0.79, F1=0.01, wt=0.5) <br/> Pneumothorax (AUC=0.75, F1=0.01, wt=1.3) | ? |
| SimplifiedDenseNet_v5 | Effusion (AUC=0.79, F1=0.33, wt=0.5) <br/> Infiltration (AUC=0.61, F1=0.32, wt=0.3) <br/> Cardiomegaly (AUC=0.82, F1=0.28, wt=2.5) | Consolidation (AUC=0.73, F1=0.00, wt=1.5) <br/> Pleural_Thickening (AUC=0.72, F1=0.00, wt=2.0) <br/> No Finding (AUC=0.71, F1=0.00, wt=0.1) | ? |
| DenseNet121_v1 | Effusion (AUC=0.84, F1=0.41, wt=0.5) <br/> Cardiomegaly (AUC=0.86, F1=0.32, wt=2.5) <br/> Hernia (AUC=0.87, F1=0.30, wt=30.2) |	Consolidation (AUC=0.75, F1=0.00, wt=1.5) <br/> Pleural_Thickening (AUC=0.74, F1=0.01, wt=2.0) <br/> No Finding (AUC=0.74, F1=0.03, wt=0.1)	| ? |
| DenseNet121_v2 | Effusion (AUC=0.82, F1=0.33, wt=0.5) <br/> No Finding (AUC=0.71, F1=0.28, wt=0.1) <br/> Hernia (AUC=0.79, F1=0.25, wt=30.2) |	Consolidation (AUC=0.73, F1=0.00, wt=1.5) <br/> Mass (AUC=0.73, F1=0.02, wt=1.2) <br/> Pleural_Thickening (AUC=0.73, F1=0.00, wt=2.0)	| ? |
| DenseNet121_v2_1 | Cardiomegaly (AUC=0.87, F1=0.30, wt=2.5) <br/> Emphysema (AUC=0.77, F1=0.22, wt=2.7) <br/> Effusion (AUC=0.81, F1=0.20, wt=0.5) | Hernia (AUC=0.80, F1=0.00, wt=30.2) <br/> Consolidation (AUC=0.75, F1=0.00, wt=1.5) <br/> Atelectasis (AUC=0.73, F1=0.01, wt=0.6) | ? |
| DenseNet121_v3	| No Finding (AUC=0.76, F1=0.50, wt=0.1) <br/> Effusion (AUC=0.86, F1=0.43, wt=0.5) <br/> Cardiomegaly (AUC=0.90, F1=0.35, wt=2.5) |	Edema (AUC=0.86, F1=0.00, wt=3.0) <br/> Emphysema (AUC=0.81, F1=0.01, wt=2.7) <br/> Consolidation (AUC=0.77, F1=0.00, wt=1.5) | ? |
| DenseNet121_v3_1	| No Finding (AUC=0.76, F1=0.63, wt=0.1) <br/> Effusion (AUC=0.86, F1=0.37, wt=0.5) <br/> Cardiomegaly (AUC=0.88, F1=0.32, wt=2.5) |	Edema (AUC=0.86, F1=0.02, wt=3.0) <br/> Consolidation (AUC=0.77, F1=0.00, wt=1.5) <br/> Pleural_Thickening (AUC=0.74, F1=0.00, wt=2.0) | ? |
| DenseNet121_v3_2	| No Finding (AUC=0.76, F1=0.64, wt=0.1) <br/> Effusion (AUC=0.86, F1=0.45, wt=0.5) <br/> Pneumothorax (AUC=0.83, F1=0.32, wt=1.3) <br/> Edema (AUC=0.87, F1=0.01, wt=3.0) <br/> Consolidation (AUC=0.76, F1=0.00, wt=1.5) <br/> Pleural_Thickening (AUC=0.74, F1=0.03, wt=2.0)  | ? |

## Architectural Changes and Impacts

### SimplifiedDenseNet_v1 (Baseline)

- **Configuration:**
  - Epochs: 30
  - Batch size: 32
  - Optimizer: Adam(learning_rate=0.0001)
  - Loss: BinaryCrossentropy(from_logits=False)
  - Image Augmentation: brightness (0.5), contrast (0.5), shifting (0.1), gaussian_noise (0.05)
  - Architecture: DenseNet without bottleneck layers, L2 regularization (1e-4)
  - Learning rate schedule: ReduceLROnPlateau(factor=0.5, patience=5)
- **Performance Characteristics:**
  - Good AUC score (0.7144) indicating reasonable discrimination ability
  - Low F1 score (max 0.0584)
  - Reasonable precision (0.5000) but very poor recall (0.0536)
    - 0.5000 is suspiciously high
  - Model showed inconsistent F1 performance with spikes and drops rather than steady improvement
  - Best validation metrics typically achieved in later epochs (20-29)

### SimplifiedDenseNet_v2
- **Changes from v1:**
  - Loss: BinaryFocalCrossentropy (gamma=2.0, alpha=0.25, label_smoothing=0.01)
- **Hypothesis:** 
  - The focal loss was implemented to address class imbalance by reducing the loss contribution from easy examples and focusing more on difficult, misclassified examples. The alpha parameter gives higher weight to the positive class while label smoothing was added to prevent overconfidence.
- **Impact:**
  - AUC decreased slightly (0.6904 vs 0.7144)
  - F1 score remained poor at 0.0540, showing no improvement
  - Precision dropped significantly (0.2438 vs 0.5000) but recall improved (0.0907 vs 0.0536)
  - The focal loss shifted the balance toward better recall at the expense of precision
  - Despite the theoretical advantages for class imbalance, this change didn't improve overall F1 performance compared to standard BCE

### SimplifiedDenseNet_v3
- **Changes from v2:**
  - Added oversampling for rare classes
- **Hypothesis:** 
  - Oversampling was introduced to directly address the class imbalance problem by ensuring rare conditions appear more frequently during training. This should help the model learn better representations for minority classes without relying solely on loss function modifications.
- **Impact:** 
  - AUC improved from v2 (0.7127 vs 0.6904), approaching the baseline level
  - F1 score showed slight improvement (0.0574 vs 0.0540) but remained low
  - Precision recovered significantly (0.4087 vs 0.2438), closer to baseline levels
  - Recall continued to improve (0.1047 vs 0.0907), showing the best recall among models so far
  - Oversampling combined with focal loss maintained the recall advantage while partially recovering the precision lost in v2

### SimplifiedDenseNet_v4
- **Changes from v3:**
  - Added rotation augmentation (0.1)
  - Increased gaussian_noise chance (0.1)
  - Optimize with clipnorm=1.0
  - Reduced LR patience from 5 to 3
- **Hypothesis:** 
  - Enhanced data augmentation (rotation + more noise) was implemented to improve generalization by exposing the model to more varied presentations of medical conditions
  - Gradient clipping (clipnorm=1.0) was added to stabilize training by preventing large gradient updates, particularly important when dealing with noisy medical image data
  - Reduced learning rate patience aimed to make the optimizer more responsive to plateaus, potentially avoiding local minima
- **Impact:** 
  - Best AUC improved to 0.7216, the highest among all models
  - F1 score slightly decreased (0.0558 vs 0.0574)
  - Precision improved further (0.4456 vs 0.4087)
  - Recall declined significantly (0.0494 vs 0.1047), reverting to levels below v2
  - The changes improved model's discrimination ability (AUC) and precision but at the cost of recall, which is critical for medical applications

### SimplifiedDenseNet_v5
- **Changes from v4:**
  - Get back to BinaryCrossentropy(from_logits=False, label_smoothing=0.01)
    - Added label smoothing
- **Hypothesis:** 
  - Best results were achieved with BinaryCrossentropy, indicating that the model is sensitive to the choice of loss function
  - Label smoothing had positive impact on model performance for focal loss function
- **Impact:**
  - Significant improvement in F1 score (0.1041 vs 0.0558), representing an 86% increase
  - Best AUC score among all simplified models (0.7510)
  - Balanced improvement in both precision (0.6667) and recall (0.1043)
  - Better performance across multiple classes with three classes exceeding F1=0.25
  - Combined with oversampling from v3, this approach provided the best overall balance of metrics

### DenseNet121_v1
- **Changes from v5:**
  - Add bottleneck layer
- **Hypothesis:**
  - Standard DenseNet121 architecture with bottleneck layers can capture more complex patterns compared to the simplified version
  - As Bottleneck layers reduce feature dimensionality before processing it should improve model feature extraction, which should lead to better performance for rare classes
- **Impact:**
  - Substantial improvement in F1 score (0.1341 vs 0.1041), nearly 30% increase from SimplifiedDenseNet_v5
  - Further improvement in AUC to 0.7552, continuing the upward trend
  - Performance gains validate the architectural benefits of bottleneck layers. We will stay with that one

### DenseNet121_v2
- **Changes from v1:**
  - Added SE block with ration 16
- **Hypothesis:**
  - Squeeze-and-Excitation (SE) blocks enable dynamic channel-wise feature extraction, which can help the model focus on relevant features and improve performance
  - Ratio of 16 is a common choice for SE blocks, providing a balance between compression and recalibration
- **Impact:**
  - Further F1 score improvement to 0.1642, showing SE blocks' effectiveness
  - Big improvement on F1 score for No Finding Class
  - Improvement on recall

### DenseNet121_v2_1
- **Changes from v2:**
  - Lowered ratio to 8
- **Hypothesis:**
  - Reducing the SE ratio from 16 to 8 increases the to provide more parameters in next layers
- **Impact:** [Results](models/DenseNet121_v2)
  - Decreased F1 score compared to v2 (0.1285 vs 0.1642), suggesting the lower ratio was not beneficial
  - Lower AUC score (0.7403 vs 0.7590) indicating reduced discriminative ability
  - Drop in recall (0.0810 vs 0.1177) indicating poorer detection of positive cases

### DenseNet121_v3
- **Changes from v2:**
  - Change loss to custom No Finding BinaryCrossEntropy 
- **Hypothesis:**
  - The custom loss function enforces logical relationships between "No Finding" class and other pathologies. When no finding is present, the model should not predict any pathologies
- **Impact:** [Results](models/DenseNet121_v2)
  - F1 score improvement to 0.3554, over 115% increase from DenseNet121_v2_1
  - Highest AUC score among all models (0.7798)

### DenseNet121_v3_1
- **Changes from v3:**
  - Reduce lambda to 0.01
- **Hypothesis:**
  - Reducing the lambda parameter will apply smaller penatlies for predicting NoFinding with other classes
  which mach allow model to focus more on predicting disases
- **Impact:**
  - Further improvement in F1 score to 0.3650
  - More balanced performance across classes as well as more stable performance

### DenseNet121_v3_2
- **Changes from v3_1:**
  - Add sigmoid calculation to custom loss function
- **Hypothesis:**
  - Applying a sigmoid calculation within the custom loss function stablize training process
- **Impact:** [Results](models/DenseNet121_v3_1)
  - Further improvement in F1 score.
  - Best untill now number of true positive predictions
  - More balanced performance across classes

## Training Dynamics Analysis

### Learning Rate Impact
[Analysis of how learning rate changes affected convergence]

### Regularization Effects
[Analysis of how different regularization strategies affected model performance]

### Data Augmentation Effectiveness
[Analysis of which augmentations proved most valuable]

## Future Directions

### Promising Avenues
- [Suggested future changes based on observed patterns]
- [Architectural modifications to consider]
- [Training strategy adjustments]

### Potential Pitfalls
- [Issues to be aware of based on previous iterations]
- [Combinations that might be problematic]