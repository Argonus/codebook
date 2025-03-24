# Model Evolution Summary

This document tracks the evolution of DenseNet architectures developed during this research, including performance metrics, architectural changes, and observed behaviors.

## Performance Comparison

### Overall Metrics Comparison
This metrics are collected from train and validation datasets logs.

#### F1 Score

| Model | Best Val F1 | Avg Val F1 | Best Training F1 | Epochs to Best |
|-------|------------|------------|------------------|----------------|
| SimplifiedDenseNet_v1 | 0.0584 | 0.0216 | 0.0465 | 20 |
| SimplifiedDenseNet_v2 | 0.0540 | 0.0095 | 0.0654 | 17 |
| SimplifiedDenseNet_v3 | 0.0574 | 0.0117 | 0.0588 | 10 |
| SimplifiedDenseNet_v4 | 0.0558 | 0.0131 | 0.0670 | 26 |
| SimplifiedDenseNet_v5 | 0.1041 | 0.0361 | 0.1166 | 27 |
| DenseNet121_v1        | 0.1041 | 0.0361 | 0.1166 | 27 |


#### AUC Score

| Model | Best Val AUC | Avg Val AUC | Training AUC | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.7144 | 0.6730 | 0.7149 | 29 |
| SimplifiedDenseNet_v2 | 0.6904 | 0.6418 | 0.6716 | 15 |
| SimplifiedDenseNet_v3 | 0.7127 | 0.6527 | 0.7235 | 28 |
| SimplifiedDenseNet_v4 | 0.7216 | 0.6736 | 0.7445 | 24 |
| SimplifiedDenseNet_v5 | 0.7510 | 0.6988 | 0.7685 | 26 |


#### Precision

| Model | Best Val Precision | Avg Val Precision | Training Precision | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.5000 | 0.2310 | 0.3834 | 10 |
| SimplifiedDenseNet_v2 | 0.2438 | 0.1020 | 0.2249 | 27 |
| SimplifiedDenseNet_v3 | 0.4087 | 0.1745 | 0.4487 | 16 |
| SimplifiedDenseNet_v4 | 0.4456 | 0.2005 | 0.4557 | 30 |
| SimplifiedDenseNet_v5 | 0.6667 | 0.2925 | 0.5004 | 2  |

#### Recall

| Model | Best Val Recall | Avg Val Recall | Training Recall | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.0536 | 0.0171 | 0.0316 | 4  |
| SimplifiedDenseNet_v2 | 0.0907 | 0.0088 | 0.0474 | 17 |
| SimplifiedDenseNet_v3 | 0.1047 | 0.0121 | 0.0424 | 9  |
| SimplifiedDenseNet_v4 | 0.0494 | 0.0098 | 0.0469 | 23 |
| SimplifiedDenseNet_v5 | 0.1043 | 0.0278 | 0.0851 | 27 |

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

#### Loss Metrics Evolution

| Model | Initial Loss | Final Loss | Rate of Convergence | Loss Volatility |
|-------|-------------|------------|---------------------|----------------|
| SimplifiedDenseNet_v1 | ? | ? | ? | ? |
| SimplifiedDenseNet_v2 | ? | ? | ? | ? |
| SimplifiedDenseNet_v3 | ? | ? | ? | ? |
| SimplifiedDenseNet_v4 | ? | ? | ? | ? |

**Loss** metric allows to track how well our model is learning over time. We can see how many mistakes, our model is doing over time. In summary, we can see that our model learned something based on diff between initial and finall loss

#### Class-wise Precision Comparison (Top 5 Classes)

| Class | v1 Precision | v2 Precision | v3 Precision | v4 Precision |
|-------|-------------|-------------|-------------|-------------|
| Cardiomegaly | ? | ? | ? | ? |
| Effusion     | ? | ? | ? | ? |
| Hernia       | ? | ? | ? | ? |
| Pneumothorax | ? | ? | ? | ? |
| Edema        | ? | ? | ? | ? |

#### Class-wise Recall Comparison (Top 5 Classes)

| Class | v1 Recall | v2 Recall | v3 Recall | v4 Recall |
|-------|-----------|-----------|-----------|----------|
| Cardiomegaly | ? | ? | ? | ? |
| Effusion     | ? | ? | ? | ? |
| Hernia       | ? | ? | ? | ? |
| Pneumothorax | ? | ? | ? | ? |
| Edema        | ? | ? | ? | ? |

### Training Dynamics Comparison

#### Learning Convergence Patterns

| Model | Converged | Epochs to Stabilize | Oscillation After Convergence | Final vs. Best Epoch |
|-------|-----------|---------------------|------------------------------|---------------------|
| SimplifiedDenseNet_v1 | ? | ? | ? | ? |
| SimplifiedDenseNet_v2 | ? | ? | ? | ? |
| SimplifiedDenseNet_v3 | ? | ? | ? | ? |
| SimplifiedDenseNet_v4 | ? | ? | ? | ? |

#### True Positive Rate Evolution

| Model | Early TP Rate | Mid TP Rate | Final TP Rate | Pattern |
|-------|---------------|-------------|---------------|--------|
| SimplifiedDenseNet_v1 | ? | ? | ? | ? |
| SimplifiedDenseNet_v2 | ? | ? | ? | ? |
| SimplifiedDenseNet_v3 | ? | ? | ? | ? |
| SimplifiedDenseNet_v4 | ? | ? | ? | ? |

#### Loss Contribution by Class (Average Across Epochs)

| Class | v1 Loss Contribution | v2 Loss Contribution | v3 Loss Contribution | v4 Loss Contribution |
|-------|----------------------|----------------------|----------------------|----------------------|
| Infiltration | ? | ? | ? | ? |
| Effusion     | ? | ? | ? | ? |
| Cardiomegaly | ? | ? | ? | ? |
| Hernia       | ? | ? | ? | ? |
| No Finding   | ? | ? | ? | ? |

## Class-wise Performance

| Model | Most Improved Classes | Problematic Classes | Class Balance Impact |
|-------|----------------------|---------------------|----------------------|
| SimplifiedDenseNet_v1 | Cardiomegaly (AUC=0.88, F1=0.34, wt=2.5) <br/> Effusion (AUC=0.80, F1=0.25, wt=0.5) <br/> Edema (AUC=0.84, F1=0.18, wt=3.0) | Pneumothorax (AUC=0.74, F1=0.03, wt=1.3) <br/> Consolidation (AUC=0.73, F1=0.00, wt=1.5) <br/> Atelectasis (AUC=0.72, F1=0.01, wt=0.6) | ? |
| SimplifiedDenseNet_v2 | Infiltration (AUC=0.51, F1=0.27, wt=0.3) <br/> Consolidation (AUC=0.53, F1=0.08, wt=1.5) <br/> Mass (AUC=0.51, F1=0.08, wt=1.2) | None identified | ? |
| SimplifiedDenseNet_v3 | Effusion (AUC=0.78, F1=0.35, wt=0.5) </br> Cardiomegaly (AUC=0.83, F1=0.20, wt=2.5) <br/> Atelectasis (AUC=0.69, F1=0.02, wt=0.6) | Edema (AUC=0.83, F1=0.00, wt=3.0) <br/> Consolidation (AUC=0.71, F1=0.00, wt=1.5) <br/> No Finding (AUC=0.70, F1=0.00, wt=0.1) | ? |
| SimplifiedDenseNet_v4 | Cardiomegaly (AUC=0.85, F1=0.22, wt=2.5) <br/> Hernia (AUC=0.73, F1=0.11, wt=30.2) <br/> Pneumothorax (AUC=0.75, F1=0.01, wt=1.3) | Edema (AUC=0.84, F1=0.00, wt=3.0) <br/> Effusion (AUC=0.79, F1=0.01, wt=0.5) <br/> Pneumothorax (AUC=0.75, F1=0.01, wt=1.3) | ? |
| SimplifiedDenseNet_v5 | Effusion (AUC=0.79, F1=0.33, wt=0.5) <br/> Infiltration (AUC=0.61, F1=0.32, wt=0.3) <br/> Cardiomegaly (AUC=0.82, F1=0.28, wt=2.5) | Consolidation (AUC=0.73, F1=0.00, wt=1.5) <br/> Pleural_Thickening (AUC=0.72, F1=0.00, wt=2.0) <br/> No Finding (AUC=0.71, F1=0.00, wt=0.1) | ? |

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
- **Hypothesis:** [Why these changes were made]
- **Impact:** [Results](models/Simplified_DenseNet_v5)

### DenseNet121_v1
- **Changes from v5:**
  - Add bottleneck layer
- **Hypothesis:**
  - 
- **Impact:** [Results](models/Simplified_DenseNet_v5)

### DenseNet121_v2
- **Changes from v1:**
  - Added SE block with ration 16
- **Hypothesis:**
  - 
- **Impact:** [Results](models/DenseNet121_v2)

### DenseNet121_v3
- **Changes from v2:**
  - Lowered ratio to 8
- **Hypothesis:**
  - 
- **Impact:** [Results](models/DenseNet121_v2)

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