# Model Evolution Summary

This document tracks the evolution of DenseNet architectures developed during this research, including performance metrics, architectural changes, and observed behaviors.

## Performance Comparison

### F1 Score

| Model | Best Val F1 | Avg Val F1 | Best Training F1 | Epochs to Best |
|-------|------------|------------|------------------|----------------|
| SimplifiedDenseNet_v1 | 0.0584 | 0.0216 | 0.0465 | 20 |
| SimplifiedDenseNet_v2 | 0.0540 | 0.0095 | 0.0654 | 17 |
| SimplifiedDenseNet_v3 | 0.0574 | 0.0117 | 0.0588 | 10 |
| SimplifiedDenseNet_v4 | 0.0558 | 0.0131 | 0.0670 | 26 |

### AUC Score

| Model | Best Val AUC | Avg Val AUC | Training AUC | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.7144 | 0.6730 | 0.7149 | 29 |
| SimplifiedDenseNet_v2 | 0.6904 | 0.6418 | 0.6716 | 15 |
| SimplifiedDenseNet_v3 | 0.7127 | 0.6527 | 0.7235 | 28 |
| SimplifiedDenseNet_v4 | 0.7216 | 0.6736 | 0.7445 | 24 |

### Precision

| Model | Best Val Precision | Avg Val Precision | Training Precision | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.5000 | 0.2310 | 0.3834 | 10 |
| SimplifiedDenseNet_v2 | 0.2438 | 0.1020 | 0.2249 | 27 |
| SimplifiedDenseNet_v3 | 0.4087 | 0.1745 | 0.4487 | 16 |
| SimplifiedDenseNet_v4 | 0.4456 | 0.2005 | 0.4557 | 30 |

### Recall

| Model | Best Val Recall | Avg Val Recall | Training Recall | Epochs to Best |
|-------|------------|------------|-------------|----------------|
| SimplifiedDenseNet_v1 | 0.0536 | 0.0171 | 0.0316 | 4 |
| SimplifiedDenseNet_v2 | 0.0907 | 0.0088 | 0.0474 | 17 |
| SimplifiedDenseNet_v3 | 0.1047 | 0.0121 | 0.0424 | 9 |
| SimplifiedDenseNet_v4 | 0.0494 | 0.0098 | 0.0469 | 23 |

## Class-wise Performance

| Model | Most Improved Classes | Problematic Classes | Class Balance Impact |
|-------|----------------------|---------------------|----------------------|
| SimplifiedDenseNet_v1 | ? | ? | ? |
| SimplifiedDenseNet_v2 | ? | ? | ? |
| SimplifiedDenseNet_v3 | ? | ? | ? |
| SimplifiedDenseNet_v4 | ? | ? | ? |

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
  - Gradient clipping may have prevented the model from making the larger updates needed to improve recall on rare classes

### SimplifiedDenseNet_v5
- **Changes from v4:**
  - Get back to BinaryCrossentropy
- **Hypothesis:** [Why these changes were made]
- **Impact:** [Results](models/Simplified_DenseNet_v5)

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