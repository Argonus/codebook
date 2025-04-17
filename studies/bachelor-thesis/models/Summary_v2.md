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
|-------|-------------|------------|------------------|----------------|

#### AUC ROC Score

AUC ROC is a metrics that shows how well the model is able to separate positive and negative cases. This is a secondary metric.

| Model | Best Val AUC | Avg Val AUC | Training AUC | Epochs to Best |
|-------|--------------|-------------|--------------|----------------|

#### Precision

Precision is a metrics that shows how many of the predicted positive cases are actually positive. This is a secondary metric that can be calculated using following algorithm

```latex
Precision = True Positives / (True Positives + False Positives)
```

#### Recall

Recall is a metrics that shows how many of the actual positive cases are predicted as positive. This is a secondary metric that can be calculated using following algorithm

```latex
Recall = True Positives / (True Positives + False Negatives)
```

### Detailed Class Performance Comparison
This metrics are collected from test dataset logs.

#### F1 Score per Class

| Class | <Model> |
|-------|---------|

#### AUC ROC per Class

| Class | <Model> |


#### Precision per Class

| Class | <Model> |
|-------|---------|

#### Recall per Class

| Class | <Model> |
|-------|---------|

### Loss Metrics Evolution

**Loss** metric allows to track how well our model is learning over time. We can see how many mistakes, our model is doing over time. In summary, we can see that our model learned something based on diff between initial and final loss. And we aim to have a low initial loss and a low final loss.

- **Initial Loss** is a metric that shows the initial loss of the model. In summary it shows how well our model was learning at the beginning. 
- **Final Loss** is a metric that shows the final loss of the model. In summary its a loss at last epoch. In summary it shows how well our model was learning at the end.
- **Rate of Convergence** is a metric that shows the rate of convergence of the model. Convergence shows us how quickly model moves to end state. Bigger value, in theory means model is learning faster.
- **Loss Volatility** is a metric that shows the volatility of the loss of the model. Volatility is a how much the loss changes over time. Lower value means more stable learning process, bigger value means more unstable learning process.

| Model | Initial Loss | Final Loss | Rate of Convergence | Loss Volatility |
|-------|--------------|------------|---------------------|-----------------|

### Training Dynamics Comparison

#### Learning Convergence Patterns

This section shows the learning convergence patterns of the models. It shows how well the models have converged and how they have stabilized.

- **Converged** is a metric that shows if the model has converged. In summary it shows if the model has reached a stable state.
- **Epochs to Stabilize** is a metric that shows the number of epochs it took for the model to stabilize. In summary it shows how many epochs it took for the model to reach a stable state.
- **Oscillation After Convergence** is a metric that shows the oscillation of the model after convergence. In summary it shows how much the model oscillates after it has reached a stable state.
- **Final vs. Best Epoch** is a metric that shows the final vs. best epoch of the model. In summary it shows how much the model has improved over time.

| Model | Converged | Epochs to Stabilize | Oscillation After Convergence | Final vs. Best Epoch |
|-------|-----------|---------------------|-------------------------------|----------------------|

### True Positive Rate Evolution

#### Training True Positive Rate Evolution

| Model | Early TP Count | Mid TP Count | Final TP Count |
|-------|----------------|--------------|----------------|

#### Validation True Positive Rate Evolution

| Class | <Model> |
|-------|---------|

## Architectural Changes and Impacts

### SimplifiedDenseNetV1_1 (Baseline)

#### Configuration
- Epochs: 
- Batch size: 
- Optimizer:
- Loss: 
- Image Augmentation: 
- Architecture: 
- Learning rate schedule: 

### SimplifiedDenseNetV1_2

#### Configuration
- **Changes from v1:**
- **Hypothesis:**
- **Impact:**