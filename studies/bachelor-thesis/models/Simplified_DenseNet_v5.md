# Simplified DenseNet v3

## Description

A simplified version of DenseNet model, which does not include bottleneck layers. Instead architecture
looks like:

| Component | Description | Specifications | Sub-components |
|-----------|-------------|----------------|----------------|
| **Input Layer** | Entry point for images | Shape: (224, 224, 3) | - |
| **Initial Convolution** | Initial feature extraction | - Filters: 64<br>- Kernel: 7x7<br>- Stride: 2<br>- Padding: same | - Conv2D<br>- BatchNorm<br>- ReLU |
| **Initial Pooling** | Downsampling | - Max Pooling<br>- Pool size: 3x3<br>- Stride: 2<br>- Padding: same | - |
| **Dense Block 1** | Feature concatenation block | - Layers: 6<br>- Growth rate: 32 | Each layer has:<br>- Conv2D (3x3)<br>- BatchNorm<br>- ReLU<br>- Concatenation |
| **Transition Layer 1** | Dimensionality reduction | - Compression: 0.5<br>- 1x1 conv<br>- Avg pooling (2x2) | - Conv2D<br>- BatchNorm<br>- ReLU<br>- AvgPool |
| **Dense Block 2** | Feature concatenation block | - Layers: 12<br>- Growth rate: 32 | Each layer has:<br>- Conv2D (3x3)<br>- BatchNorm<br>- ReLU<br>- Concatenation |
| **Transition Layer 2** | Dimensionality reduction | - Compression: 0.5<br>- 1x1 conv<br>- Avg pooling (2x2) | - Conv2D<br>- BatchNorm<br>- ReLU<br>- AvgPool |
| **Dense Block 3** | Feature concatenation block | - Layers: 24<br>- Growth rate: 32 | Each layer has:<br>- Conv2D (3x3)<br>- BatchNorm<br>- ReLU<br>- Concatenation |
| **Transition Layer 3** | Dimensionality reduction | - Compression: 0.5<br>- 1x1 conv<br>- Avg pooling (2x2) | - Conv2D<br>- BatchNorm<br>- ReLU<br>- AvgPool |
| **Dense Block 4** | Final feature concatenation | - Layers: 16<br>- Growth rate: 32 | Each layer has:<br>- Conv2D (3x3)<br>- BatchNorm<br>- ReLU<br>- Concatenation |
| **Global Pooling** | Feature aggregation | Global Average Pooling | - |
| **Dropout** | Regularization | Rate: 0.5 | - |
| **Output Layer** | Classification layer | - Dense layer<br>- Activation: sigmoid<br>- Units: num_classes | - |

**Additional Implementation Details:**
- Weight regularization: L2 (1e-4)
- Weight initialization: He normal
- Each Dense Block concatenates outputs from all its previous layers
- Conv blocks use kernel_size=(3,3) by default
- No bias in convolutional layers
- Loss function: Binary Focal crossentropy, labels smoothed by 0.1, gamma=2, alpha=0.25
- Optimizer: Adam, learning rate: 1e-3

**Difference from v4:**
- Optimize with clipnorm=3.0 
- Reduce factor from 0.5 to 0.3
- Increase l2 from 1e-4 to 5e-4