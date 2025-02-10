# Chest X-Ray Image Classification

## Overview

This project implements a deep learning system for automated chest X-ray image classification using TensorFlow. The system is designed to identify multiple pathological conditions from X-ray images using state-of-the-art deep learning techniques.

## Project Structure

```
├── src/                    # Source code
│   ├── tensorflow_utils.py # TensorFlow utilities and data processing
│   ├── image_utils.py      # Image processing utilities
│   ├── data_entry_utils.py # Data entry and preprocessing
│   └── consts.py           # Project constants
├── datasets/               # Dataset directory
│   ├── raw-data/           # Original NIH dataset
│   ├── cleared-data/       # Preprocessed dataset
│   └── tfrecord-dataset/   # TFRecord format dataset
└── notebooks/          # Jupyter notebooks
    ├── DataAnalysis.ipynb # Jupyter notebook for data analysis
    ├── DataFiltering.ipynb # Jupyter notebook for data filtering
    ├── DataStorage.ipynb # Jupyter notebook for data storage
    ├── ModelTraining.ipynb # Jupyter notebook for model training
```

## Features

- Data Analysis and Visualization of findings in raw dataset
- Data Filtering and Preprocessing of a dataset to match model needs
- Data Storage as TFRecord format for model training
- Model Training and Evaluation

## Prerequisites

- Python 3.8+
- TensorFlow 2.x

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

## Data

The project uses the NIH Chest X-ray Dataset, one of the largest publicly available collections of chest X-ray images. The dataset contains over 100,000 X-ray images with multiple pathological conditions.

## Models

### Simplified_DensNet_v1

The Simplified_DensNet_v1 model is a simplified version of the DensNet model, that
does not implement bottleneck layers and has a smaller number of parameters. It is
trained on the NIH Chest X-ray Dataset and can be used for classification of
multiple pathological conditions.

#### Simplified_DensNet_v1 Implementation Details

- Number Of Layers: 63
- Oversampling for rare classes: None
- Batch Size: 32
- Learning Rate: 0.0001 
- Learning Rate Scheduling: ReduceLROnPlateau(monitor="val_f1", factor=0.5, patience=3, min_lr=1e-6)
- Loss Function: BinaryCrossentropy(label_smoothing=0.01)
- Optimizer: Adam
- Epochs: 20
- Data Augmentation: Yes (random brightness, contrast, shifting, Gaussian noise)
- Early Stopping: None

#### Simplified_DensNet_v1 Results

### Simplified_DensNet_v2

The Simplified_DensNet_v2 model is a simplified version of the DensNet model, that
does not implement bottleneck layers and has a smaller number of parameters. Main differences are:

#### Simplified_DensNet_v2 Implementation Details

- Loss Function: BinaryFocalCrossentropy(gamma=2.0, alpha=0.25, from_logits=False, label_smoothing=0.01)
Usage of Focal Loss instead of BinaryCrossentropy, may help with multi labels imbalanced dataset. 

#### Simplified_DensNet_v2 Results
