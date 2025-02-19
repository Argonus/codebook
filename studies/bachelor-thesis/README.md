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

