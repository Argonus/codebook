import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
datasets_path = os.path.join(project_root, "datasets")

"""RAW DATASET - path to dataset downloaded from kaggle"""
RAW_DATASET = os.path.join(datasets_path, "raw-data", "nih-dataset")
"""CLEARED_DATASET - path to dataset after data filtering"""
CLEARED_DATASET = os.path.join(datasets_path, "cleared-data", "nih-dataset")
"""TF_RECORD_DATASET - path to tf record file"""
TF_RECORD_DATASET = os.path.join(datasets_path, "tfrecord-dataset", "nih-dataset")