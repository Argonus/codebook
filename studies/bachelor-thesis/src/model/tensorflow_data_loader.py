"""TensorFlow data loader module."""
import os
import tensorflow as tf
import shutil

from tqdm import tqdm
from src.model.tensorflow_data_splitter import DatasetSplitter
from src.utils.consts import TF_RECORD_DATASET, TF_BUFFER_SIZE

class DataLoader:
    def __init__(self) -> None:
        self.splitter = DatasetSplitter(feature_description=self._feature_description())
        self.train_dataset = self._load_datasets_from_disk("train")
        self.val_dataset = self._load_datasets_from_disk("val")
        self.test_dataset = self._load_datasets_from_disk("test")

    def get_datasets(self) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            self._load_and_store()
            self.train_dataset = self._load_datasets_from_disk("train")
            self.val_dataset = self._load_datasets_from_disk("val")
            self.test_dataset = self._load_datasets_from_disk("test")

        return self.train_dataset, self.val_dataset, self.test_dataset

    def _load_and_store(self) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        dataset = self._load_dataset()
        self.splitter.split_dataset(dataset, val_ratio=0.15, test_ratio=0.15)
        temp_dir = self.splitter.temp_dir
        
        train_temp_path = os.path.join(temp_dir, "temp_train_dataset.tfrecord")
        val_temp_path = os.path.join(temp_dir, "temp_val_dataset.tfrecord") 
        test_temp_path = os.path.join(temp_dir, "temp_test_dataset.tfrecord")
        
        train_dest_path = f"{TF_RECORD_DATASET}/train.tfrecord"
        val_dest_path = f"{TF_RECORD_DATASET}/val.tfrecord"
        test_dest_path = f"{TF_RECORD_DATASET}/test.tfrecord"
        
        os.makedirs(os.path.dirname(train_dest_path), exist_ok=True)
                
        if os.path.exists(train_temp_path):
            print(f"Copying train dataset from {train_temp_path} to {train_dest_path}")
            shutil.copy2(train_temp_path, train_dest_path)
        
        if os.path.exists(val_temp_path):
            print(f"Copying validation dataset from {val_temp_path} to {val_dest_path}")
            shutil.copy2(val_temp_path, val_dest_path)
        
        if os.path.exists(test_temp_path):
            print(f"Copying test dataset from {test_temp_path} to {test_dest_path}")
            shutil.copy2(test_temp_path, test_dest_path)

        return None

    def _load_dataset(self):
        dataset = tf.data.TFRecordDataset([f"{TF_RECORD_DATASET}/chest_xray_data.tfrecord"], buffer_size=TF_BUFFER_SIZE)
        dataset = dataset.map(lambda x: tf.io.parse_single_example(x, self._feature_description()))
        return dataset

    @staticmethod
    def _load_datasets_from_disk(set_name):
        if os.path.isfile(f"{TF_RECORD_DATASET}/{set_name}.tfrecord"):
            return tf.data.TFRecordDataset(f"{TF_RECORD_DATASET}/{set_name}.tfrecord", buffer_size=TF_BUFFER_SIZE)
        else:
            return None

    @staticmethod
    def _feature_description():
        return {
            "image": tf.io.FixedLenFeature([], tf.string),
            "encoded_finding_labels": tf.io.VarLenFeature(tf.int64)
        }