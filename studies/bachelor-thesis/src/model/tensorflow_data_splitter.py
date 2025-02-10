"""TensorFlow data loader module for medical imaging datasets."""
import gc
import os
import tempfile
import json
import pickle

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from typing import Dict, List, Tuple
from skmultilearn.model_selection import iterative_train_test_split

from src.model.tensorflow_utils import to_bytes_feature, to_float_feature, to_int64_feature
from src.utils.consts import NUM_CLASSES

class DatasetSplitter:
    """
    Class for splitting TFRecord datasets into stratified train/validation/test sets
    with proper representation of each disease class, supporting multiple features.
    """
    
    def __init__(self, feature_description: Dict, collect_statistics: bool = True):
        """
        Initialize the TFRecord splitter with support for multiple features.
        :params feature_description: Dictionary describing the TFRecord feature structure
        """
        self.feature_description = feature_description
        self.label_key = "encoded_finding_labels"
  
        # Set later when data is loaded
        self.original_dataset = None

        # For dataset analysis
        self.analyze_results = None    
        self.collect_statistics = collect_statistics

        # For stratified splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.split_info = {}
        
        # For memory extraction
        self.examples_list = None
        self.labels_list = None
        
        # For disk storage
        self.temp_dir = None
        self.example_paths = None
        self.labels_file = None

    def split_dataset(self, dataset: tf.data.Dataset, val_ratio: float, test_ratio: float) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        self.original_dataset = dataset
        self.analyze_results = self._analyze_dataset()
        gc.collect()

        self._create_stratified_splits(val_ratio, test_ratio)
        gc.collect()

        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def split_statistics(self) -> Dict:
        return self.split_info
    
    def _analyze_dataset(self) -> Dict:
        """Analyze the loaded dataset and return statistics on labels."""
        if not self.collect_statistics:
            return {}
        
        total_samples = 0
        class_counts = {}
        
        for record in tqdm(self.original_dataset, desc="Analysis progress: "):
            labels = record[self.label_key]
            if isinstance(labels, tf.sparse.SparseTensor):
                labels = tf.sparse.to_dense(labels)            

            label_np = labels.numpy()            
            flat_labels = np.sort(np.atleast_1d(label_np).flatten())
            class_id = "_".join(map(str, flat_labels))
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            total_samples += 1
        
        # Get unique class combinations
        unique_class_ids = list(class_counts.keys())
        
        # Calculate statistics using the string keys directly
        stats = {
            'total_samples': total_samples,
            'class_counts': [class_counts[class_id] for class_id in unique_class_ids],
            'class_percentages': [class_counts[class_id]/total_samples*100 for class_id in unique_class_ids] if total_samples > 0 else [],
            'class_ids': unique_class_ids  # Store the mapping of indices to actual class combinations
        }
        
        return stats

    def _create_stratified_splits(self, val_ratio: float, test_ratio: float) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create stratified train/val/test splits preserving disease distribution using
        iterative stratification, which is designed for multi-label datasets.
        :param val_ratio: Proportion for validation set
        :param test_ratio: Proportion for test set
        :return: Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.examples_list is None or self.labels_list is None:
            self.examples_list, self.labels_list = self._extract_dataset_to_memory()
        
        X = np.array([[i] for i in range(len(self.examples_list))])
        y = np.zeros((len(self.labels_list), NUM_CLASSES), dtype=np.int32)
        
        for i, label in enumerate(self.labels_list):
            label_array = np.asarray(label)
            
            if hasattr(label_array, 'indices'):
                label_indices = tf.sparse.to_dense(label_array)
            elif isinstance(label, list) and len(label) > 0 and hasattr(label[0], 'indices'):
                label_indices = tf.sparse.to_dense(label[0])
            else:
                label_indices = label_array
                
            indices = np.atleast_1d(label_indices).astype(np.int32)
            for idx in indices:
                if 0 <= idx < NUM_CLASSES:
                    y[i, idx] = 1

        # 0.2 / (1 - 0.2) = 0.5
        val_size_relative = val_ratio / (1 - test_ratio)

        x_temp, y_temp, x_test, y_test = iterative_train_test_split(X, y, test_size=test_ratio)
        x_train, y_train, x_val, y_val = iterative_train_test_split(x_temp, y_temp, test_size=val_size_relative)
        
        train_indices = [x[0] for x in x_train]
        val_indices = [x[0] for x in x_val]
        test_indices = [x[0] for x in x_test]
        
        self.split_info = {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }
        
        self.train_dataset = self._create_dataset_from_examples(train_indices, 'train')
        self.val_dataset = self._create_dataset_from_examples(val_indices, 'val')
        self.test_dataset = self._create_dataset_from_examples(test_indices, 'test')
        
        if self.collect_statistics:
            self.split_info['label_distribution'] = {
                'train': self._analyze_split_distribution(train_indices),
                'val': self._analyze_split_distribution(val_indices),
                'test': self._analyze_split_distribution(test_indices)
            }
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def _extract_dataset_to_memory(self) -> Tuple[List[Dict], List]:
        """
        Extract dataset and store examples on disk to reduce memory usage.
        Only keeps indices and labels in memory.
        :return: Tuple of (example_indices, labels_list)
        """
        print("Extracting dataset with disk storage to reduce memory usage...")
        
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="tf_dataset_")
            print(f"Created temporary directory at {self.temp_dir}")
        
        example_indices = []
        labels_list = []        
        self.labels_file = os.path.join(self.temp_dir, "labels.npy")
        
        chunk_size = 10000
        num_chunks = 0
        current_chunk = []
        current_labels = []
        
        for i, record in enumerate(tqdm(self.original_dataset, desc="Extraction progress: ")):
            current_chunk.append(record)            
            label = record[self.label_key]
            # Handle SparseTensor objects
            if isinstance(label, tf.sparse.SparseTensor):
                label = tf.sparse.to_dense(label)
            
            label_data = label.numpy().tolist()
            current_labels.append(label_data)
            
            if len(current_chunk) >= chunk_size:
                chunk_file = os.path.join(self.temp_dir, f"chunk_{num_chunks}.pickle")
                self._write_chunk_to_disk(current_chunk, chunk_file)
                
                chunk_indices = list(range(i - len(current_chunk) + 1, i + 1))
                example_indices.extend(chunk_indices)                
                labels_list.extend(current_labels)
                
                current_chunk = []
                current_labels = []
                num_chunks += 1
                
                gc.collect()
        
        if current_chunk:
            chunk_file = os.path.join(self.temp_dir, f"chunk_{num_chunks}.pickle")
            self._write_chunk_to_disk(current_chunk, chunk_file)
            remaining_start = len(example_indices)
            chunk_indices = list(range(remaining_start, remaining_start + len(current_chunk)))

            example_indices.extend(chunk_indices)
            labels_list.extend(current_labels)
        
        self.example_paths = {
            'temp_dir': self.temp_dir,
            'num_chunks': num_chunks + 1,
            'chunk_size': chunk_size
        }
        
        with open(os.path.join(self.temp_dir, "metadata.json"), 'w') as f:
            json.dump(self.example_paths, f)
        
        print(f"Dataset extraction complete. Stored in {num_chunks + 1} chunks on disk.")
        print(f"Memory usage reduced by keeping only indices and labels in memory.")        
        gc.collect()
        
        return example_indices, labels_list
    
    def _write_chunk_to_disk(self, examples, filename: str):
        """
        Write a chunk of examples to disk using pickle to preserve exact tensor types.
        This avoids the complex TFRecord serialization that was causing data type issues.
        """        
        with open(filename, 'wb') as f:
            pickle.dump(examples, f)
    

    def _analyze_split_distribution(self, indices):
        """
        Analyze the label distribution in a split.
        :params indices: Indices for the split
        :return Dictionary with class distribution
        """
        if not self.collect_statistics:
            return {}
            
        split_labels = []
        for i in indices:
            if 0 <= i < len(self.labels_list):
                split_labels.append(self.labels_list[i])
            else:
                print(f"Warning: Index {i} out of range")
        
        if not split_labels:
            return {}
        
        class_counts = {}
        for label in split_labels:
            try:
                if isinstance(label, list):
                    flattened = []
                    if label and isinstance(label[0], list):
                        for sublist in label:
                            flattened.extend(sublist)
                    else:
                        flattened = label
                    
                    if flattened:
                        max_idx = flattened.index(max(flattened))
                        class_key = str(max_idx)
                    else:
                        class_key = "0"
                else:
                    label_array = np.asarray(label)
                    if label_array.size > 0:
                        max_idx = np.argmax(label_array.flatten())
                        class_key = str(max_idx)
                    else:
                        class_key = "0"
                
                if class_key in class_counts:
                    class_counts[class_key] += 1
                else:
                    class_counts[class_key] = 1
            except Exception as e:
                print(f"Error analyzing label distribution: {e}")        

    def _create_dataset_from_examples(self, indices, split_name):
        """
        Create a dataset from examples stored on disk.
        :param indices: List of indices to include in the dataset
        :return: TensorFlow Dataset with parsed examples
        """        
        chunk_size = self.example_paths['chunk_size']
        chunks_to_read = {}
        
        for idx in tqdm(indices, desc=f"Creating {split_name} dataset from examples: "):
            chunk_id = idx // chunk_size
            if chunk_id not in chunks_to_read:
                chunks_to_read[chunk_id] = []
            chunks_to_read[chunk_id].append(idx % chunk_size)

        examples_count = 0        
        temp_tfrecord = os.path.join(self.temp_dir, f"temp_{split_name}_dataset.tfrecord")
        
        with tf.io.TFRecordWriter(temp_tfrecord) as writer:
            for chunk_id, local_indices in tqdm(chunks_to_read.items(), desc="Loading and writing chunks: "):
                chunk_file = os.path.join(self.temp_dir, f"chunk_{chunk_id}.pickle")
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as f:
                        chunk_examples = pickle.load(f)
                    
                    for local_idx in sorted(local_indices):
                        if local_idx < len(chunk_examples):
                            example = chunk_examples[local_idx]
                            feature_dict = {}
                            
                            for key, tensor in example.items():
                                feature_dict[key] = self._parse_feature(key, tensor)
                            
                            tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                            writer.write(tf_example.SerializeToString())
                            examples_count += 1
                else:
                    print(f"Warning: Chunk file {chunk_file} not found")
                

        dataset = tf.data.TFRecordDataset(temp_tfrecord)
        parsed_dataset = dataset.map(self._parse_function)
        
        return parsed_dataset

    def _parse_feature(self, key, tensor):
        """
        Encode a tensor based on its expected type in feature_description.
        :param key: Feature key
        :param tensor: TensorFlow tensor or SparseTensor
        :return: tf.train.Feature
        """
        if key not in self.feature_description:
            raise ValueError(f"Feature {key} not found in feature_description")
            
        feature_config = self.feature_description[key]
        expected_dtype = feature_config.dtype
        
        if isinstance(tensor, tf.sparse.SparseTensor):
            tensor = tf.sparse.to_dense(tensor)
        
        if expected_dtype == tf.string:
            value = tensor.numpy()
            
            if key == "image":
                if not isinstance(value, bytes):
                    print(f"Warning: Image data for {key} is not bytes, type={type(value)}")
                    if hasattr(value, 'tobytes'):
                        value = value.tobytes()
                    else:
                        value = bytes(value)
            else:
                if not isinstance(value, bytes):
                    value = str(value).encode()
                    
            return to_bytes_feature(value)
            
        elif expected_dtype == tf.int64:
            values = tf.cast(tensor, tf.int64).numpy().flatten().tolist()
            return to_int64_feature(values)
            
        elif expected_dtype == tf.float32:
            values = tf.cast(tensor, tf.float32).numpy().flatten().tolist()
            return to_float_feature(values)
            
        else:
            raise ValueError(f"Unsupported dtype in feature_description: {expected_dtype}")
    
    def _parse_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.feature_description)    

    @staticmethod
    def _create_stratification_key(labels_list):
        """
        Create a stratification key from multi-label labels.
        :param labels_list: list of labels
        :return: List of stratification keys
        """
        strat_keys = []
        for label in labels_list:
            if isinstance(label, (list, np.ndarray)):
                label_array = np.asarray(label)
                flat_labels = np.sort(np.atleast_1d(label_array).flatten())
                key = "_".join(map(str, flat_labels))
            else:
                key = str(label)

            strat_keys.append(key)
            
        return strat_keys    
    
    def __del__(self):
        """Clean up temporary directory when object is destroyed"""
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            print(f"Cleaning up temporary directory {self.temp_dir}")
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
