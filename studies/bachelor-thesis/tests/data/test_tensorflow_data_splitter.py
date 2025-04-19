import pytest
import tensorflow as tf
from src.data.tensorflow_data_splitter import DatasetSplitter

from src.utils.consts import TF_RECORD_DATASET

def test_returns_three_datasets_with_duplicates():
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "encoded_finding_labels": tf.io.VarLenFeature(tf.int64)
    }

    dataset = tf.data.TFRecordDataset(f"{TF_RECORD_DATASET}/chest_xray_data.tfrecord", buffer_size=262144)
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
    
    dataset = dataset.take(50)
    dataset = dataset.concatenate(dataset) # 100
    dataset = dataset.concatenate(dataset) # 200
    dataset = dataset.concatenate(dataset) # 400
    dataset = dataset.concatenate(dataset) # 800


    splitter = DatasetSplitter(feature_description=feature_description)
    train, val, test = splitter.split_dataset(dataset, val_ratio=0.2, test_ratio=0.2)

    # Check that the datasets are not empty
    assert train is not None
    assert val is not None
    assert test is not None


    train_size = sum(1 for _ in train)
    val_size = sum(1 for _ in val)
    test_size = sum(1 for _ in test)

    assert train_size == pytest.approx(479, 1)
    assert val_size == pytest.approx(161, 1)
    assert test_size == pytest.approx(160, 1)
    
    # Check that the datasets have the correct number of elements
    assert "image" in train.element_spec
    assert "encoded_finding_labels" in train.element_spec

    for record in train.take(1):
        assert record["image"]
        assert record["encoded_finding_labels"]

    for record in val.take(1):
        assert record["image"]
        assert record["encoded_finding_labels"]

    for record in test.take(1):
        assert record["image"]
        assert record["encoded_finding_labels"]


def test_returns_three_datasets_without_duplicates():
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "encoded_finding_labels": tf.io.VarLenFeature(tf.int64)
    }

    dataset = tf.data.TFRecordDataset(f"{TF_RECORD_DATASET}/chest_xray_data.tfrecord", buffer_size=262144)
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))

    dataset = dataset.take(50)
    splitter = DatasetSplitter(feature_description=feature_description)
    train, val, test = splitter.split_dataset(dataset, val_ratio=0.2, test_ratio=0.2)

    # Check that the datasets are not empty
    assert train is not None
    assert val is not None
    assert test is not None

    train_size = sum(1 for _ in train)
    val_size = sum(1 for _ in val)
    test_size = sum(1 for _ in test)

    assert train_size == pytest.approx(29, 1)
    assert val_size == pytest.approx(11, 1)
    assert test_size == pytest.approx(10, 1)

    # Check that the datasets have the correct number of elements
    assert "image" in train.element_spec
    assert "encoded_finding_labels" in train.element_spec

    for record in train.take(1):
        assert record["image"]
        assert record["encoded_finding_labels"]

    for record in val.take(1):
        assert record["image"]
        assert record["encoded_finding_labels"]

    for record in test.take(1):
        assert record["image"]
        assert record["encoded_finding_labels"]