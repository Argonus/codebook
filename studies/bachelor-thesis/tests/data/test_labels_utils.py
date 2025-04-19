import pytest
# Import the functions you want to test
from src.data.labels_utils import encode_labels, decode_labels

def test_encode_labels_basic():
    labels = ["No Finding", "Atelectasis", "Cardiomegaly"]
    labels_dict = {"No Finding": 0, "Atelectasis": 1, "Cardiomegaly": 2}

    expected = [0, 1, 2]
    assert encode_labels(labels, labels_dict) == expected

def test_encode_labels_unknown():
    labels = ["No Finding", "Unknown Label", "Cardiomegaly"]
    labels_dict = {"No Finding": 0, "Atelectasis": 1, "Cardiomegaly": 2}

    expected = [0, -1, 2]
    assert encode_labels(labels, labels_dict) == expected

def test_decode_labels_basic():
    encoded = [0, 1, 2]
    labels_dict = {"No Finding": 0, "Atelectasis": 1, "Cardiomegaly": 2}

    expected = ["No Finding", "Atelectasis", "Cardiomegaly"]
    assert decode_labels(encoded, labels_dict) == expected

def test_decode_labels_unknown():
    encoded = [0, -1, 2]
    labels_dict = {"No Finding": 0, "Atelectasis": 1, "Cardiomegaly": 2}

    expected = ["No Finding", "Unknown", "Cardiomegaly"]
    assert decode_labels(encoded, labels_dict) == expected
