import pytest
import tensorflow as tf
from src.model.tensorflow_utils import to_bytes_feature, to_float_feature, to_int64_feature

# Tests for to_bytes_feature
def test_to_bytes_feature_with_string():
    feature = to_bytes_feature("hello")
    assert isinstance(feature, tf.train.Feature)
    assert feature.bytes_list.value[0] == b"hello"

def test_to_bytes_feature_with_bytes():
    feature = to_bytes_feature(b"world")
    assert isinstance(feature, tf.train.Feature)
    assert feature.bytes_list.value[0] == b"world"

def test_to_bytes_feature_with_invalid_type():
    with pytest.raises(ValueError):
        to_bytes_feature(123)

# Tests for to_float_feature
def test_to_float_feature_with_single_float():
    feature = to_float_feature(3.14)
    assert isinstance(feature, tf.train.Feature)
    assert feature.float_list.value[0] == pytest.approx(3.14)

def test_to_float_feature_with_list_of_floats():
    feature = to_float_feature([1.1, 2.2, 3.3])
    assert isinstance(feature, tf.train.Feature)
    assert list(feature.float_list.value) == pytest.approx([1.1, 2.2, 3.3])

def test_to_float_feature_with_single_int():
    feature = to_float_feature(42)
    assert isinstance(feature, tf.train.Feature)
    assert feature.float_list.value[0] == pytest.approx(42.0)

def test_to_float_feature_with_invalid_type():
    with pytest.raises(ValueError):
        to_float_feature("not a float")

# Tests for to_int64_feature
def test_to_int64_feature_with_single_int():
    feature = to_int64_feature(42)
    assert isinstance(feature, tf.train.Feature)
    assert feature.int64_list.value[0] == 42

def test_to_int64_feature_with_list_of_ints():
    feature = to_int64_feature([1, 2, 3])
    assert isinstance(feature, tf.train.Feature)
    assert list(feature.int64_list.value) == [1, 2, 3]

def test_to_int64_feature_with_invalid_type():
    with pytest.raises(ValueError):
        to_int64_feature(3.14)