import tensorflow as tf
from typing import Union, List

def to_bytes_feature(value: Union[str, bytes]) -> tf.train.Feature:
    """Converts a string or bytes to TensorFlow feature format."""
    if isinstance(value, str):
        value = value.encode()
    elif not isinstance(value, bytes):
        raise ValueError(f"Expected str or bytes, got {type(value)}")

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_float_feature(value: Union[float, List[float]]) -> tf.train.Feature:
    """
    Converts a float or list of floats to TensorFlow feature format.
    """
    if isinstance(value, (float, int)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
    elif isinstance(value, list) and all(isinstance(v, (float, int)) for v in value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v) for v in value]))
    else:
        raise ValueError(f"Expected float or list of floats, got {type(value)}")

def to_int64_feature(value: Union[int, List[int]]) -> tf.train.Feature:
    """
    Converts an int or list of ints to TensorFlow feature format.
    """
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, list) and all(isinstance(v, int) for v in value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        raise ValueError(f"Expected int or list of ints, got {type(value)}")
