import pytest
import tensorflow as tf
import numpy as np
from src.model.tensorflow_utils import to_bytes_feature, to_float_feature, to_int64_feature, get_sampling_rates

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

# Tests for get_sampling_rates
def test_get_sampling_rates_basic():
    class_weights = {i: float(i + 1) for i in range(15)}
    
    sampling_rates = get_sampling_rates(class_weights)
    rates = sampling_rates.numpy()
    
    assert rates.shape == (15,)
    
    # [THEN] No Finding should have rate 1.0
    assert rates[10] == 1.0

    # [THEN] All rates should be between 1.0 and 5.0
    assert np.all(rates >= 1.0)
    assert np.all(rates <= 5.0)
    
    for i in range(14):
        if i != 9 and i != 10:
            assert rates[i] <= rates[i+1]

def test_get_sampling_rates_custom_range():
    class_weights = {i: float(i) for i in range(5)}
    
    min_rate = 2.0
    max_rate = 10.0
    
    # [WHEN] Custom range is specified
    sampling_rates = get_sampling_rates(class_weights, min_rate, max_rate)
    rates = sampling_rates.numpy()
    
    # [THEN] All rates should be between the specified range
    assert np.all(rates >= min_rate)
    assert np.all(rates <= max_rate)
    
    # [THEN] First and last rates should be equal to the specified range
    assert rates[0] == min_rate
    assert rates[4] == max_rate

def test_get_sampling_rates_real_world():
    class_weights = {
        0: 5.2,   
        1: 18.9,  
        2: 27.3,  
        3: 12.0,  
        4: 4.8,   
        5: 25.1,  
        6: 28.0,  
        7: 26.0,  
        8: 1.2,   
        9: 14.7,  
        10: 0.13, 
        11: 9.5,  
        12: 20.8, 
        13: 22.1, 
        14: 12.5  
    }
    
    # [WHEN] Class weights are specified
    sampling_rates = get_sampling_rates(class_weights)
    rates = sampling_rates.numpy()
    
    # [THEN] No Finding should have rate 1.0
    assert rates[10] == 1.0
    
    # [THEN] Common classes should have sampling rates between 1.0 and 5.0
    assert rates[8] < 2.0 
    assert rates[6] > 4.0

    # [THEN] Higher weighted classes should have higher sampling rates
    assert rates[8] < rates[4]
    assert rates[4] < rates[0]
    
    # [THEN] Rare classes should have high sampling rates
    assert rates[2] > 4.0
    assert rates[6] > 4.0

def test_get_sampling_rates_edge_cases():
    identical_weights = {i: 1.0 for i in range(15)}
    rates_identical = get_sampling_rates(identical_weights).numpy()
    
    # [THEN] No Finding should have rate 1.0
    assert rates_identical[10] == 1.0
    
    non_no_finding_rates = np.delete(rates_identical, 10)
    assert np.allclose(non_no_finding_rates, non_no_finding_rates[0], rtol=1e-5)
    
    single_weight = {0: 1.0}
    rates_single = get_sampling_rates(single_weight).numpy()
    assert rates_single.shape == (1,)
    assert rates_single[0] == 5.0