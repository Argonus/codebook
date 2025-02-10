from src.calculations import standard_deviation_bounds
import pytest

def test_three_standard_deviation_bounds():
    mean = 3.0
    std = 0.2

    lower_bound, upper_bound = standard_deviation_bounds(mean, std, 3)
    assert lower_bound == 2.4
    assert upper_bound == 3.6

def test_raises_error_on_invalid_std():
    mean = 3
    std = -1

    with pytest.raises(ValueError):
        standard_deviation_bounds(mean, std, 3)

def test_raises_error_on_invalid_n():
    mean = 3
    std = 0.2

    with pytest.raises(ValueError):
        standard_deviation_bounds(mean, std, -1)