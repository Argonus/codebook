import pytest
from src import Interpolation


def test_interpolation_one_l():
    x_values = [0, 1, 2]
    y_values = [1, 2, 3]
    x = 3
    assert Interpolation(x_values, y_values).lagrange(x) == 4


def test_interpolation_two_l():
    x_values = [1, 1.3, 1.6, 1.9, 2.2]
    y_values = [0.1411, -1 * 0.6878, -1 * 0.9962, -1 * 0.5507, 0.3115]
    x = 1.5
    result = Interpolation(x_values, y_values).lagrange(x)
    assert round(result, 4) == -0.9774


def test_interpolation_one_n():
    x_values = [0, 1, 2]
    y_values = [1, 2, 3]
    x = 3
    assert Interpolation(x_values, y_values, True).newton(x) == 4


def test_interpolation_two_n():
    x_values = [1, 1.3, 1.6, 1.9, 2.2]
    y_values = [0.1411, -1 * 0.6878, -1 * 0.9962, -1 * 0.5507, 0.3115]
    x = 1.5
    result = Interpolation(x_values, y_values).newton(x)
    assert round(result, 4) == -0.9774
