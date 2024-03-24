import pytest
from src import NumericSystems

def test_napierRepresentation():
    number = NumericSystems()
    assert number.napierRepresentation('256') == 100

def test_napierAddition():
    number = NumericSystems()
    assert number.napierAddition('0235', '256') ==  [0, 2, 2, 3, 5, 5, 6]

def test_napierMultiplication():
    number = NumericSystems()
    assert number.napierMultiplication('0235', '256') == [
        [0, 2], [0, 5], [0, 6],
        [2, 2], [2, 5], [2, 6],
        [3, 2], [3, 5], [3, 6],
        [5, 2], [5, 5], [5, 6]
    ]

def test_numbersWithBias():
    number = NumericSystems()
    assert number.binaryWithBias('101', 2) == [1, 0, 1]