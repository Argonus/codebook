import pytest
import sympy as sp

from src import ContinuedFractions

def test_case_one():
    cf = ContinuedFractions(sp.sqrt)
    print("Continued Fractions for 12:")
    assert cf.findChain(12) == "3 2 6"

def test_case_two():
    cf = ContinuedFractions(sp.sqrt)
    print("Continued Fractions for 58:")
    assert cf.findChain(58) == "7 1 1 1 1 1 1 14"