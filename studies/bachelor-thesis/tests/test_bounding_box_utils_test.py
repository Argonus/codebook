import pytest
import pandas as pd

from src.bounding_box_utils import extract_bbox_data

@pytest.fixture
def sample_data_frame():
    return pd.DataFrame([{
        "Image Index": "img.png",
        "Finding Label": "Something",
        "x": "225.084745762712",
        "y": "547.019216763771",
        "h": "86.7796610169491",
        "w": "79.1864406779661"
    }], index=[0])

def test_extract_bbox_data_with_matching_index(sample_data_frame):
    result = extract_bbox_data(sample_data_frame, "img.png")

    assert isinstance(result, dict)
    assert result["bbox_finding_label"] == "Something"
    assert result["x_coords"] == 225.084745762712
    assert result["y_coords"] == 547.019216763771
    assert result["widths"] == 79.1864406779661
    assert result["heights"] == 86.7796610169491



def test_extract_bbox_data_with_invalid_index(sample_data_frame):
    result = extract_bbox_data(sample_data_frame, "invalid.png")

    assert isinstance(result, dict)
    assert result == {}



