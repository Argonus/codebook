import pytest
import pandas as pd

from src.data_entry_utils import extract_patient_info, extract_image_info

@pytest.fixture
def sample_row():
    return pd.Series({
        "Patient ID": 1,
        "Patient Age": 45,
        "Patient Gender": "M",
        "Finding Labels": "Cardiomegaly|Emphysema",
        "Image Index": "image001.png",
        "View Position": "PA",
        "OriginalImage[Width": 1024,
        "Height]": 1024
    })

def test_extract_patient_info(sample_row):
    result = extract_patient_info(sample_row)

    assert isinstance(result, dict)
    assert result["patient_id"] == 1
    assert result["patient_age"] == 45
    assert result["patient_gender"] == "M"

def test_extract_patient_info_missing_data(sample_row):
    # Remove a key to simulate missing data
    sample_row = sample_row.drop("Patient Gender")

    with pytest.raises(KeyError):
        extract_patient_info(sample_row)

def test_extract_image_info(sample_row):
    result = extract_image_info(sample_row)

    assert isinstance(result, dict)
    assert result["image_index"] == "image001.png"
    assert result["finding_labels"] == ["Cardiomegaly", "Emphysema"]
    assert result["view_position"] == "PA"
    assert result["image_width"] == 1024
    assert result["image_height"] == 1024

def test_extract_image_info_missing_data(sample_row):
    # Remove a key to simulate missing data
    sample_row = sample_row.drop("View Position")

    with pytest.raises(KeyError):
        extract_image_info(sample_row)
