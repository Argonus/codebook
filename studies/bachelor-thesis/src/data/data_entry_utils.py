import pandas as pd
from typing import Tuple
from src.utils.calculations import standard_deviation_bounds

def remove_age_outliers(data: pd.DataFrame, n_std: int = 3) -> Tuple[pd.DataFrame, float]:
    """
    Remove age outliers from the data based on standard deviation.
    """
    mean = data.mean()
    std_dev = data.std()
    _, upper_bound = standard_deviation_bounds(mean, std_dev, n_std)
    return data[data <= upper_bound], upper_bound

def split_finding_labels(df):
    """
    Split the 'Finding Labels' column into a list of labels.
    """
    df_copy = df.copy()
    df_copy['Split Labels'] = df_copy['Finding Labels'].str.split('|')
    return df_copy

def extract_patient_info(row: pd.Series) -> dict:
    """
    Extract patient information from the row.
    """
    return {
        "patient_id": int(row["Patient ID"]),
        "patient_age": int(row["Patient Age"]),
        "patient_gender": row["Patient Gender"],
    }

def extract_image_info(row: pd.Series) -> dict:
    """
    Extract image information from the row.
    """
    return {
        "finding_labels": row["Finding Labels"].split('|'),
        "image_index": row["Image Index"],
        "view_position": row["View Position"],
        "image_width": float(row["OriginalImage[Width"]),
        "image_height": float(row["Height]"]),
    }