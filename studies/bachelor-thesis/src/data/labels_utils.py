from typing import List
import csv


def encode_labels(labels: List[str], labels_dict: dict) -> List[int]:
    """
    Encodes a list of string labels into a list of integer indices.
    """
    return [labels_dict.get(label, -1) for label in labels]

def decode_labels(encoded_labels: List[int], labels_dict: dict) -> List[str]:
    """
    Decodes a list of integer indices into a list of string labels.
    """
    reversed_dict = {v: k for k, v in labels_dict.items()}
    return [reversed_dict.get(index, "Unknown") for index in encoded_labels]

def get_labels_dict(dict_path: str) -> dict:
    """
    Returns a dictionary mapping string labels to integer indices.
    """
    with open(dict_path, "r", ) as file:
        csvFile = csv.reader(file)
        next(csvFile, None)  # skip the headers
        return {row[0]: int(row[1]) for i, row in enumerate(csvFile)}