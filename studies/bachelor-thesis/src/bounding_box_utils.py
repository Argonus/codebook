import pandas as pd

def extract_bbox_data(bounding_boxes_df: pd.DataFrame, image_index: str) -> dict:
    bbox = bounding_boxes_df[bounding_boxes_df["Image Index"] == image_index]

    if not bbox.empty:
        box_label = bbox["Finding Label"].iloc[0]
        return {
            "bbox_finding_label": box_label,
            "x_coords": float(bbox["x"].iloc[0]),
            "y_coords": float(bbox["y"].iloc[0]),
            "widths": float(bbox["w"].iloc[0]),
            "heights": float(bbox["h"].iloc[0])
        }
    else:
        return {}
