import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


def _get_files(data_dir):
    files = []
    for ext in ["*.dcm", "*.nii.gz", "*.nii"]:
        files.extend(list(Path(data_dir).rglob(ext)))
    return [str(f) for f in files]


def build_datalist(cfg):
    """
    Builds a MONAI-style datalist dictionary:
    {"train": [{"image": path}, ...], "val": [...], "test": [...]}
    """
    data_dir = cfg.get("dataset_dir")
    cache_path = os.path.join(data_dir, "datalist_cache.json")

    if os.path.exists(cache_path) and cfg.get("use_cache", True):
        with open(cache_path, "r") as f:
            return json.load(f)

    # Gather all valid files
    all_files = _get_files(data_dir)

    # In a real scenario, you'd group by PatientID/SeriesUID to split properly.
    # Here is a basic 80/10/10 split over files.
    if not all_files:
        return {"train": [], "val": [], "test": []}

    train_files, test_val_files = train_test_split(
        all_files, test_size=0.2, random_state=42
    )
    val_files, test_files = train_test_split(
        test_val_files, test_size=0.5, random_state=42
    )

    datalist = {
        "train": [{"image": f} for f in train_files],
        "val": [{"image": f} for f in val_files],
        "test": [{"image": f} for f in test_files],
    }

    with open(cache_path, "w") as f:
        json.dump(datalist, f, indent=4)

    return datalist
