"""One-shot: add IXI-DTI and Isles2022 NIfTI paths to manifest."""
import csv, os
from pathlib import Path

SRC_ROOT = Path(r"C:\projetos\Datasets")
MANIFEST_PATH = Path(r"D:\Dataset MRI\manifest.csv")
MANIFEST_COLS = [
    "file_path","source_dataset","subject_id","anatomy","sequence","plane","is_3d",
    "vendor","field_strength_T","slice_thickness_mm","pixel_spacing_mm","rows","cols",
    "original_format","has_noise_map","pathology","split",
]

existing = set()
with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        existing.add(row["file_path"])
print(f"Existing: {len(existing)}")

new_records = []
datasets_to_add = [
    {"src": SRC_ROOT / "IXI-DTI",  "ds": "IXI-DTI",   "anatomy": "brain", "vendor": "philips", "field_T": "1.5", "pathology": "normal", "max": 3000},
    {"src": SRC_ROOT / "Isles2022","ds": "Isles2022",  "anatomy": "brain", "vendor": "unknown", "field_T": "1.5", "pathology": "stroke", "max": 1500},
]

for cfg in datasets_to_add:
    found = []
    for root, _, files in os.walk(cfg["src"]):
        for f in files:
            if f.lower().endswith((".nii", ".nii.gz", ".h5")):
                found.append(Path(root) / f)
                if len(found) >= cfg["max"]:
                    break
        if len(found) >= cfg["max"]:
            break
    print(f"{cfg['ds']}: {len(found)} files found")
    for p in found:
        fp = str(p)
        if fp not in existing:
            new_records.append({
                "file_path": fp, "source_dataset": cfg["ds"],
                "subject_id": p.parent.name, "anatomy": cfg["anatomy"],
                "sequence": "", "plane": "", "is_3d": "True",
                "vendor": cfg["vendor"], "field_strength_T": cfg["field_T"],
                "slice_thickness_mm": "", "pixel_spacing_mm": "",
                "rows": "", "cols": "", "original_format": "nifti",
                "has_noise_map": "False", "pathology": cfg["pathology"],
                "split": "unassigned",
            })
            existing.add(fp)

print(f"New records: {len(new_records)}")
with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
    w.writerows(new_records)
print("Done.")
