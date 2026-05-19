"""
IXI Dataset loader for multi-sequence MRI evaluation.

The IXI dataset contains 600+ volumes across 3 MRI sequences (T1, T2, PD)
for evaluating denoising across different sequence types.

Reference: https://brain-development.org/ixi-dataset/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List

import json
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

logger = logging.getLogger(__name__)


class IXIDatasetBuilder:
    """
    Builds IXI dataset splits for cross-sequence evaluation.

    Supports:
    - T1-weighted volumes
    - T2-weighted volumes
    - PD (Proton Density) volumes
    """

    # Standard IXI sequences
    SEQUENCES = ["T1", "T2", "PD"]

    def __init__(
        self,
        root_dir: str | Path,
        sequence: str = "T1",
        num_volumes: int = 10,
        spatial_dims: int = 2,
    ) -> None:
        """
        Args:
            root_dir: Path to IXI dataset root (should contain .nii.gz files)
            sequence: Sequence type ("T1", "T2", or "PD")
            num_volumes: Number of volumes to use (default: 10)
            spatial_dims: 2 for 2D slicing, 3 for 3D volumes
        """
        self.root_dir = Path(root_dir)
        self.sequence = sequence.upper()
        self.num_volumes = num_volumes
        self.spatial_dims = spatial_dims

        if self.sequence not in self.SEQUENCES:
            raise ValueError(f"Sequence must be one of {self.SEQUENCES}")

        self._validate_root()

    def _validate_root(self) -> None:
        """Check that root directory contains IXI data."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"IXI root directory not found: {self.root_dir}")

        nii_files = list(self.root_dir.glob(f"**/*{self.sequence}*.nii.gz"))
        if not nii_files:
            logger.warning(
                f"No {self.sequence} files found in {self.root_dir}. "
                "Please ensure IXI dataset is properly organized."
            )

    def build_datalist(self) -> Dict[str, List[Dict]]:
        """
        Build MONAI datalist from IXI files.

        Returns:
            Dict with 'train', 'val', 'test' splits
        """
        # Find all sequence files
        nii_files = sorted(self.root_dir.glob(f"**/*{self.sequence}*.nii.gz"))[
            : self.num_volumes
        ]

        if not nii_files:
            logger.warning(
                f"No {self.sequence} volumes found. Creating dummy datalist."
            )
            nii_files = []

        # Create data items (image is both input and reference for clean data)
        data_items = [
            {
                "image": str(f),
                "label": str(f),  # For denoising: clean reference is same image
            }
            for f in nii_files
        ]

        # Split into train/val/test (80/10/10)
        n = len(data_items)
        train_count = int(0.8 * n)
        val_count = int(0.1 * n)

        datalist = {
            "train": data_items[:train_count],
            "val": data_items[train_count : train_count + val_count],
            "test": data_items[train_count + val_count :],
        }

        logger.info(
            f"IXI {self.sequence} datalist: "
            f"train={len(datalist['train'])}, "
            f"val={len(datalist['val'])}, "
            f"test={len(datalist['test'])}"
        )

        return datalist

    def get_transforms(self) -> Dict[str, Callable]:
        """
        Get transform pipelines for train/val.

        Returns:
            Dict with 'train' and 'val' transform composers
        """
        from ..data.transforms import build_train_transforms, build_val_transforms

        return {
            "train": build_train_transforms(
                spatial_dims=self.spatial_dims,
                image_size=(256, 256) if self.spatial_dims == 2 else (256, 256, 64),
                noise_sigma_range=(0.01, 0.15),  # Realistic MRI noise
            ),
            "val": build_val_transforms(
                spatial_dims=self.spatial_dims,
                image_size=(256, 256) if self.spatial_dims == 2 else (256, 256, 64),
            ),
        }

    def create_datasets(
        self,
        cache_rate: float = 0.5,
        num_workers: int = 4,
    ) -> Dict[str, CacheDataset]:
        """
        Create MONAI CacheDatasets.

        Args:
            cache_rate: Fraction of data to cache in RAM
            num_workers: Number of workers for data loading

        Returns:
            Dict with 'train', 'val', 'test' datasets
        """
        datalist = self.build_datalist()
        transforms = self.get_transforms()

        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = CacheDataset(
                data=datalist[split],
                transform=transforms.get(split, transforms["val"]),
                cache_rate=cache_rate,
                num_workers=num_workers,
            )

        logger.info(
            f"Created IXI {self.sequence} datasets: "
            f"train={len(datasets['train'])}, "
            f"val={len(datasets['val'])}, "
            f"test={len(datasets['test'])}"
        )

        return datasets

    @staticmethod
    def create_multi_sequence_datalist(
        root_dir: str | Path,
        num_volumes_per_sequence: int = 10,
        spatial_dims: int = 2,
    ) -> Dict[str, Dict]:
        """
        Create datalists for all sequences.

        Args:
            root_dir: Path to IXI dataset
            num_volumes_per_sequence: Volumes per sequence type
            spatial_dims: 2 or 3

        Returns:
            Dict with datalists for each sequence (T1, T2, PD)
        """
        result = {}
        for seq in IXIDatasetBuilder.SEQUENCES:
            builder = IXIDatasetBuilder(
                root_dir=root_dir,
                sequence=seq,
                num_volumes=num_volumes_per_sequence,
                spatial_dims=spatial_dims,
            )
            result[seq] = builder.build_datalist()

        return result

    @staticmethod
    def save_datalist(datalist: Dict, output_path: str | Path) -> None:
        """Save datalist to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            # Convert Path objects to strings for JSON serialization
            datalist_serializable = {}
            for split, items in datalist.items():
                datalist_serializable[split] = [
                    {k: str(v) if isinstance(v, Path) else v for k, v in item.items()}
                    for item in items
                ]
            json.dump(datalist_serializable, f, indent=2)
        logger.info(f"Saved datalist to {output_path}")
