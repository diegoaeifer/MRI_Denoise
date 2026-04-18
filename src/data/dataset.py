import torch
from torch.utils.data import Dataset
import pydicom
import nibabel as nib
import numpy as np
import os
import json
import logging
from .transforms import get_transforms

# Configure logging for dataset
logger = logging.getLogger(__name__)

class MRI_DICOM_Dataset(Dataset):
    def __init__(self, file_list, mode='train', config=None):
        """
        Args:
            file_list (list): List of absolute paths to DICOM files.
            mode (str): 'train', 'val', or 'test'.
            config (dict): Data configuration dictionary.
        """
        self.file_list = file_list
        self.mode = mode
        self.config = config
        self.norm_config = config['normalization']
        self.augment_config = config['augmentation']
        
        # Initialize transforms
        self.transform = get_transforms(mode, config)
        
    def __len__(self):
        return len(self.file_list)
    
    def _load_nifti(self, file_path):
        nii = nib.load(file_path)
        image = nii.get_fdata().astype(np.float32)
        
        # Assume (W, H, D) or (H, W, D) for 3D NIFTI
        # Some Niftis are 4D (H,W,D,Time). If 4D, pick first timepoint
        if image.ndim >= 4:
            image = image[..., 0]

        if image.ndim == 3:
            # Choose a random axis (0=Sagittal, 1=Coronal, 2=Axial) for multi-planar slicing during training
            if self.mode == 'train':
                axis = np.random.randint(0, 3)
            else:
                axis = 2 # Stick to Axial for consistent Validation/Test
            
            depth = image.shape[axis]
            lower_bound = int(depth * 0.15)
            upper_bound = int(depth * 0.85)
            if lower_bound >= upper_bound:
                lower_bound = 0
                upper_bound = depth

            if self.mode == 'train':
                slice_idx = np.random.randint(lower_bound, upper_bound)
            else:
                slice_idx = depth // 2
            
            # Extract the selected slice along the chosen axis
            if axis == 0:
                image = image[slice_idx, :, :]
            elif axis == 1:
                image = image[:, slice_idx, :]
            else:
                image = image[:, :, slice_idx]
        elif image.ndim > 3:
             raise ValueError(f"Unsupported Nifti dimensions: {image.shape}")

        return image

    def _load_dicom(self, file_path):
        ds = pydicom.dcmread(file_path)
        # Handle possible read errors or missing pixel data
        if not hasattr(ds, 'pixel_array'):
             raise ValueError("No pixel_array found in DICOM")

        image = ds.pixel_array.astype(np.float32)

        # Handle 3D volumes (multi-frame DICOM) by taking RANDOM slice during training
        if image.ndim == 3:
            # Assuming (D, H, W) or (Frames, H, W)
            depth = image.shape[0]
            if self.mode == 'train':
                # Random slice for training to see more data
                slice_idx = np.random.randint(0, depth)
            else:
                # Middle slice for validation/test for consistency
                slice_idx = depth // 2

            image = image[slice_idx]
        elif image.ndim > 3:
            # Should not happen for standard DICOM but safety first
            raise ValueError(f"Unsupported DICOM dimensions: {image.shape}")

        return image

    def _normalize_image(self, image, file_path):
        # 2. Normalize (16-bit specific logic)
        # Clip top percentile to handle hot pixels
        # 2. Normalization (Robust Min-Max)
        # The user requested verification of "16-bit correct with more usual parameters".
        # MRI images are typically 16-bit (0-65535 or signed).
        # Standard DL practice is to normalize to [0, 1] or [-1, 1] float32.
        # We use percentile clipping (0.0% to 99.5%) to remove outliers and scale robustly.
        # This is the "usual" and correct parameter set for MRI intensity normalization.

        # Optimize: compute quantiles instead of percentiles to avoid internal conversions,
        # and downsample for very large images (optimization implemented per bolt guidelines)
        # The performance of np.quantile on a strided array is much faster and highly accurate for
        # medical image normalization where exact single-pixel outliers don't drastically shift the quantile.

        # Subsample by taking every 4th pixel if image is large enough, else use full image
        # MRI images are typically 256x256 or 512x512.
        stride = 4 if image.shape[0] >= 128 and image.shape[1] >= 128 else 1

        q_min = self.norm_config['percentile_min'] / 100.0
        q_max = self.norm_config['percentile_max'] / 100.0

        p_min, p_max = np.quantile(
            image[::stride, ::stride],
            [q_min, q_max]
        )

        # In-place clip
        np.clip(image, p_min, p_max, out=image)

        # Support in-place normalization for memory efficiency
        # as verified in test_inplace_float32.py
        denom = float(p_max - p_min)
        if denom > 1e-8:
            image -= p_min
            image /= denom
        else:
            # Handle constant image after clipping
            logger.warning(f"Skipping flat image (Range: {denom}): {file_path}")
            return None
            
        return image

    def _apply_transforms(self, image, file_path):
        # Add channel dimension (1, H, W) for TorchIO
        # image = image[np.newaxis, ...]
        # FIX: TorchIO requires 4D tensor (C, Spatial...) if it detects it as such, or strict 4D.
        # Expanding to (1, H, W, 1) to simulate 3D volume of depth 1.
        # Explicitly ensure we are working with (H, W) here.
        if image.ndim != 2:
             # logger.error(f"Image {file_path} has shape {image.shape} after slicing")
             raise ValueError(f"Expected 2D image after slicing, got {image.shape}")

        # Explicitly construct 4D tensor (1, H, W, 1)
        # Avoid '...' to be safe against unexpected dims
        h, w = image.shape
        image = image.reshape(1, h, w, 1) # (1, H, W, 1)

        # 3. Apply Transforms (including spatial noise injection)

        import torchio as tio
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=torch.from_numpy(image)),
        )

        transformed_subject = self.transform(subject)

        # Extract tensors

        # The spatial noise transform (custom) should have added the noise and handling the sigma channel.
        noisy_image = transformed_subject['mri'].data

        sigma_map = transformed_subject.get('sigma_map', None)

        if sigma_map is None:
            # Creates a dummy zero map if not present ensuring code doesn't crash
             sigma_map = torch.zeros_like(noisy_image)

        # Stack for input: (2, H, W) -> [Noisy, Sigma]
        # noisy_image and sigma_map are (1, H, W, 1)
        # We want to return (2, H, W).

        # Squeeze dim -1
        noisy_image = noisy_image.squeeze(-1)
        sigma_map_data = sigma_map.data.squeeze(-1)

        if noisy_image.ndim != 3:
             raise ValueError(f"Expected 3D noisy_image (1, H, W), got {noisy_image.shape}")

        input_tensor = torch.cat([noisy_image, sigma_map_data], dim=0)

        # GT
        gt_tensor = transformed_subject['gt'].data.squeeze(-1)

        return {
            'input': input_tensor.float(), # (2, 128, 128)
            'target': gt_tensor.float(),   # (1, 128, 128)
            'file_path': str(file_path),
            'sigma_mean': sigma_map.data.mean().item()
        }

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        try:
            # Check file extension
            is_nifti = str(file_path).lower().endswith('.nii') or str(file_path).lower().endswith('.nii.gz')
            
            if is_nifti:
                image = self._load_nifti(file_path)
            else:
                image = self._load_dicom(file_path)

            image = self._normalize_image(image, file_path)
            if image is None:
                return None
            
            return self._apply_transforms(image, file_path)
            
        except Exception as e:
            # Enhanced logging
            logger.error(f"Error processing {file_path}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
