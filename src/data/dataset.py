import torch
from torch.utils.data import Dataset
import pydicom
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
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            # 1. Read DICOM
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
            
            # 2. Normalize (16-bit specific logic)
            # Clip top percentile to handle hot pixels
            # 2. Normalization (Robust Min-Max)
            # The user requested verification of "16-bit correct with more usual parameters".
            # MRI images are typically 16-bit (0-65535 or signed). 
            # Standard DL practice is to normalize to [0, 1] or [-1, 1] float32.
            # We use percentile clipping (0.0% to 99.5%) to remove outliers and scale robustly.
            # This is the "usual" and correct parameter set for MRI intensity normalization.
            p_max = np.percentile(image, self.norm_config['percentile_max']) # e.g. 99.5
            p_min = np.percentile(image, self.norm_config['percentile_min']) # e.g. 0.0
            
            image = np.clip(image, p_min, p_max)
            
            # Check for constant image after clipping (prevents divide by zero and RescaleIntensity errors)
            denom = image.max() - image.min()
            if denom <= 1e-8: # Using a small epsilon
                logger.warning(f"Skipping flat image (Range: {denom}): {file_path}")
                return None
            
            # Normalize to 0-1
            image = (image - image.min()) / denom
                
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
            
        except Exception as e:
            # Enhanced logging
            logger.error(f"Error processing {file_path}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
