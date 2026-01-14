import torch
import torchio as tio
import numpy as np
import random
from torch.nn.functional import interpolate
import logging

logger = logging.getLogger(__name__)

class SpatiallyVaryingGaussianNoise(tio.Transform):
    def __init__(self, sigma_range=(0.02, 0.3), grid_size=4, multiplier_range=(0.5, 3.0), target_size=(128, 128), p=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma_range = sigma_range
        self.grid_size = grid_size
        self.multiplier_range = multiplier_range
        self.target_size = target_size
        self.p = p

    def apply_transform(self, subject):
        if not isinstance(subject, tio.Subject):
            return subject

        if random.random() > self.p:
            for image_name, image in subject.get_images_dict(intensity_only=True).items():
                if image_name == 'gt': continue 
                sigma_map = torch.zeros_like(image.data)
                subject.add_image(tio.ScalarImage(tensor=sigma_map, name='sigma_map'), 'sigma_map')
            return subject

        # Generate Noise Params
        sigma_base = random.uniform(*self.sigma_range)
        
        # Grid - Uniform(min, max)
        mod_grid = torch.FloatTensor(1, 1, self.grid_size, self.grid_size).uniform_(*self.multiplier_range)
        
        for image_name, image in subject.get_images_dict(intensity_only=True).items():
            if image_name == 'gt': continue

            data = image.data 

            if data.ndim != 4:
                continue

            H, W = data.shape[1], data.shape[2]
            
            # Upsample
            mod_map = interpolate(mod_grid, size=(H, W), mode='bicubic', align_corners=False) # (1, 1, H, W)
            
            # Remove batch dimension (dim 0) -> (1, H, W)
            mod_map = mod_map.squeeze(0)
            
            # Add spatial depth dim -> (1, H, W, 1)
            mod_map = mod_map.unsqueeze(-1)
            
            sigma_map = sigma_base * mod_map
            
            noise = torch.randn_like(data)
            weighted_noise = noise * sigma_map
            
            noisy_data = data + weighted_noise
            image.set_data(noisy_data)
            
            subject.add_image(tio.ScalarImage(tensor=sigma_map.cpu(), name='sigma_map'), 'sigma_map')
            
        return subject

class CopyMRIToGT(tio.Transform):
    """Explicit Transform to copy MRI to GT."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        if not isinstance(subject, tio.Subject):
             return subject
        
        if 'mri' in subject:
             # logger.info(f"CopyMRIToGT MRI Shape: {subject['mri'].data.shape}")
             subject.add_image(tio.ScalarImage(tensor=subject['mri'].data.clone(), name='gt'), 'gt')
             
        return subject

class RandomRot90(tio.Transform):
    """
    Random 90-degree rotations (k=1 or k=3) with probability p.
    Avoids interpolation artifacts.
    """
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def apply_transform(self, subject):
        if random.random() > self.p:
            return subject
            
        # 50% chance of +90 (k=1), 50% chance of -90 (k=3)
        k = 1 if random.random() > 0.5 else 3
        
        # logger.info(f"Applying Rot90 (k={k})")
        
        for image_name, image in subject.get_images_dict(intensity_only=False).items():
            # Data is (C, H, W, D). We rotate in H,W plane (dims 1, 2)
            # Check shape
            data = image.data
            if data.ndim >= 3:
                # Rotate
                data = torch.rot90(data, k, dims=(1, 2))
                image.set_data(data)
                
        return subject

def get_transforms(mode, config):
    """
    Returns the composed transforms for the given mode.
    """
    patch_size = config['patch_size'] 
    aug_cfg = config['augmentation']
    
    transforms = []
    
    # 0. Preparation: Copy mri to gt
    transforms.append(CopyMRIToGT())
    
    # 1. Random Crop (Standardize input size)
    transforms.append(tio.CropOrPad(target_shape=(patch_size[0], patch_size[1], 1)))

    # NOTE: Applying augmentations in ALL modes for now to visualize effects in Validation/Test.
    # In a pure production testing scenario, we might want to disable some, but for this pipeline
    # we want to validate robustness against these artifacts.

    # 2. Geometric - Initial (Physical distortion during acquisition)
    transforms.append(tio.RandomAffine(scales=0, degrees=0, translation=5, p=aug_cfg.get('affine_prob', 0.2)))
    
    # Keep Anisotropy (Resolution artifact - often happens during acquisition)
    # Downsampling ratio (default 1.5 in config)
    ani_down = aug_cfg.get('anisotropy_downsampling', 1.5)
    # If single float, TorchIO expects range or tuple? It expects scalar or tuple.
    # We'll use (1.1, num) range or just num? Let's use tuple (1.0, ani_down) or similar.
    # Config says 'anisotropy_downsampling: 1.5'. Let's use (1.1, 1.5) as hardcoded or range?
    # User asked for 'options for the strength'. Let's interpret the config value as the MAX downsampling.
    transforms.append(tio.RandomAnisotropy(axes=(0, 1), downsampling=(1.1, float(ani_down)), p=aug_cfg.get('anisotropy_prob', 0.1)))
    
    # 3. Degradations (Sequence: Acquisition -> Artifacts -> Noise -> Post)
    
    # Acquisition/Intensity effects
    # Gamma Log range
    gamma_range = aug_cfg.get('gamma_range', [0.8, 1.2]) 
    gamma = tio.RandomGamma(log_gamma=tuple(gamma_range), p=aug_cfg.get('gamma_prob', 0.2))
    
    # Bias Field Coefficients
    bias_coeffs = aug_cfg.get('bias_field_coeffs', 0.3)
    bias = tio.RandomBiasField(coefficients=float(bias_coeffs), p=aug_cfg.get('bias_field_prob', 0.0))
    
    # Artifacts (K-space based)
    ghost = tio.RandomGhosting(p=aug_cfg.get('ghosting_prob', 0.1))
    spike = tio.RandomSpike(p=aug_cfg.get('spike_prob', 0.1))
    motion = tio.RandomMotion(p=aug_cfg.get('motion_prob', 0.0))
    
    # Optical/Filter effects
    blur = tio.RandomBlur(p=aug_cfg.get('blur_prob', 0.0))
    
    # Assign degradations only to 'mri' (except Affine/Flip/Rotation/Normalize)
    # Note: Affine/Anisotropy above apply to both MRI and GT?
    # Wait, RandomAffine should apply to GT if it's geometric? 
    # Usually for denoising, we want Input to be misaligned? NO.
    # We want Input = Degraded(GT).
    # Geometric shifts (Affine) should apply to BOTH or just Input? 
    # If it's motion *during* scan, it's complex. 
    # If it's just data augmentation (rotation/flip), it must apply to both.
    # RandomAffine here is small translation. Let's keep it for both.
    
    # Artifacts apply ONLY to MRI (Input)
    # Correct way: Affine/Flip/Rotation apply to all (Geometric).
    # The degradations below apply only to 'mri'.
    for deg in [gamma, bias, ghost, spike, motion, blur]:
        deg.include = ['mri']
        transforms.append(deg)
    
    # 4. Post-processing Augmentations (Rotation/Flip)
    transforms.append(tio.RandomFlip(axes=(0, 1), p=aug_cfg.get('flip_prob', 0.5)))
    transforms.append(RandomRot90(p=aug_cfg.get('rotate_prob', 0.5)))

    # 4. Spatially Varying Noise (Use for Train AND Val/Test to synthesize noisy input)
    noise_transform = SpatiallyVaryingGaussianNoise(
        sigma_range=(aug_cfg['sigma_min'], aug_cfg['sigma_max']),
        grid_size=aug_cfg['noise_grid_size'],
        multiplier_range=(aug_cfg.get('noise_multiplier_min', 0.5), aug_cfg.get('noise_multiplier_max', 3.0)),
        target_size=patch_size,
        p=1.0 
    )
    transforms.append(noise_transform)
        
    # 5. Final Standardization
    # We use Clamp to ensure [0, 1] range after augmentations.
    # RescaleIntensity can crash on flat (background) crops.
    transforms.append(tio.Clamp(0, 1, include=['mri', 'gt']))
    
    return tio.Compose(transforms)
