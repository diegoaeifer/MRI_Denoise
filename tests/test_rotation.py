import torch
import torchio as tio
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add src to path
# Add src to path
# We are running from 'D:\Diego trabalho\Trainer MRI' (root)
# Script is in 'FMImaging_MRI_Denoise/tests/test_rotation.py'
# We need to import from 'FMImaging_MRI_Denoise.src.data.transforms'
# So we need 'D:\Diego trabalho\Trainer MRI' in path, which is implicit if running from there?
# Actually, if I run `python FMImaging_MRI_Denoise/tests/...`, CWD is in path.
# But `FMImaging_MRI_Denoise` is a folder.
# So `import FMImaging_MRI_Denoise.src...` should work if `FMImaging_MRI_Denoise` has `__init__.py`.
# It probably doesn't.
# Let's add the inner folder to path.

sys.path.append(os.path.join(os.getcwd(), 'FMImaging_MRI_Denoise'))
from src.data.transforms import get_transforms

def test_rotation():
    out_dir = "tests/rotation_test_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Create a dummy subject (Square with a line to see rotation easily)
    # H, W = 128, 128
    # data = torch.zeros(1, H, W, 1)
    # data[:, 30:90, 60:68, :] = 1.0 # Vertical bar
    # data[:, 30:40, 30:90, :] = 1.0 # Horizontal bar top
    
    # Better: Use a real image if available, or just the synthetic one.
    # Let's try to load one from the 'background_samples.png' folder if possible, or just synthetic.
    # Synthetic is robust.
    
    H, W = 256, 256
    data = torch.zeros(1, 1, H, W)
    # Draw an 'L' shape
    data[0, 0, 50:200, 50:80] = 1.0 # Vertical
    data[0, 0, 170:200, 50:150] = 1.0 # Horizontal
    
    # TorchIO expects (C, W, H, D) or (C, spatial...)
    # 2D case: (C, H, W, 1)
    data_tio = data.permute(0, 2, 3, 1) # (1, H, W, D=1)
    
    subject = tio.Subject(
        mri=tio.ScalarImage(tensor=data_tio)
    )
    
    # 2. Config for transforms
    config = {
        'patch_size': [256, 256],
        'augmentation': {
            'flip_prob': 0.0, # Disable flip to check rotation only
            'rotate_prob': 1.0, # Force rotation
            'blur_prob': 0.0,
            'motion_prob': 0.0,
            'bias_field_prob': 0.0,
            'anisotropy_prob': 0.0,
            'sigma_min': 0.0, 'sigma_max': 0.0, 'noise_grid_size': 4 # Disable noise
        }
    }
    
    transform = get_transforms('train', config)
    
    # 3. Apply multiple times and plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    print("Generating rotation samples...")
    for i in range(10):
        transformed = transform(subject)
        # Extract data: (C, H, W, D) -> (H, W)
        img_data = transformed['mri'].data.squeeze().numpy()
        
        axes[i].imshow(img_data, cmap='gray')
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rotation_check.png"))
    print(f"Saved rotation_check.png to {out_dir}")

if __name__ == "__main__":
    test_rotation()
