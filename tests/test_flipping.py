import torch
import torchio as tio
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add src to path
# We are running from 'd:\Diego trabalho\Trainer MRI\FMImaging_MRI_Denoise' (context suggests this is root)
sys.path.append(os.path.join(os.getcwd()))
from src.data.transforms import get_transforms

def test_flipping():
    out_dir = "tests/flipping_test_results"
    os.makedirs(out_dir, exist_ok=True)
    
    H, W = 256, 256
    data = torch.zeros(1, 1, H, W)
    # Draw an 'L' shape to see flips easily
    # Vertical bar on the left
    data[0, 0, 50:200, 50:80] = 1.0 
    # Horizontal bar at the BOTTOM
    data[0, 0, 170:200, 50:150] = 1.0 
    
    # TorchIO expects (C, H, W, D=1) for 2D
    data_tio = data.permute(0, 2, 3, 1) # (1, H, W, 1)
    
    subject = tio.Subject(
        mri=tio.ScalarImage(tensor=data_tio)
    )
    
    # 2. Config for transforms
    config = {
        'patch_size': [256, 256],
        'augmentation': {
            'flip_prob': 1.0,   # Force flip
            'rotate_prob': 0.0, # Disable rotation
            'blur_prob': 0.0,
            'motion_prob': 0.0,
            'bias_field_prob': 0.0,
            'anisotropy_prob': 0.0,
            'sigma_min': 0.0, 'sigma_max': 0.0, 'noise_grid_size': 4 
        }
    }
    
    transform = get_transforms('train', config)
    
    # 3. Apply multiple times and plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # First slot is original
    axes[0].imshow(data_tio.squeeze().numpy(), cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    print("Generating flipping samples...")
    for i in range(1, 10):
        transformed = transform(subject)
        # Extract data: (C, H, W, D) -> (H, W)
        img_data = transformed['mri'].data.squeeze().numpy()
        
        axes[i].imshow(img_data, cmap='gray')
        axes[i].set_title(f"Sample {i}")
        axes[i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, "flipping_check.png")
    plt.savefig(save_path)
    print(f"Saved flipping_check.png to {out_dir}")
    return save_path

if __name__ == "__main__":
    test_flipping()
