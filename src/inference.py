import torch
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import MRI_DICOM_Dataset
from models.factory import get_model
from utils.metrics import calculate_roi_snr
from monai.metrics import compute_psnr, compute_ssim

def run_inference(model_name='drunet'):
    # Load Configs
    root_conf = "FMImaging_MRI_Denoise/configs"
    with open(os.path.join(root_conf, "config_train.yaml")) as f: c_train = yaml.safe_load(f)
    with open(os.path.join(root_conf, "config_data.yaml")) as f: c_data = yaml.safe_load(f)
    with open(os.path.join(root_conf, "config_model.yaml")) as f: c_model = yaml.safe_load(f)
    config = {**c_train, **c_data, **c_model}
    
    device = torch.device(f"cuda:{config['training']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = get_model(model_name, config['models']).to(device)
    checkpoint_path = "FMImaging_MRI_Denoise/experiments/checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', '?')}")
    else:
        print("No checkpoint found, using random weights (Verification mode)")
        
    model.eval()
    
    # Load Test Data
    # Assuming splits exists
    splits_path = os.path.join(config['data']['splits_path'], "test_files.json")
    if not os.path.exists(splits_path):
        print("Test split not found.")
        return

    import json
    with open(splits_path, 'r') as f:
        test_files = json.load(f)
        
    test_ds = MRI_DICOM_Dataset(test_files, mode='test', config=config['data'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    output_dir = "FMImaging_MRI_Denoise/experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, batch in enumerate(test_loader):
        if i >= 10: break # Visualize just 10 examples
        if batch is None: continue
        
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        with torch.no_grad():
            preds = model(inputs)
            
        # Metrics
        # psnr = compute_psnr(preds, targets, data_range=1.0)
        # ssim = compute_ssim(preds, targets, data_range=1.0)
        # Using simple local calc for display
        val_mse = torch.mean((preds - targets)**2)
        val_psnr = 20 * torch.log10(1.0 / torch.sqrt(val_mse + 1e-8))
        
        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        noisy_img = inputs[0, 0, :, :].cpu().numpy()
        clean_img = targets[0, 0, :, :].cpu().numpy()
        denoised_img = preds[0, 0, :, :].cpu().numpy()
        
        axs[0].imshow(clean_img, cmap='gray')
        axs[0].set_title("Original (Clean)")
        axs[0].axis('off')
        
        axs[1].imshow(noisy_img, cmap='gray')
        axs[1].set_title("Noisy Input")
        axs[1].axis('off')
        
        axs[2].imshow(denoised_img, cmap='gray')
        axs[2].set_title(f"Denoised\nPSNR: {val_psnr:.2f} dB")
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"result_{i}.png"))
        plt.close()
        
    print(f"Inference complete. Results saved to {output_dir}")

if __name__ == "__main__":
    run_inference()
