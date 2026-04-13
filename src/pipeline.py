import torch
import numpy as np
import pydicom
from PIL import Image
from models.factory import get_model
import os

class DenoisePipeline:
    """
    Standard Denoising Pipeline for MRI images.
    Handles DICOM/NIFTI preprocessing, model execution, and post-processing.
    """
    def __init__(self, model_name, config, checkpoint_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = get_model(model_name, config['models']).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Pipeline: Loaded model {model_name} from {checkpoint_path}")
        else:
            self.model.eval()
            print(f"Pipeline: Model {model_name} initialized with random weights.")

    @torch.no_grad()
    def denoise_image(self, image_np, sigma=0.05):
        """
        Denoise a single numpy image (H, W).
        Expected range: [0, 1] float32.
        """
        # Prepare 2-channel input: [Image, Sigma Map]
        h, w = image_np.shape
        img_tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        sigma_map = torch.full((1, 1, h, w), sigma).float()
        
        input_tensor = torch.cat([img_tensor, sigma_map], dim=1).to(self.device)
        
        # Inference
        output = self.model(input_tensor)
        output = torch.clamp(output, 0, 1)
        
        return output.squeeze().cpu().numpy()

    def process_dicom(self, dicom_path, output_path=None, sigma=0.05):
        """
        Process a DICOM file and optionally save the denoised version.
        """
        ds = pydicom.dcmread(dicom_path)
        pixel_data = ds.pixel_array.astype(np.float32)
        
        # 16-bit Normalization consistency
        p1, p99 = np.quantile(pixel_data, [0.01, 0.99])
        denom = p99 - p1 if p99 > p1 else 1.0
        norm_img = (pixel_data - p1) / denom
        norm_img = np.clip(norm_img, 0, 1)
        
        # Denoise
        denoised_norm = self.denoise_image(norm_img, sigma=sigma)
        
        # Re-scale back to original uint16 range
        denoised_final = (denoised_norm * denom) + p1
        denoised_final = np.clip(denoised_final, 0, 65535).astype(np.uint16)
        
        if output_path:
            ds.PixelData = denoised_final.tobytes()
            ds.save_as(output_path)
            
        return denoised_final

    def process_folder(self, input_folder, output_folder, sigma=0.05):
        """
        Batch process a folder of DICOMs.
        """
        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(input_folder) if f.endswith(('.dcm', '.DCM'))]
        print(f"Processing {len(files)} files from {input_folder}...")
        
        for f in files:
            self.process_dicom(
                os.path.join(input_folder, f),
                os.path.join(output_folder, f),
                sigma=sigma
            )
        print(f"Batch processing complete. Results in {output_folder}")
