import torch
import numpy as np
from monai.metrics import PSNRMetric, SSIMMetric

def calculate_roi_snr(image, box_size=20):
    """
    Calculates SNR in a central ROI.
    SNR_dB = 20 * log10(mu / sigma)
    
    Args:
        image (torch.Tensor or np.ndarray): Image (H, W).
        box_size (int): Size of the central square ROI.
    
    Returns:
        float: SNR in dB.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        
    H, W = image.shape[-2:]
    
    cy, cx = H // 2, W // 2
    r = box_size // 2
    
    roi = image[..., cy-r:cy+r, cx-r:cx+r]
    
    mu = np.mean(roi)
    sigma = np.std(roi)
    
    if sigma == 0:
        return float('inf')
        
    snr = 20 * np.log10(abs(mu / sigma) + 1e-8)
    return snr

class MetricsTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.psnr = []
        self.ssim = []
        self.snr = []
        self.losses = []
        
    def update(self, psnr, ssim, snr, loss):
        self.psnr.append(psnr)
        self.ssim.append(ssim)
        self.snr.append(snr)
        self.losses.append(loss)
        
    def avg(self):
        return {
            'PSNR': np.mean(self.psnr) if self.psnr else 0,
            'SSIM': np.mean(self.ssim) if self.ssim else 0,
            'SNR': np.mean(self.snr) if self.snr else 0,
            'Loss': np.mean(self.losses) if self.losses else 0
        }
