import torch
import torch.nn as nn
from .auxiliary import CharbonnierLoss, MCSURELoss, VGGPerceptualLoss
from monai.losses import SSIMLoss

class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2) + 1e-8
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        # We want to maximize PSNR, so minimize negative PSNR
        return -psnr

class CompositeLoss(nn.Module):
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary containing weights and settings.
        """
        super(CompositeLoss, self).__init__()
        self.weights = config['weights']
        self.aux_cfg = config['auxiliary']
        
        # Main Metrics
        self.l1 = nn.L1Loss()
        # Monai SSIM Loss minimizes 1 - SSIM, which is what we want.
        self.ssim = SSIMLoss(spatial_dims=2, data_range=1.0) 
        self.psnr = PSNRLoss()
        
        # Aux
        self.charbonnier = CharbonnierLoss(eps=self.aux_cfg.get('charbonnier_eps', 1e-3))
        self.vgg = VGGPerceptualLoss(layer_name=self.aux_cfg.get('vgg_layer', 'relu3_3'))
        
        # SURE is special, dealt with in forward with explicit call if needed
        self.sure = MCSURELoss(eps=1e-4)

    def forward(self, pred, target, model=None, input_tensor=None):
        """
        Compute weighted composite loss.
        """
        L_l1 = self.l1(pred, target)
        L_ssim = self.ssim(pred, target)
        L_psnr = self.psnr(pred, target)
        
        # Base Composite
        total_loss = (
            self.weights.get('l1', 1.0) * L_l1 +
            self.weights.get('ssim', 1.0) * L_ssim +
            self.weights.get('psnr', 0.1) * L_psnr
        )
        
        # Optional Aux terms (if weights > 0 in config, but current config only lists weights for main 3)
        # We'll just compute them for logging or if weights added later.
        # But if the user wants Charbonnier instead of L1, they can swap weights.
        
        # Let's support arbitrary weights if present
        if self.weights.get('charbonnier', 0.0) > 0:
            total_loss += self.weights['charbonnier'] * self.charbonnier(pred, target)
            
        if self.weights.get('vgg', 0.0) > 0:
            total_loss += self.weights['vgg'] * self.vgg(pred, target)
            
        if self.weights.get('sure', 0.0) > 0 and model is not None and input_tensor is not None:
            # Extract sigma map from input (channel 1)
            sigma_map = input_tensor[:, 1:2, :, :]
            L_sure = self.sure(model, input_tensor, pred, sigma_map)
            total_loss += self.weights['sure'] * L_sure
            
        return total_loss, {
            'l1': L_l1,
            'ssim': L_ssim,
            'psnr': -L_psnr, # Log positive PSNR
        }
