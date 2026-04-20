import torch
import torch.nn as nn
from .auxiliary import CharbonnierLoss, MCSURELoss, VGGPerceptualLoss
from monai.losses import SSIMLoss
import piq
from piq import LPIPS, DISTS

class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2) + 1e-8
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        # We want to maximize PSNR, so minimize negative PSNR
        return -psnr


class EPILoss(nn.Module):
    """
    Edge Preservation Index (EPI) Loss.
    Computes EPI based on Sobel gradients.
    Returns 1 - EPI, so minimizing the loss maximizes EPI.
    """
    def __init__(self):
        super(EPILoss, self).__init__()
        # Define Sobel kernels for Gx and Gy
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _imgradientxy(self, img):
        # img shape: (B, C, H, W)
        # Apply padding to keep spatial dimensions same
        gx = torch.nn.functional.conv2d(img, self.sobel_x.expand(img.size(1), 1, 3, 3), padding=1, groups=img.size(1))
        gy = torch.nn.functional.conv2d(img, self.sobel_y.expand(img.size(1), 1, 3, 3), padding=1, groups=img.size(1))
        return gx, gy

    def forward(self, pred, target):
        # pred: denoised, target: clean
        Gx1, Gy1 = self._imgradientxy(target)
        Gx2, Gy2 = self._imgradientxy(pred)

        grad1 = torch.sqrt(Gx1**2 + Gy1**2 + 1e-8)
        grad2 = torch.sqrt(Gx2**2 + Gy2**2 + 1e-8)

        # Correlation (sum over spatial dims H, W)
        num = torch.sum(grad1 * grad2, dim=[-2, -1])
        den = torch.sqrt(torch.sum(grad1**2, dim=[-2, -1]) * torch.sum(grad2**2, dim=[-2, -1]))

        # e = num / (den + 1e-8)
        # Average over channels and batch
        e = torch.mean(num / (den + 1e-8))

        return 1.0 - e

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
        
        is_3d_loss = config.get('data', {}).get('is_3d', False)
        spatial_dims = 3 if is_3d_loss else 2
        
        self.ssim = SSIMLoss(spatial_dims=spatial_dims, data_range=1.0) 
        self.ms_ssim = piq.MultiScaleSSIMLoss(data_range=1.0, scale_weights=torch.tensor([0.0448, 0.2856, 0.3001, 0.2363]))
        self.psnr = PSNRLoss()
        self.haarpsi = piq.HaarPSILoss(data_range=1.0, c=5.0, alpha=5.8)
        self.epi = EPILoss()
        
        # Aux
        self.charbonnier = CharbonnierLoss(eps=self.aux_cfg.get('charbonnier_eps', 1e-3))

        # Only initialize VGG if it is going to be used
        if self.weights.get('vgg', 0.0) > 0:
            self.vgg = VGGPerceptualLoss(layer_name=self.aux_cfg.get('vgg_layer', 'relu3_3'))
        
        # SURE is special, dealt with in forward with explicit call if needed
        self.sure = MCSURELoss(eps=1e-4)

        # piq losses (LPIPS/DISTS from jules branch)
        if self.weights.get('lpips', 0.0) > 0:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self.lpips_vgg = LPIPS(replace_pooling=False)
        if self.weights.get('dists', 0.0) > 0:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self.dists = DISTS()

    def forward(self, pred, target, model=None, input_tensor=None):
        """
        Compute weighted composite loss.
        """
        # PIQ metrics and MONAI SSIM need 4D input
        p_piq, t_piq = pred, target
        if pred.ndim == 5:
            b, c, h, w, d = pred.shape
            p_piq = pred.permute(0, 4, 1, 2, 3).reshape(b*d, c, h, w)
            t_piq = target.permute(0, 4, 1, 2, 3).reshape(b*d, c, h, w)

        L_l1 = self.l1(pred, target) # L1 doesn't care about shapes
        L_ssim = self.ssim(p_piq, t_piq)
        
        p_ssim, t_ssim = p_piq, t_piq
        if p_ssim.shape[-1] < 81 or p_ssim.shape[-2] < 81:
            pad_h = max(0, 81 - p_ssim.shape[-2])
            pad_w = max(0, 81 - p_ssim.shape[-1])
            import torch.nn.functional as F
            p_ssim = F.pad(p_ssim, (0, pad_w, 0, pad_h), mode='reflect')
            t_ssim = F.pad(t_ssim, (0, pad_w, 0, pad_h), mode='reflect')

        try:
            L_ms_ssim = self.ms_ssim(torch.clamp(p_ssim, 0, 1), t_ssim)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"MS_SSIM Error: {e}. shape: {p_ssim.shape}")
            L_ms_ssim = torch.tensor(1.0, device=pred.device)

        L_psnr = self.psnr(pred, target)
        
        # Prevent HaarPSI crashing on constant batches
        try:
            L_haarpsi = self.haarpsi(torch.clamp(p_piq, 0, 1), t_piq)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"HAARpsi Error: {e}. shape: {p_piq.shape}")
            L_haarpsi = torch.tensor(1.0, device=pred.device)
        
        L_epi = self.epi(p_piq, t_piq)

        # Base Composite
        total_loss = (
            self.weights.get('l1', 1.0) * L_l1 +
            self.weights.get('ssim', 1.0) * L_ssim +
            self.weights.get('ms_ssim', 0.0) * L_ms_ssim +
            self.weights.get('psnr', 0.1) * L_psnr +
            self.weights.get('haarpsi', 0.0) * L_haarpsi +
            self.weights.get('epi', 0.0) * L_epi
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
            
        # Perceptual metrics (LPIPS/DISTS)
        L_lpips = torch.tensor(0.0, device=pred.device)
        L_dists = torch.tensor(0.0, device=pred.device)
        
        if self.weights.get('lpips', 0.0) > 0 or self.weights.get('dists', 0.0) > 0:
            # These require 3-channel input in [0, 1]
            pred_3c = torch.clamp(p_piq, 0, 1).repeat(1, 3, 1, 1)
            target_3c = torch.clamp(t_piq, 0, 1).repeat(1, 3, 1, 1)
            
            if self.weights.get('lpips', 0.0) > 0:
                L_lpips = self.lpips_vgg(pred_3c, target_3c)
                total_loss += self.weights['lpips'] * L_lpips

            if self.weights.get('dists', 0.0) > 0:
                L_dists = self.dists(pred_3c, target_3c)
                total_loss += self.weights['dists'] * L_dists
            
        return total_loss, {
            'l1': L_l1,
            'ssim': L_ssim,
            'ms_ssim': L_ms_ssim,
            'psnr': -L_psnr, # Log positive PSNR
            'haarpsi': L_haarpsi,
            'epi': 1.0 - L_epi, # Return actual EPI metric value (not loss)
            'lpips': L_lpips,
            'dists': L_dists,
        }
