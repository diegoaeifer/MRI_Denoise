import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class MCSURELoss(nn.Module):
    """
    Monte Carlo Stein's Unbiased Risk Estimate (SURE) Loss.
    Requires access to the model to perform the second forward pass for divergence estimation.
    """
    def __init__(self, sigma=None, eps=1e-4):
        super(MCSURELoss, self).__init__()
        self.sigma = sigma
        self.eps = eps

    def forward(self, model, noisy_input, predicted_output, sigma_map=None):
        """
        Args:
            model: The neural network.
            noisy_input: The input tensor (B, 2, H, W) including noise map.
            predicted_output: The current prediction (h(y)).
            sigma_map: (B, 1, H, W) - if spatially varying, we need the sigma map. 
                       If sigma is scalar, use that.
        """
        # Parse Sigma
        if sigma_map is not None:
            var = sigma_map ** 2 # Variance map
            sigma_sq = var
        elif self.sigma is not None:
            sigma_sq = self.sigma ** 2
        else:
            raise ValueError("SURE requires a sigma value or map.")

        # Monte Carlo Divergence Estimate
        b = torch.randn_like(noisy_input) # random probe vector (same shape as input)
        # Note: input has 2 channels. b should affect the IMAGE channel only?
        # Perturbation should be on the Noisy Image y.
        # Structure of noisy_input: [NoisyImage, SigmaMap]
        
        # We only perturb the image channel (channel 0)
        img_input = noisy_input[:, 0:1, :, :]
        sigma_chan = noisy_input[:, 1:, :, :]
        
        b_img = torch.randn_like(img_input)
        
        y_epsilon = img_input + self.eps * b_img
        
        # Reconstruct input for second pass
        input_epsilon = torch.cat([y_epsilon, sigma_chan], dim=1)
        
        # Second forward pass
        # We assume model is in training mode, but for SURE gradient we might want 
        # to ensure we don't track gradients for this pass if we only want divergence?
        # Actually, we need gradients relative to model parameters if SURE is minimized.
        
        h_y_epsilon = model(input_epsilon)
        
        # Divergence approx: (b . (h(y+eps) - h(y))) / eps
        # Dot product: sum(b * difference)
        
        diff = h_y_epsilon - predicted_output
        div = torch.sum(b_img * diff) / self.eps
        # Normalize by batch/dim?
        # Formula: 2*sigma^2 * div / d.
        # d = number of pixels.
        
        d = img_input.numel()
        
        # The first term: ||h(y) - y||^2
        # h(y) is predicted_output. y is noisy_input (image part).
        mse_term = torch.sum((predicted_output - img_input) ** 2)
        
        # Weighted divergence if sigma is spatially varying?
        # Formula: avg( (h(y)-y)^2 + 2*sigma^2 * div_term - sigma^2 )
        # If sigma varies: sum( (h(y)-y)^2 + 2*sigma^2 * (b * diff/eps) - sigma^2 )
        
        # Element-wise calculation to support spatial sigma
        term1 = (predicted_output - img_input) ** 2
        term2 = 2 * sigma_sq * (b_img * diff / self.eps)
        term3 = sigma_sq
        
        sure_map = term1 + term2 - term3
        return torch.mean(sure_map)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_name='relu3_3'):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        # Fix weights
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Extract layers up to target
        features = list(vgg.features)
        
        # Map intuitive names to indices (approximate for relu3_3)
        # VGG16: 
        # 0: conv1_1, 1: relu, 2: conv1_2, 3: relu, 4: pool
        # 5: conv2_1, 6: relu, 7: conv2_2, 8: relu, 9: pool
        # 10: conv3_1, 11: relu, 12: conv3_2, 13: relu, 14: conv3_3, 15: relu (relu3_3)
        
        if layer_name == 'relu3_3':
            self.features = nn.Sequential(*features[:16])
        else:
            self.features = nn.Sequential(*features[:16]) # Default
            
        self.features.eval()
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # x, y are grayscale (B, 1, H, W).
        # Convert to pseudo-RGB
        x_3c = x.repeat(1, 3, 1, 1)
        y_3c = y.repeat(1, 3, 1, 1)
        
        # Normalize
        x_norm = (x_3c - self.mean) / self.std
        y_norm = (y_3c - self.mean) / self.std
        
        x_feat = self.features(x_norm)
        y_feat = self.features(y_norm)
        
        return F.l1_loss(x_feat, y_feat)
