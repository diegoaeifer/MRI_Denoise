import torch
import torch.nn as nn
import torch.nn.functional as F

class FFDNet(nn.Module):
    """
    Fast and Flexible Denoising Network (FFDNet).
    Operates on downsampled sub-images to expand the receptive field and speed up computation.
    """
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        Args:
            in_nc: Number of input channels. Default is 2 (image + sigma map).
                   Note: The downsampling process will expand these channels.
                         Specifically, a factor of 2 downsampling will result in
                         4 sub-images per channel.
            out_nc: Number of output channels (e.g., 1 for grayscale).
            nc: Number of feature maps in intermediate layers.
            nb: Number of convolution blocks.
            act_mode: Activation mode 'R' for ReLU.
        """
        super(FFDNet, self).__init__()

        # We downsample by a factor of 2, so channels are multiplied by 2^2 = 4
        # But we only downsample the image, and just append the downsampled noise map,
        # or we downsample both. If we downsample both:
        # 1 image channel -> 4 channels
        # 1 noise map channel -> 4 channels (or we can just use 1 if it's constant, but let's downsample it too)
        # So in_nc (2) -> 2 * 4 = 8 channels

        # Actually, let's follow the standard FFDNet where input to the CNN is
        # the downsampled image (4 channels) + the noise map (1 channel).
        # We'll adapt it here for our pipeline which passes (img, sigma_map) together.

        # Standard FFDNet: Downsampled image (c*4) + noise map (c).
        # Our input: x is (B, 2, H, W). x[:, 0:1] is img, x[:, 1:2] is noise map.
        # After PixelUnshuffle(2) on img: (B, 4, H/2, W/2)
        # We can just downsample the noise map to (B, 1, H/2, W/2) using avg pooling or just taking a slice.
        # So total input channels to CNN = 4 + 1 = 5.

        self.cnn_in_nc = 5 # 4 for un-shuffled image, 1 for downsampled noise map
        self.out_nc = out_nc

        m_head = [
            nn.Conv2d(self.cnn_in_nc, nc, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]

        m_body = []
        for i in range(nb - 2):
            m_body.append(nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=False))
            m_body.append(nn.BatchNorm2d(nc))
            m_body.append(nn.ReLU(inplace=True))

        m_tail = [nn.Conv2d(nc, out_nc * 4, kernel_size=3, padding=1, bias=False)]

        self.model = nn.Sequential(*m_head, *m_body, *m_tail)

    def forward(self, x):
        """
        x: [B, 2, H, W] where x[:, 0:1] is image, x[:, 1:2] is noise map
        """
        # 1. Separate image and noise map
        img = x[:, 0:1, :, :]
        sigma_map = x[:, 1:2, :, :]

        # 2. Pixel unshuffle (downsample) the image
        # (B, 1, H, W) -> (B, 4, H/2, W/2)
        # We can use PyTorch's PixelUnshuffle (requires PyTorch >= 1.8)
        img_down = F.pixel_unshuffle(img, 2)

        # 3. Downsample noise map (it's usually constant across spatial dims, so avg pool is fine)
        sigma_down = F.avg_pool2d(sigma_map, kernel_size=2, stride=2)

        # 4. Concatenate
        # (B, 4, H/2, W/2) and (B, 1, H/2, W/2) -> (B, 5, H/2, W/2)
        h = torch.cat([img_down, sigma_down], dim=1)

        # 5. Denoise in low-res space
        h = self.model(h) # Outputs (B, 4, H/2, W/2)

        # 6. Pixel shuffle (upsample)
        out = F.pixel_shuffle(h, 2) # (B, 1, H, W)

        return out
