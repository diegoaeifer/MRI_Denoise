import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, bias=bias, padding_mode='reflect')
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, bias=bias, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out

class DRUNet(nn.Module):
    # cleaned legacy code

    def __init__(self, in_channels=2, out_channels=1, base_channels=64, num_res_blocks=4):
        super(DRUNet, self).__init__()
        
        bc = base_channels
        
        self.head = nn.Conv2d(in_channels, bc, 3, 1, 1, bias=True)
        
        # Encoder
        self.down1 = self._make_layer(bc, num_res_blocks)
        self.down_conv1 = nn.Conv2d(bc, bc*2, 3, 2, 1)
        
        self.down2 = self._make_layer(bc*2, num_res_blocks)
        self.down_conv2 = nn.Conv2d(bc*2, bc*4, 3, 2, 1)
        
        self.down3 = self._make_layer(bc*4, num_res_blocks)
        self.down_conv3 = nn.Conv2d(bc*4, bc*8, 3, 2, 1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResBlock(bc*8, dilation=2) for _ in range(num_res_blocks)]
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(bc*8, bc*4, 2, 2)
        self.reduce3 = nn.Conv2d(bc*8, bc*4, 1) # 256+256 -> 256
        self.recon3 = self._make_layer(bc*4, num_res_blocks)
        
        self.up2 = nn.ConvTranspose2d(bc*4, bc*2, 2, 2)
        self.reduce2 = nn.Conv2d(bc*4, bc*2, 1) # 128+128 -> 128
        self.recon2 = self._make_layer(bc*2, num_res_blocks)
        
        self.up1 = nn.ConvTranspose2d(bc*2, bc, 2, 2)
        self.reduce1 = nn.Conv2d(bc*2, bc, 1) # 64+64 -> 64
        self.recon1 = self._make_layer(bc, num_res_blocks)
        
        self.tail = nn.Conv2d(bc, out_channels, 3, 1, 1)

    def _make_layer(self, channels, blocks):
        layers = [ResBlock(channels) for _ in range(blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.head(x)
        
        d1 = self.down1(h)
        d2 = self.down2(self.down_conv1(d1))
        d3 = self.down3(self.down_conv2(d2))
        
        neck = self.bottleneck(self.down_conv3(d3))
        
        u3 = self.up3(neck)
        cat3 = torch.cat([u3, d3], dim=1) # 256 + 256
        h3 = self.recon3(self.reduce3(cat3))
        
        u2 = self.up2(h3)
        cat2 = torch.cat([u2, d2], dim=1)
        h2 = self.recon2(self.reduce2(cat2))
        
        u1 = self.up1(h2)
        cat1 = torch.cat([u1, d1], dim=1)
        h1 = self.recon1(self.reduce1(cat1))
        
        return self.tail(h1)
