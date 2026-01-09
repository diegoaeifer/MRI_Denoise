import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DoubleConv(512, 1024 // factor)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(1024 // factor, 512 // factor, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024 // factor, 512 // factor) # 512 + 512 -> 512
        
        self.up2 = nn.ConvTranspose2d(512 // factor, 256 // factor, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512 // factor, 256 // factor)

        self.up3 = nn.ConvTranspose2d(256 // factor, 128 // factor, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256 // factor, 128 // factor)

        self.up4 = nn.ConvTranspose2d(128 // factor, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.maxpool(x1))
        x3 = self.down2(self.maxpool(x2))
        x4 = self.down3(self.maxpool(x3))
        x5 = self.down4(self.maxpool(x4))
        
        u1 = self.up1(x5)
        # Pad if necessary
        diffY = x4.size()[2] - u1.size()[2]
        diffX = x4.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, u1], dim=1)
        x = self.conv_up1(x)
        
        u2 = self.up2(x)
        diffY = x3.size()[2] - u2.size()[2]
        diffX = x3.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, u2], dim=1)
        x = self.conv_up2(x)
        
        u3 = self.up3(x)
        diffY = x2.size()[2] - u3.size()[2]
        diffX = x2.size()[3] - u3.size()[3]
        u3 = F.pad(u3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, u3], dim=1)
        x = self.conv_up3(x)
        
        u4 = self.up4(x)
        diffY = x1.size()[2] - u4.size()[2]
        diffX = x1.size()[3] - u4.size()[3]
        u4 = F.pad(u4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, u4], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits
