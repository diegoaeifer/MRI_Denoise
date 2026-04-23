import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SeparBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.branch_x = nn.Sequential(
            SeparableConv3d(in_channels, filters, kernel_size=3, padding='same'),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            SeparableConv3d(filters, filters, kernel_size=3, padding='same'),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        self.branch_y = nn.Sequential(
            SeparableConv3d(in_channels, filters, kernel_size=1, padding='same'),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        
    def forward(self, x):
        return self.branch_x(x) + self.branch_y(x)

class Conv3d_BN(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, filters, kernel_size, dilation=dilation, padding='same', bias=False)
        self.bn = nn.BatchNorm3d(filters)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, dilation=1, with_conv_shortcut=False):
        super().__init__()
        self.with_conv_shortcut = with_conv_shortcut
        self.conv1 = Conv3d_BN(in_channels, filters, kernel_size, dilation)
        self.conv2 = Conv3d_BN(filters, filters, kernel_size, dilation)
        
        if with_conv_shortcut:
            self.shortcut = Conv3d_BN(in_channels, filters, kernel_size, dilation)
            self.dropout = nn.Dropout3d(0.2)
        
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.with_conv_shortcut:
            out = self.dropout(out)
            return out + self.shortcut(x)
        else:
            return out + x

class RicianNet3D(nn.Module):
    def __init__(self, in_channels, out_channels=1, base_filters=16):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        
        self.blocks_x = nn.ModuleList()
        self.blocks_y = nn.ModuleList()
        
        dilations = [1]*6 + [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        
        current_in = in_channels
        for i in range(18):
            dil = dilations[i]
            # y_n = Separ(y_{n-1})
            self.blocks_y.append(SeparBlock(current_in, base_filters))
            # x_n = IdentityBlock(x_{n-1})
            self.blocks_x.append(IdentityBlock(current_in, base_filters, kernel_size=3, dilation=dil, with_conv_shortcut=True))
            current_in = base_filters
            
        self.final_conv = nn.Sequential(
            nn.Conv3d(base_filters, out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        
    def forward(self, x):
        # x is (B, in_channels, D, H, W)
        image = x[:, 0:1, ...] # The image to be subtracted/residualized
        
        x_out, y_out = x, x
        for i in range(18):
            y_out_new = self.blocks_y[i](y_out)
            x_out_new = self.blocks_x[i](x_out)
            x_out = x_out_new + y_out_new
            y_out = y_out_new
            
        residual = self.final_conv(x_out)
        out = image - residual
        return out
