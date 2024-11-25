"""
Paper:      U-Net: Convolutional Networks for Biomedical Image Segmentation
Url:        https://arxiv.org/abs/1505.04597
Create by:  zh320
Date:       2024/11/08
"""

import torch
import torch.nn as nn

from .modules import conv1x1, ConvBNAct, DeConvBNAct


class UNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, base_channel=64, act_type='relu'):
        super().__init__()
        self.down_stage1 = DownsampleBlock(n_channel, base_channel, act_type)
        self.down_stage2 = DownsampleBlock(base_channel, base_channel*2, act_type)
        self.down_stage3 = DownsampleBlock(base_channel*2, base_channel*4, act_type)
        self.down_stage4 = DownsampleBlock(base_channel*4, base_channel*8, act_type)
        self.mid_stage = ConvBlock(base_channel*8, base_channel*16, act_type)

        self.up_stage4 = UpsampleBlock(base_channel*16, base_channel*8, act_type)
        self.up_stage3 = UpsampleBlock(base_channel*8, base_channel*4, act_type)
        self.up_stage2 = UpsampleBlock(base_channel*4, base_channel*2, act_type)
        self.up_stage1 = UpsampleBlock(base_channel*2, base_channel, act_type)
        self.seg_head = conv1x1(base_channel, num_class)

    def forward(self, x):
        x, x1 = self.down_stage1(x)
        x, x2 = self.down_stage2(x)
        x, x3 = self.down_stage3(x)
        x, x4 = self.down_stage4(x)
        x = self.mid_stage(x)

        x = self.up_stage4(x, x4)
        x = self.up_stage3(x, x3)
        x = self.up_stage2(x, x2)
        x = self.up_stage1(x, x1)
        x = self.seg_head(x)

        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, act_type)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        residual = self.conv(x)
        x = self.pool(residual)

        return x, residual


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.up = DeConvBNAct(in_channels, out_channels, act_type=act_type)
        self.conv = ConvBlock(in_channels, out_channels, act_type)

    def forward(self, x, residual):
        x = self.up(x)
        x = torch.cat([x, residual], dim=1)
        x = self.conv(x)

        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__(
        ConvBNAct(in_channels, out_channels, 3, act_type=act_type, inplace=True),
        ConvBNAct(out_channels, out_channels, 3, act_type=act_type, inplace=True)
        )