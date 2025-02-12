"""
Paper:      Road Extraction by Deep Residual U-Net
Url:        https://arxiv.org/abs/1711.10684
Create by:  zh320
Date:       2025/01/05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct
from .model_registry import register_model


@register_model()
class ResUNet(nn.Module):
    def __init__(self, num_class, n_channel=3, base_channel=64, act_type='relu'):
        super().__init__()
        self.encoding1 = ResBlock(n_channel, base_channel, 1, act_type)
        self.encoding2 = ResBlock(base_channel, base_channel*2, 2, act_type)
        self.encoding3 = ResBlock(base_channel*2, base_channel*4, 2, act_type)
        self.bridge = ResBlock(base_channel*4, base_channel*8, 2, act_type)
        self.decoding3 = ResBlock(base_channel*(8+4), base_channel*4, 1, act_type)
        self.decoding2 = ResBlock(base_channel*(4+2), base_channel*2, 1, act_type)
        self.decoding1 = ResBlock(base_channel*(2+1), base_channel, 1, act_type)
        self.seg_head = conv1x1(base_channel, num_class)

    def forward(self, x):
        x1 = self.encoding1(x)
        x2 = self.encoding2(x1)
        x3 = self.encoding3(x2)

        x = self.bridge(x3)

        x = F.interpolate(x, x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.decoding3(x)

        x = F.interpolate(x, x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.decoding2(x)

        x = F.interpolate(x, x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.decoding1(x)

        x = self.seg_head(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        self.conv = nn.Sequential(
                        ConvBNAct(in_channels, out_channels, stride=stride, act_type=act_type),
                        ConvBNAct(out_channels, out_channels, act_type=act_type)
                    )

        self.has_skip_conv = in_channels != out_channels or stride != 1
        if self.has_skip_conv:
            self.conv_skip = conv1x1(in_channels, out_channels, stride=stride)

    def forward(self, x):
        residual = x
        if self.has_skip_conv:
            residual = self.conv_skip(residual)

        x = self.conv(x)
        x += residual

        return x