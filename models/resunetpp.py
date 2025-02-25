"""
Paper:      ResUNet++: An Advanced Architecture for Medical Image Segmentation
Url:        https://arxiv.org/abs/1911.07067
Create by:  zh320
Date:       2025/02/09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, Activation, SEBlock, ASPP
from .model_registry import register_model


@register_model()
class ResUNetPP(nn.Module):
    def __init__(self, num_class, n_channel=3, base_channel=32, act_type='relu'):
        super().__init__()
        self.encoding_block1 = EncodingBlock(n_channel, base_channel, act_type)
        self.encoding_block2 = EncodingBlock(base_channel, base_channel*2, act_type)
        self.encoding_block3 = EncodingBlock(base_channel*2, base_channel*4, act_type)
        self.encoding_block4 = EncodingBlock(base_channel*4, base_channel*8, act_type, has_se=False)
        self.aspp = ASPP(base_channel*8, base_channel*8, act_type=act_type)
        self.decoding_block3 = DecodingBlock(base_channel*8, base_channel*4, base_channel*4, act_type)
        self.decoding_block2 = DecodingBlock(base_channel*4, base_channel*2, base_channel*2, act_type)
        self.decoding_block1 = DecodingBlock(base_channel*2, base_channel, base_channel, act_type)
        self.seg_head = nn.Sequential(
                            ASPP(base_channel, base_channel, act_type=act_type),
                            conv1x1(base_channel, num_class)
                        )

    def forward(self, x):
        size = x.size()[2:]

        # Encoding
        x, x1 = self.encoding_block1(x)
        x, x2 = self.encoding_block2(x)
        x, x3 = self.encoding_block3(x)
        x = self.encoding_block4(x)

        # Bridge
        x = self.aspp(x)

        # Decoding
        x = self.decoding_block3(x, x3)
        x = self.decoding_block2(x, x2)
        x = self.decoding_block1(x, x1)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, has_se=True):
        super().__init__()
        self.has_se = has_se
        self.conv = nn.Sequential(
                        ConvBNAct(in_channels, out_channels, stride=2, act_type=act_type),
                        ConvBNAct(out_channels, out_channels, act_type=act_type),
                    )
        self.conv_skip = ConvBNAct(in_channels, out_channels, 1, 2, act_type=act_type)
        if has_se:
            self.se = SEBlock(out_channels, act_type)

    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.conv(x)
        skip += x

        if self.has_se:
            x = self.se(skip)
            return x, skip
        else:
            return skip


class DecodingBlock(nn.Module):
    def __init__(self, in_channels, gate_channels, out_channels, act_type):
        super().__init__()
        self.att = Attention(in_channels, gate_channels, act_type)
        self.conv = nn.Sequential(
                        ConvBNAct(in_channels + gate_channels, out_channels, act_type=act_type),
                        ConvBNAct(out_channels, out_channels, act_type=act_type)
                    )
        self.conv_skip = ConvBNAct(in_channels + gate_channels, out_channels, 1, act_type=act_type)

    def forward(self, x, g):
        size = g.size()[2:]
        x = self.att(x, g)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        x = torch.cat([x, g], dim=1)
        skip = self.conv_skip(x)

        x = self.conv(x)
        x += skip

        return x


class Attention(nn.Module):
    def __init__(self, in_channels, gate_channels, act_type, hid_channels=128):
        super().__init__()
        self.conv_x = ConvBNAct(in_channels, hid_channels, act_type=act_type)
        self.conv_g = ConvBNAct(gate_channels, hid_channels, act_type=act_type)
        self.conv = ConvBNAct(hid_channels, 1, act_type='sigmoid')

    def forward(self, x, g):
        x_conv = self.conv_x(x)
        g = self.conv_g(g)

        size = x.size()[2:]
        g = F.interpolate(g, size, mode='nearest')
        g += x_conv
        g = self.conv(g)
        x = x*g

        return x