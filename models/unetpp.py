"""
Paper:      UNet++: A Nested U-Net Architecture for Medical Image Segmentation
Url:        https://arxiv.org/abs/1807.10165
Create by:  zh320
Date:       2025/02/09
"""

import torch
import torch.nn as nn

from .modules import conv1x1, DeConvBNAct
from .unet import ConvBlock
from .model_registry import register_model, aux_models


@register_model(aux_models)
class UNetPP(nn.Module):
    def __init__(self, num_class=1, n_channel=3, base_channel=32, use_aux=False, act_type='relu'):
        super().__init__()
        # Backbone
        self.stage00 = UNetPPBlock(n_channel, base_channel, has_up=False, act_type=act_type)
        self.stage10 = UNetPPBlock(base_channel, base_channel*2, base_channel, act_type=act_type)
        self.stage20 = UNetPPBlock(base_channel*2, base_channel*4, base_channel*2, act_type=act_type)
        self.stage30 = UNetPPBlock(base_channel*4, base_channel*8, base_channel*4, act_type=act_type)
        self.stage40 = UNetPPBlock(base_channel*8, base_channel*16, base_channel*8, has_down=False, act_type=act_type)

        self.stage01 = ConvBlock(base_channel*(1*2), base_channel, act_type)
        self.stage02 = ConvBlock(base_channel*(1*3), base_channel, act_type)
        self.stage03 = ConvBlock(base_channel*(1*4), base_channel, act_type)
        self.stage11 = UNetPPBlock(base_channel*(2*2), base_channel*2, base_channel, has_down=False, act_type=act_type)
        self.stage12 = UNetPPBlock(base_channel*(2*3), base_channel*2, base_channel, has_down=False, act_type=act_type)
        self.stage21 = UNetPPBlock(base_channel*(4*2), base_channel*4, base_channel*2, has_down=False, act_type=act_type)

        self.stage31 = UNetPPBlock(base_channel*(8*2), base_channel*4, base_channel*4, has_down=False, act_type=act_type)
        self.stage22 = UNetPPBlock(base_channel*(4*3), base_channel*2, base_channel*2, has_down=False, act_type=act_type)
        self.stage13 = UNetPPBlock(base_channel*(2*4), base_channel, base_channel, has_down=False, act_type=act_type)
        self.stage04 = ConvBlock(base_channel*(1*5), base_channel, act_type)
        self.seg_head = conv1x1(base_channel, num_class)

        self.use_aux = use_aux
        if use_aux:
            self.aux_heads = nn.ModuleList([conv1x1(base_channel, num_class) for _ in range(3)])

    def forward(self, x, is_training=False):
        # Backbone path
        x00, x = self.stage00(x)
        x10_skip, x10_up, x = self.stage10(x)
        x20_skip, x20_up, x = self.stage20(x)
        x30_skip, x30_up, x = self.stage30(x)
        _, x = self.stage40(x)

        # Stage 3
        x = torch.cat([x, x30_skip], dim=1)
        _, x = self.stage31(x)

        # Stage 2
        x21_in = torch.cat([x30_up, x20_skip], dim=1)
        x21_skip, x21_up = self.stage21(x21_in)

        x = torch.cat([x, x20_skip, x21_skip], dim=1)
        _, x = self.stage22(x)

        # Stage 1
        x11_in = torch.cat([x20_up, x10_skip], dim=1)
        x11_skip, x11_up = self.stage11(x11_in)

        x12_in = torch.cat([x21_up, x10_skip, x11_skip], dim=1)
        x12_skip, x12_up = self.stage12(x12_in)

        x = torch.cat([x, x10_skip, x11_skip, x12_skip], dim=1)
        _, x = self.stage13(x)

        # Stage0
        x01 = torch.cat([x10_up, x00], dim=1)
        x01 = self.stage01(x01)

        x02 = torch.cat([x11_up, x00, x01], dim=1)
        x02 = self.stage02(x02)

        x03 = torch.cat([x12_up, x00, x01, x02], dim=1)
        x03 = self.stage03(x03)

        x = torch.cat([x, x00, x01, x02, x03], dim=1)
        x = self.stage04(x)

        # Seg heads
        x = self.seg_head(x)

        if self.use_aux and is_training:    # a.k.a. deep supervision
            aux_ins = [x01, x02, x03]
            assert len(aux_ins) == len(self.aux_heads)

            auxs = []
            for i, aux_head in enumerate(self.aux_heads):
                aux = aux_head(aux_ins[i])
                auxs.append(aux)

            return x, auxs

        else:
            return x


class UNetPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_channels=None, has_up=True, has_down=True, act_type='relu'):
        super().__init__()
        self.has_up = has_up
        self.has_down = has_down
        self.conv = ConvBlock(in_channels, out_channels, act_type)
        if has_up:
            assert up_channels is not None
            self.up = DeConvBNAct(out_channels, up_channels, act_type=act_type)
        if has_down:
            self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        feats = []

        x = self.conv(x)        # skip path
        feats.append(x)

        if self.has_up:         # upsample path
            x_up = self.up(x)
            feats.append(x_up)

        if self.has_down:       # downsample path
            x_down = self.pool(x)
            feats.append(x_down)

        return feats    # [skip, up, down]