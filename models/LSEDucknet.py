"""
Paper:      Using DUCK-Net for Polyp Image Segmentation
Url:        https://arxiv.org/abs/2311.02239
Create by:  zh320
Date:       2024/11/08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, Activation
from .model_registry import register_model


@register_model()
class LSEDuckNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, base_channel=34, act_type='relu'):
        super().__init__()
        self.down_stage1 = DownsampleBlock(n_channel, base_channel*2, act_type, fuse_channels=base_channel)
        self.down_stage2 = DownsampleBlock(base_channel*2, base_channel*4, act_type)
        self.down_stage3 = DownsampleBlock(base_channel*4, base_channel*8, act_type)
        self.down_stage4 = DownsampleBlock(base_channel*8, base_channel*16, act_type)
        self.down_stage5 = DownsampleBlock(base_channel*16, base_channel*32, act_type)
        self.mid_stage = nn.Sequential(
                            ResidualBlock(base_channel*32, base_channel*32, act_type),
                            ResidualBlock(base_channel*32, base_channel*32, act_type),
                            ResidualBlock(base_channel*32, base_channel*16, act_type),
                            ResidualBlock(base_channel*16, base_channel*16, act_type),
                        )

        self.up_stage5 = UpsampleBlock(base_channel*16, base_channel*8, act_type)
        self.up_stage4 = UpsampleBlock(base_channel*8, base_channel*4, act_type)
        self.up_stage3 = UpsampleBlock(base_channel*4, base_channel*2, act_type)
        self.up_stage2 = UpsampleBlock(base_channel*2, base_channel, act_type)
        self.up_stage1_lse = UpsampleBlock(base_channel, base_channel, act_type)

        #Freeze all layers
        for name, param in self.named_parameters():
            param.requires_grad = False


        #Unfreeze some layers
        #ADJUST IF NECESSARY #Unfreezing JUST the classification layer.
        self.seg_head_lse = conv1x1(base_channel, num_class)
        self.check_trainable_layers()

        # Unfreeze BatchNorm and LayerNorm layers

        '''
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                for param in m.parameters():
                    param.requires_grad = True'''

        print("Model LSEDuckNet: Please review below to ensure proper architecture and training updates")
        print('----------------------------------------------------------------------------------------')

        #Freeze all Layers

    def check_trainable_layers(self):
        print("\n--- Trainable Layers ---")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name} — trainable")
        print("\n--- Frozen Layers ---")
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"{name} — frozen")
        

    def forward(self, x):
        x1, x1_skip, x = self.down_stage1(x)
        x2, x2_skip, x = self.down_stage2(x1+x, x)
        x3, x3_skip, x = self.down_stage3(x2+x, x)
        x4, x4_skip, x = self.down_stage4(x3+x, x)
        x5, x5_skip, x = self.down_stage5(x4+x, x)
        x = self.mid_stage(x5+x)

        x = self.up_stage5(x, x5_skip)
        x = self.up_stage4(x, x4_skip)
        x = self.up_stage3(x, x3_skip)
        x = self.up_stage2(x, x2_skip)
        x = self.up_stage1_lse(x, x1_skip)
        x = self.seg_head_lse(x)

        return x
        

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, fuse_channels=None):
        super().__init__()
        fuse_channels = in_channels if fuse_channels is None else fuse_channels
        self.duck = DUCK(in_channels, fuse_channels, act_type)
        self.conv1 = ConvBNAct(fuse_channels, out_channels, 3, 2, act_type=act_type)
        self.conv2 = ConvBNAct(in_channels, out_channels, 2, 2, act_type=act_type)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = self.conv2(x1)
        else:
            x2 = self.conv2(x2)

        skip = self.duck(x1)
        x1 = self.conv1(skip)

        return x1, skip, x2


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.duck = DUCK(in_channels, out_channels, act_type)

    def forward(self, x, residual):
        size = residual.size()[2:]
        x = F.interpolate(x, size, mode='nearest')

        x += residual
        x = self.duck(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.upper_branch = conv1x1(in_channels, out_channels)
        self.lower_branch = nn.Sequential(
                                ConvBNAct(in_channels, out_channels, 3, act_type=act_type),
                                ConvBNAct(out_channels, out_channels, 3, act_type=act_type),
                            )
        self.bn = nn.Sequential(
                        nn.BatchNorm2d(out_channels),
                        Activation(act_type)
                    )

    def forward(self, x):
        x_up = self.upper_branch(x)
        x_low = self.lower_branch(x)

        x = x_up + x_low
        x = self.bn(x)

        return x


class DUCK(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, filter_size=6+1):
        '''
        Here I change the filter size of separated block to be odd number.
        '''
        super().__init__()
        self.in_bn = nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        Activation(act_type)
                    )
        self.branch1 = WidescopeBlock(in_channels, out_channels, act_type)
        self.branch2 = MidscopeBlock(in_channels, out_channels, act_type)
        self.branch3 = ResidualBlock(in_channels, out_channels, act_type)
        self.branch4 = nn.Sequential(
                            ResidualBlock(in_channels, out_channels, act_type),
                            ResidualBlock(out_channels, out_channels, act_type),
                        )
        self.branch5 = nn.Sequential(
                            ResidualBlock(in_channels, out_channels, act_type),
                            ResidualBlock(out_channels, out_channels, act_type),
                            ResidualBlock(out_channels, out_channels, act_type),
                        )
        self.branch6 = SeparatedBlock(in_channels, out_channels, filter_size, act_type)
        self.out_bn = nn.Sequential(
                        nn.BatchNorm2d(out_channels),
                        Activation(act_type)
                    )

    def forward(self, x):
        x = self.in_bn(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x6 = self.branch6(x)

        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.out_bn(x)

        return x


class MidscopeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__(
            ConvBNAct(in_channels, out_channels, 3, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, dilation=2, act_type=act_type)
        )


class WidescopeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__(
            ConvBNAct(in_channels, out_channels, 3, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, dilation=2, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, dilation=3, act_type=act_type),
        )


class SeparatedBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, filter_size, act_type):
        super().__init__(
            ConvBNAct(in_channels, out_channels, (1, filter_size), act_type=act_type),
            ConvBNAct(out_channels, out_channels, (filter_size, 1), act_type=act_type),
        )