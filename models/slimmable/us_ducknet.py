"""
Paper:      Using DUCK-Net for Polyp Image Segmentation
Url:        https://arxiv.org/abs/2311.02239
Create by:  zh320
Date:       2024/11/08
"""
from typing import List

import torch.nn as nn
import torch.nn.functional as F

from models.slimmable.slimmable_ops import USBatchNorm2d
from .universally_slimmable_modules import us_conv1x1, USConvBNAct, Activation
from models.model_registry import register_model, slimmable_models


@register_model(slimmable_models)
class USDuckNet(nn.Module):
    def __init__(self, slim_width_mult_list: List[float], num_class=1, n_channel=3, base_channel=34, act_type='relu'):
        super().__init__()

        self.down_stage1 = USDownsampleBlock(n_channel, base_channel * 2, act_type, fuse_channels=base_channel,
                                             width_mult_list=slim_width_mult_list)
        self.down_stage2 = USDownsampleBlock(base_channel * 2, base_channel * 4, act_type,
                                             width_mult_list=slim_width_mult_list)
        self.down_stage3 = USDownsampleBlock(base_channel * 4, base_channel * 8, act_type,
                                             width_mult_list=slim_width_mult_list)
        self.down_stage4 = USDownsampleBlock(base_channel * 8, base_channel * 16, act_type,
                                             width_mult_list=slim_width_mult_list)
        self.down_stage5 = USDownsampleBlock(base_channel * 16, base_channel * 32, act_type,
                                             width_mult_list=slim_width_mult_list)
        self.mid_stage = nn.Sequential(
            USResidualBlock(base_channel * 32, base_channel * 32, act_type, width_mult_list=slim_width_mult_list),
            USResidualBlock(base_channel * 32, base_channel * 32, act_type, width_mult_list=slim_width_mult_list),
            USResidualBlock(base_channel * 32, base_channel * 16, act_type, width_mult_list=slim_width_mult_list),
            USResidualBlock(base_channel * 16, base_channel * 16, act_type, width_mult_list=slim_width_mult_list),
        )

        self.up_stage5 = USUpsampleBlock(base_channel * 16, base_channel * 8, act_type, width_mult_list=slim_width_mult_list)
        self.up_stage4 = USUpsampleBlock(base_channel * 8, base_channel * 4, act_type, width_mult_list=slim_width_mult_list)
        self.up_stage3 = USUpsampleBlock(base_channel * 4, base_channel * 2, act_type, width_mult_list=slim_width_mult_list)
        self.up_stage2 = USUpsampleBlock(base_channel * 2, base_channel, act_type, width_mult_list=slim_width_mult_list)
        self.up_stage1 = USUpsampleBlock(base_channel, base_channel, act_type, width_mult_list=slim_width_mult_list)
        self.seg_head = us_conv1x1(base_channel, num_class)

    def forward(self, x):
        x1, x1_skip, x = self.down_stage1(x)
        x2, x2_skip, x = self.down_stage2(x1 + x, x)
        x3, x3_skip, x = self.down_stage3(x2 + x, x)
        x4, x4_skip, x = self.down_stage4(x3 + x, x)
        x5, x5_skip, x = self.down_stage5(x4 + x, x)
        x = self.mid_stage(x5 + x)

        x = self.up_stage5(x, x5_skip)
        x = self.up_stage4(x, x4_skip)
        x = self.up_stage3(x, x3_skip)
        x = self.up_stage2(x, x2_skip)
        x = self.up_stage1(x, x1_skip)
        x = self.seg_head(x)

        return x


class USDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, fuse_channels=None,
                 width_mult_list: list[float] | None = None):
        super().__init__()
        fuse_channels = in_channels if fuse_channels is None else fuse_channels
        self.duck = USDUCK(in_channels, fuse_channels, act_type, width_mult_list=width_mult_list)
        self.conv1 = USConvBNAct(fuse_channels, out_channels, 3, 2, act_type=act_type, width_mult_list=width_mult_list)
        self.conv2 = USConvBNAct(in_channels, out_channels, 2, 2, act_type=act_type, width_mult_list=width_mult_list)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = self.conv2(x1)
        else:
            x2 = self.conv2(x2)

        skip = self.duck(x1)
        x1 = self.conv1(skip)

        return x1, skip, x2


class USUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, width_mult_list: list[float] | None = None):
        super().__init__()
        self.duck = USDUCK(in_channels, out_channels, act_type, width_mult_list=width_mult_list)

    def forward(self, x, residual):
        size = residual.size()[2:]
        x = F.interpolate(x, size, mode='nearest')

        x += residual
        x = self.duck(x)

        return x


class USResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, width_mult_list: list[float] | None = None):
        super().__init__()
        self.upper_branch = us_conv1x1(in_channels, out_channels)
        self.lower_branch = nn.Sequential(
            USConvBNAct(in_channels, out_channels, 3, act_type=act_type, width_mult_list=width_mult_list),
            USConvBNAct(out_channels, out_channels, 3, act_type=act_type, width_mult_list=width_mult_list),
        )
        self.bn = nn.Sequential(
            USBatchNorm2d(out_channels, width_mult_list=width_mult_list),
            Activation(act_type)
        )

    def forward(self, x):
        x_up = self.upper_branch(x)
        x_low = self.lower_branch(x)

        x = x_up + x_low
        x = self.bn(x)

        return x


class USDUCK(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, filter_size=6 + 1,
                 width_mult_list: list[float] | None = None):
        '''
        Here I change the filter size of separated block to be odd number.
        '''
        super().__init__()
        self.in_bn = nn.Sequential(
            USBatchNorm2d(in_channels, width_mult_list=width_mult_list),
            Activation(act_type)
        )
        self.branch1 = USWidescopeBlock(in_channels, out_channels, act_type, width_mult_list=width_mult_list)
        self.branch2 = USMidscopeBlock(in_channels, out_channels, act_type, width_mult_list=width_mult_list)
        self.branch3 = USResidualBlock(in_channels, out_channels, act_type, width_mult_list=width_mult_list)
        self.branch4 = nn.Sequential(
            USResidualBlock(in_channels, out_channels, act_type, width_mult_list=width_mult_list),
            USResidualBlock(out_channels, out_channels, act_type, width_mult_list=width_mult_list),
        )
        self.branch5 = nn.Sequential(
            USResidualBlock(in_channels, out_channels, act_type, width_mult_list=width_mult_list),
            USResidualBlock(out_channels, out_channels, act_type, width_mult_list=width_mult_list),
            USResidualBlock(out_channels, out_channels, act_type, width_mult_list=width_mult_list),
        )
        self.branch6 = USSeparatedBlock(in_channels, out_channels, filter_size, act_type,
                                        width_mult_list=width_mult_list)
        self.out_bn = nn.Sequential(
            USBatchNorm2d(out_channels, width_mult_list=width_mult_list),
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


class USMidscopeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type, width_mult_list: list[float] | None = None):
        super().__init__(
            USConvBNAct(in_channels, out_channels, 3, act_type=act_type, width_mult_list=width_mult_list),
            USConvBNAct(out_channels, out_channels, 3, dilation=2, act_type=act_type, width_mult_list=width_mult_list)
        )


class USWidescopeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type, width_mult_list: list[float] | None = None):
        super().__init__(
            USConvBNAct(in_channels, out_channels, 3, act_type=act_type, width_mult_list=width_mult_list),
            USConvBNAct(out_channels, out_channels, 3, dilation=2, act_type=act_type, width_mult_list=width_mult_list),
            USConvBNAct(out_channels, out_channels, 3, dilation=3, act_type=act_type, width_mult_list=width_mult_list),
        )


class USSeparatedBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, filter_size, act_type, width_mult_list: list[float] | None = None):
        super().__init__(
            USConvBNAct(in_channels, out_channels, (1, filter_size), act_type=act_type,
                        width_mult_list=width_mult_list),
            USConvBNAct(out_channels, out_channels, (filter_size, 1), act_type=act_type,
                        width_mult_list=width_mult_list),
        )
