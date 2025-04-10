"""
Paper:      Using DUCK-Net for Polyp Image Segmentation
Url:        https://arxiv.org/abs/2311.02239
Create by:  zh320
Date:       2024/11/08
"""
from typing import List

import torch.nn as nn
import torch.nn.functional as F

from models.slimmable.slimmable_ops import SwitchableBatchNorm2d
from .slimmable_modules import slimmable_conv1x1, SlimmableConvBNAct, Activation
from models.model_registry import register_model, slimmable_models


@register_model(slimmable_models)
class SlimmableDuckNet(nn.Module):
    def __init__(self, slim_width_mult_list: List[float], num_class=1, n_channel=3, base_channel=34, act_type='relu'):
        super().__init__()

        base_channel_list: List[int] = [int(base_channel * multiplier) for multiplier in slim_width_mult_list]


        self.down_stage1 = SlimmableDownsampleBlock(
            slim_width_mult_list,
            [n_channel for _ in range(len(slim_width_mult_list))], 
            base_channel_list*2, 
            act_type, 
            fuse_channels_list=base_channel_list
        )
        self.down_stage2 = SlimmableDownsampleBlock(slim_width_mult_list, base_channel_list*2, base_channel_list*4, act_type)
        self.down_stage3 = SlimmableDownsampleBlock(slim_width_mult_list, base_channel_list*4, base_channel_list*8, act_type)
        self.down_stage4 = SlimmableDownsampleBlock(slim_width_mult_list, base_channel_list*8, base_channel_list*16, act_type)
        self.down_stage5 = SlimmableDownsampleBlock(slim_width_mult_list, base_channel_list*16, base_channel_list*32, act_type)
        self.mid_stage = nn.Sequential(
                            SlimmableResidualBlock(slim_width_mult_list, base_channel_list*32, base_channel_list*32, act_type),
                            SlimmableResidualBlock(slim_width_mult_list, base_channel_list*32, base_channel_list*32, act_type),
                            SlimmableResidualBlock(slim_width_mult_list, base_channel_list*32, base_channel_list*16, act_type),
                            SlimmableResidualBlock(slim_width_mult_list, base_channel_list*16, base_channel_list*16, act_type),
                        )

        self.up_stage5 = SlimmableUpsampleBlock(slim_width_mult_list, base_channel_list*16, base_channel_list*8, act_type)
        self.up_stage4 = SlimmableUpsampleBlock(slim_width_mult_list, base_channel_list*8, base_channel_list*4, act_type)
        self.up_stage3 = SlimmableUpsampleBlock(slim_width_mult_list, base_channel_list*4, base_channel_list*2, act_type)
        self.up_stage2 = SlimmableUpsampleBlock(slim_width_mult_list, base_channel_list*2, base_channel_list, act_type)
        self.up_stage1 = SlimmableUpsampleBlock(slim_width_mult_list, base_channel_list, base_channel_list, act_type)
        self.seg_head = slimmable_conv1x1(slim_width_mult_list, base_channel_list, [num_class for _ in range(len(slim_width_mult_list))])

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
        x = self.up_stage1(x, x1_skip)
        x = self.seg_head(x)

        return x


class SlimmableDownsampleBlock(nn.Module):
    def __init__(self, width_mult_list, in_channels_list, out_channels_list, act_type, fuse_channels_list=None):
        super().__init__()
        fuse_channels_list = in_channels_list if fuse_channels_list is None else fuse_channels_list
        self.duck = SlimmableDUCK(width_mult_list, in_channels_list, fuse_channels_list, act_type)
        self.conv1 = SlimmableConvBNAct(width_mult_list, fuse_channels_list, out_channels_list, 3, 2, act_type=act_type)
        self.conv2 = SlimmableConvBNAct(width_mult_list, in_channels_list, out_channels_list, 2, 2, act_type=act_type)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = self.conv2(x1)
        else:
            x2 = self.conv2(x2)

        skip = self.duck(x1)
        x1 = self.conv1(skip)

        return x1, skip, x2


class SlimmableUpsampleBlock(nn.Module):
    def __init__(self, width_mult_list, in_channels_list, out_channels_list, act_type):
        super().__init__()
        self.duck = SlimmableDUCK(width_mult_list, in_channels_list, out_channels_list, act_type)

    def forward(self, x, residual):
        size = residual.size()[2:]
        x = F.interpolate(x, size, mode='nearest')

        x += residual
        x = self.duck(x)

        return x


class SlimmableResidualBlock(nn.Module):
    def __init__(self, width_mult_list, in_channels_list, out_channels_list, act_type):
        super().__init__()
        self.upper_branch = slimmable_conv1x1(width_mult_list, in_channels_list, out_channels_list)
        self.lower_branch = nn.Sequential(
                                SlimmableConvBNAct(width_mult_list, in_channels_list, out_channels_list, 3, act_type=act_type),
                                SlimmableConvBNAct(width_mult_list, out_channels_list, out_channels_list, 3, act_type=act_type),
                            )
        self.bn = nn.Sequential(
                        SwitchableBatchNorm2d(width_mult_list, out_channels_list),
                        Activation(act_type)
                    )

    def forward(self, x):
        x_up = self.upper_branch(x)
        x_low = self.lower_branch(x)

        x = x_up + x_low
        x = self.bn(x)

        return x


class SlimmableDUCK(nn.Module):
    def __init__(self, width_mult_list, in_channels_list, out_channels_list, act_type, filter_size=6+1):
        '''
        Here I change the filter size of separated block to be odd number.
        '''
        super().__init__()
        self.in_bn = nn.Sequential(
                        SwitchableBatchNorm2d(width_mult_list, in_channels_list),
                        Activation(act_type)
                    )
        self.branch1 = SlimmableWidescopeBlock(width_mult_list, in_channels_list, out_channels_list, act_type)
        self.branch2 = SlimmableMidscopeBlock(width_mult_list, in_channels_list, out_channels_list, act_type)
        self.branch3 = SlimmableResidualBlock(width_mult_list, in_channels_list, out_channels_list, act_type)
        self.branch4 = nn.Sequential(
                            SlimmableResidualBlock(width_mult_list, in_channels_list, out_channels_list, act_type),
                            SlimmableResidualBlock(width_mult_list, out_channels_list, out_channels_list, act_type),
                        )
        self.branch5 = nn.Sequential(
                            SlimmableResidualBlock(width_mult_list, in_channels_list, out_channels_list, act_type),
                            SlimmableResidualBlock(width_mult_list, out_channels_list, out_channels_list, act_type),
                            SlimmableResidualBlock(width_mult_list, out_channels_list, out_channels_list, act_type),
                        )
        self.branch6 = SlimmableSeparatedBlock(width_mult_list, in_channels_list, out_channels_list, filter_size, act_type)
        self.out_bn = nn.Sequential(
                        SwitchableBatchNorm2d(width_mult_list, out_channels_list),
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


class SlimmableMidscopeBlock(nn.Sequential):
    def __init__(self, width_mult_list, in_channels_list, out_channels_list, act_type):
        super().__init__(
            SlimmableConvBNAct(width_mult_list, in_channels_list, out_channels_list, 3, act_type=act_type),
            SlimmableConvBNAct(width_mult_list, out_channels_list, out_channels_list, 3, dilation=2, act_type=act_type)
        )


class SlimmableWidescopeBlock(nn.Sequential):
    def __init__(self, width_mult_list, in_channels_list, out_channels_list, act_type):
        super().__init__(
            SlimmableConvBNAct(width_mult_list, in_channels_list, out_channels_list, 3, act_type=act_type),
            SlimmableConvBNAct(width_mult_list, out_channels_list, out_channels_list, 3, dilation=2, act_type=act_type),
            SlimmableConvBNAct(width_mult_list, out_channels_list, out_channels_list, 3, dilation=3, act_type=act_type),
        )


class SlimmableSeparatedBlock(nn.Sequential):
    def __init__(self, width_mult_list, in_channels_list, out_channels_list, filter_size, act_type):
        super().__init__(
            SlimmableConvBNAct(width_mult_list, in_channels_list, out_channels_list, (1, filter_size), act_type=act_type),
            SlimmableConvBNAct(width_mult_list, out_channels_list, out_channels_list, (filter_size, 1), act_type=act_type),
        )