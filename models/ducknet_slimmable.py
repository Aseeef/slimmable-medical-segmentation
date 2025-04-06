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
class DuckNet(nn.Module):
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
        self.up_stage1 = UpsampleBlock(base_channel, base_channel, act_type)
        self.seg_head = conv1x1(base_channel, num_class)

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

# --------------------------- define slimablle blocks -----------------------------------

## SLIMMABLE VERSION
class DuckNet_slimmable(nn.Module):
    def __init__(self, num_class=1, n_channel=3, base_channel=34, act_type='relu', width_mult_list, in_channels_list, out_channels_list):
        
        # parameters for slimming
        self.width_mult_list = width_mult_list
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list

        # unchanged
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
        self.up_stage1 = UpsampleBlock(base_channel, base_channel, act_type)
        self.seg_head = conv1x1(base_channel, num_class)

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


class DownsampleBlock_slimmable(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, fuse_channels=None,width_mult_list, in_channels_list, out_channels_list):

        # parameters for slimming
        self.width_mult_list = width_mult_list
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list

        super().__init__() #????? What's the purpose of initilizing the superclass here again? 

        fuse_channels = in_channels if fuse_channels is None else fuse_channels
        self.duck = DUCK(in_channels, fuse_channels, act_type)
        self.conv1 = ConvBNAct(fuse_channels, out_channels, 3, 2, act_type=act_type)   ### how to do we slim ConvBNAct???????
        self.conv2 = ConvBNAct(in_channels, out_channels, 2, 2, act_type=act_type)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = self.conv2(x1)
        else:
            x2 = self.conv2(x2)

        skip = self.duck(x1)
        x1 = self.conv1(skip)

        return x1, skip, x2


class UpsampleBlock_slimmable(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.duck = DUCK(in_channels, out_channels, act_type)

    def forward(self, x, residual):
        size = residual.size()[2:]
        x = F.interpolate(x, size, mode='nearest')

        x += residual
        x = self.duck(x)

        return x


class ResidualBlock_slimmable(nn.Module):
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


class DUCK_slimmable(nn.Module):
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


class ConvBNAct_slimmable(nn.Sequential):    # ???????? Good idea to past this here rather than import from above?
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                    bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),        # replace this ??????
            nn.BatchNorm2d(out_channels),                                                                      # replace this ??????
            Activation(act_type, **kwargs)
        )


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super().__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)

# --------------------------- define slimablle blocks -----------------------------------

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, width_mult_list, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.width_mult_list = width_mult_list
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, width_mult_list, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        

        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        
        
        self.width_mult_list = width_mult_list
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]

        # select weight that modifies input/output channels
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, width_mult_list, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.width_mult_list = width_mult_list
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# TODO: we should probably do the universally slimmable
class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        if getattr(FLAGS, 'conv_averaged', False):
            y = y * (max(self.in_channels_list) / self.in_channels)
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.width_mult = None
        self.us = us

    def forward(self, input):
        if self.us[0]:
            self.in_features = make_divisible(
                self.in_features_max * self.width_mult)
        if self.us[1]:
            self.out_features = make_divisible(
                self.out_features_max * self.width_mult)
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=False) for i in [
                make_divisible(
                    self.num_features_max * width_mult / ratio) * ratio
                for width_mult in FLAGS.width_mult_list]])
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        if self.width_mult in FLAGS.width_mult_list:
            idx = FLAGS.width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y


def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]


def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None

