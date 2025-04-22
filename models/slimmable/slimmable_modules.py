from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slimmable.slimmable_ops import SlimmableLinear, SlimmableConv2d, SwitchableBatchNorm2d, \
    SlimmableConvTranspose2d


# Regular convolution with kernel size 3x3
def slimmable_conv3x3(width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                      stride: int | tuple[int, int] = 1, bias=False):
    return SlimmableConv2d(width_mult_list, in_channels_list, out_channels_list, kernel_size=3, stride=stride,
                           padding=1, bias=bias)


# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def slimmable_conv1x1(width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                      stride: int | tuple[int, int] = 1, bias=False):
    return SlimmableConv2d(width_mult_list, in_channels_list, out_channels_list, kernel_size=1, stride=stride,
                           padding=0, bias=bias)


# Depth-wise seperable convolution with batchnorm and activation
class SlimmableDSConvBNAct(nn.Sequential):
    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 kernel_size: int | tuple[int, ...], stride: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1, act_type: str = 'relu', **kwargs):
        super().__init__(
            SlimmableDWConvBNAct(width_mult_list, in_channels_list, in_channels_list, kernel_size, stride, dilation,
                                 act_type, **kwargs),
            SlimmablePWConvBNAct(width_mult_list, in_channels_list, out_channels_list, act_type, **kwargs)
        )


# Depth-wise convolution -> batchnorm -> activation
class SlimmableDWConvBNAct(nn.Sequential):
    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 kernel_size: int | tuple[int, ...], stride: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1, act_type: str = 'relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            raise ValueError('kernel_size should be int or list/tuple of ints')

        super().__init__(
            SlimmableConv2d(width_mult_list, in_channels_list, out_channels_list, kernel_size, stride, padding,
                            dilation=dilation, groups_list=in_channels_list, bias=False),
            SwitchableBatchNorm2d(width_mult_list, out_channels_list),
            Activation(act_type, **kwargs)
        )


# Point-wise convolution -> batchnorm -> activation
class SlimmablePWConvBNAct(nn.Sequential):
    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 act_type: str = 'relu', bias: bool = True, **kwargs):
        super().__init__(
            SlimmableConv2d(width_mult_list, in_channels_list, out_channels_list, 1, bias=bias),
            SwitchableBatchNorm2d(width_mult_list, out_channels_list),
            Activation(act_type, **kwargs)
        )


# Regular convolution -> batchnorm -> activation
class SlimmableConvBNAct(nn.Sequential):
    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 kernel_size: int | tuple[int, ...] = 3, stride: int | tuple[int, int] = 1, dilation=1,
                 groups_list: list[int] = [1], bias: bool = False, act_type: str = 'relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            raise ValueError('kernel_size should be int or list/tuple of ints')

        super().__init__(
            SlimmableConv2d(width_mult_list, in_channels_list, out_channels_list, kernel_size, stride, padding,
                            dilation, groups_list, bias),
            SwitchableBatchNorm2d(width_mult_list, out_channels_list),
            Activation(act_type, **kwargs)
        )


# Transposed /de- convolution -> batchnorm -> activation
class DeConvBNAct(nn.Module):
    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 scale_factor: int = 2, kernel_size: int | tuple[int, ...] = None,
                 padding: int | tuple[int, int] = None, act_type: str = 'relu', **kwargs):
        super().__init__()
        if kernel_size is None:
            kernel_size = 2 * scale_factor - 1
        if padding is None:
            padding = (kernel_size - 1) // 2
        output_padding = scale_factor - 1
        self.up_conv = nn.Sequential(
            SlimmableConvTranspose2d(width_mult_list, in_channels_list, out_channels_list,
                                     kernel_size=kernel_size,
                                     stride=scale_factor, padding=padding,
                                     output_padding=output_padding),
            SwitchableBatchNorm2d(width_mult_list, out_channels_list),
            Activation(act_type, **kwargs)
        )

    def forward(self, x):
        return self.up_conv(x)


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super().__init__()
        activation_hub = {'relu': nn.ReLU, 'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
                          'celu': nn.CELU, 'elu': nn.ELU,
                          'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU, 'glu': nn.GLU,
                          'selu': nn.SELU, 'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
                          'tanh': nn.Tanh, 'none': nn.Identity,
                          }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


class PyramidPoolingModule(nn.Module):
    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 act_type, pool_sizes=[1, 2, 4, 6], bias: bool = False):
        super().__init__()
        assert len(pool_sizes) == 4, 'Length of pool size should be 4.\n'
        hid_channels_list = [int(in_channels // 4) for in_channels in in_channels_list]
        self.stage1 = self._make_stage(width_mult_list, in_channels_list, hid_channels_list, pool_sizes[0])
        self.stage2 = self._make_stage(width_mult_list, in_channels_list, hid_channels_list, pool_sizes[1])
        self.stage3 = self._make_stage(width_mult_list, in_channels_list, hid_channels_list, pool_sizes[2])
        self.stage4 = self._make_stage(width_mult_list, in_channels_list, hid_channels_list, pool_sizes[3])
        self.conv = SlimmablePWConvBNAct(width_mult_list, 2 * in_channels_list, out_channels_list, act_type=act_type,
                                         bias=bias)

    def _make_stage(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                    pool_size: int):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            slimmable_conv1x1(width_mult_list, in_channels_list, out_channels_list)
        )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.stage1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.stage2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.stage3(x), size, mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.stage4(x), size, mode='bilinear', align_corners=True)
        x = self.conv(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x


class ASPP(nn.Module):
    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 dilations: List[int] = [1, 6, 12, 18], act_type: str = 'relu'):
        super().__init__()
        assert isinstance(dilations, (list, tuple))
        assert len(dilations) > 0
        num_branch = len(dilations) + 1

        hid_channels_list = [out_channels // num_branch for out_channels in out_channels_list]

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SlimmableConvBNAct(width_mult_list,
                               in_channels_list,
                               hid_channels_list,
                               1,
                               act_type=act_type)
        )
        self.dilated_convs = nn.ModuleList([
            SlimmableConvBNAct(width_mult_list, in_channels_list, hid_channels_list, 3, dilation=d, act_type=act_type)
            for d in dilations
        ])

        self.conv = SlimmableConvBNAct(width_mult_list, hid_channels_list * num_branch, out_channels_list)

    def forward(self, x):
        size = x.size()[2:]
        x_pool = self.pool(x)
        x_pool = F.interpolate(x_pool, size, mode='bilinear', align_corners=True)

        feats = [x_pool]
        for dilated_conv in self.dilated_convs:
            feat = dilated_conv(x)
            feats.append(feat)

        x = torch.cat(feats, dim=1)
        x = self.conv(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, width_mult_list: List[float], channels_list: List[int], act_type: str, reduction_num: int = 16):
        super().__init__()
        squeeze_channels = [channels // reduction_num for channels in channels_list]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential(
            SlimmableLinear(width_mult_list, channels_list, squeeze_channels),
            Activation(act_type),
            SlimmableLinear(width_mult_list, squeeze_channels, channels_list),
            Activation('sigmoid')
        )

    def forward(self, x):
        residual = x
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.se_block(x).unsqueeze(-1).unsqueeze(-1)
        x = x * residual

        return x


class SlimmableSegHead(nn.Sequential):
    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], num_class: int,
                 act_type: str, hid_channels_list: List[int]):
        num_class_list = [num_class for _ in in_channels_list]
        super().__init__(
            SlimmableConvBNAct(width_mult_list, in_channels_list, hid_channels_list, 3, act_type=act_type),
            slimmable_conv1x1(width_mult_list, hid_channels_list, num_class_list)
        )
