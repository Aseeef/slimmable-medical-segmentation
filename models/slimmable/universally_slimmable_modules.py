from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slimmable.slimmable_ops import USLinear, USConv2d, USBatchNorm2d, \
    USConvTranspose2d


# Regular convolution with kernel size 3x3
def us_conv3x3(in_channels: int, out_channels: int,
               stride: int | tuple[int, int] = 1, bias=False):
    return USConv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                    padding=1, bias=bias)


# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def us_conv1x1(in_channels: int, out_channels: int,
               stride: int | tuple[int, int] = 1, bias=False):
    return USConv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                    padding=0, bias=bias)


# Depth-wise seperable convolution with batchnorm and activation
class USDSConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int, ...], stride: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1, act_type: str = 'relu', **kwargs):
        super().__init__(
            USDWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation,
                          act_type, **kwargs),
            USPWConvBNAct(in_channels, out_channels, act_type, **kwargs)
        )


# Depth-wise convolution -> batchnorm -> activation
class USDWConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int, ...], stride: int | tuple[int, int] = 1,
                 dilation: int | tuple[int, int] = 1, act_type: str = 'relu',
                 width_mult_list: list[float] = None, **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            raise ValueError('kernel_size should be int or list/tuple of ints')

        super().__init__(
            USConv2d(in_channels, out_channels, kernel_size, stride, padding,
                     dilation=dilation, groups=in_channels, bias=False),
            USBatchNorm2d(out_channels, width_mult_list=width_mult_list),
            Activation(act_type, **kwargs)
        )


# Point-wise convolution -> batchnorm -> activation
class USPWConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 act_type: str = 'relu', bias: bool = True,
                 width_mult_list: list[float] = None, **kwargs):
        super().__init__(
            USConv2d(in_channels, out_channels, 1, bias=bias),
            USBatchNorm2d(out_channels, width_mult_list=width_mult_list),
            Activation(act_type, **kwargs)
        )


# Regular convolution -> batchnorm -> activation
class USConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int, ...] = 3, stride: int | tuple[int, int] = 1, dilation=1,
                 groups: int = 1, bias: bool = False, act_type: str = 'relu',
                 width_mult_list: list[float] = None, **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation
        else:
            raise ValueError('kernel_size should be int or list/tuple of ints')

        super().__init__(
            USConv2d(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, groups, bias),
            USBatchNorm2d(out_channels, width_mult_list=width_mult_list),
            Activation(act_type, **kwargs)
        )


# Transposed /de- convolution -> batchnorm -> activation
class DeConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 scale_factor: int = 2, kernel_size: int | tuple[int, ...] = None,
                 padding: int | tuple[int, int] = None, act_type: str = 'relu',
                 width_mult_list: list[float] = None, **kwargs):
        super().__init__()
        if kernel_size is None:
            kernel_size = 2 * scale_factor - 1
        if padding is None:
            padding = (kernel_size - 1) // 2
        output_padding = scale_factor - 1
        self.up_conv = nn.Sequential(
            USConvTranspose2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=scale_factor, padding=padding,
                              output_padding=output_padding),
            USBatchNorm2d(out_channels, width_mult_list=width_mult_list),
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
    def __init__(self, in_channels: int, out_channels: int,
                 act_type: str, pool_sizes: List[int] = [1, 2, 4, 6], bias: bool = False,
                 width_mult_list: list[float] = None):
        super().__init__()
        assert len(pool_sizes) == 4, 'Length of pool size should be 4.\n'
        hid_channels = int(in_channels // 4)
        self.stage1 = self._make_stage(in_channels, hid_channels, pool_sizes[0])
        self.stage2 = self._make_stage(in_channels, hid_channels, pool_sizes[1])
        self.stage3 = self._make_stage(in_channels, hid_channels, pool_sizes[2])
        self.stage4 = self._make_stage(in_channels, hid_channels, pool_sizes[3])
        self.conv = USPWConvBNAct(2 * in_channels, out_channels, act_type=act_type,
                                  bias=bias, width_mult_list=width_mult_list)

    def _make_stage(self, in_channels: int, out_channels: int, pool_size: int):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            us_conv1x1(in_channels, out_channels)
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
    def __init__(self, in_channels: int, out_channels: int,
                 dilations: List[int] = [1, 6, 12, 18], act_type: str = 'relu',
                 width_mult_list: list[float] = None):
        super().__init__()
        assert isinstance(dilations, (list, tuple))
        assert len(dilations) > 0
        num_branch = len(dilations) + 1

        hid_channels = out_channels // num_branch

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            USConvBNAct(
                in_channels,
                hid_channels,
                1,
                act_type=act_type,
                width_mult_list=width_mult_list
            )
        )
        self.dilated_convs = nn.ModuleList([
            USConvBNAct(in_channels, hid_channels, 3,
                        dilation=d, act_type=act_type, width_mult_list=width_mult_list)
            for d in dilations
        ])

        self.conv = USConvBNAct(hid_channels * num_branch, out_channels, width_mult_list=width_mult_list)

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
    def __init__(self, channels: int, act_type: str, reduction_num: int = 16):
        super().__init__()
        squeeze_channels = channels // reduction_num
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential(
            USLinear(channels, squeeze_channels),
            Activation(act_type),
            USLinear(squeeze_channels, channels),
            Activation('sigmoid')
        )

    def forward(self, x):
        residual = x
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.se_block(x).unsqueeze(-1).unsqueeze(-1)
        x = x * residual

        return x


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=128, width_mult_list: list[float] | None = None):
        super().__init__(
            USConvBNAct(in_channels, hid_channels, 3, act_type=act_type, width_mult_list=width_mult_list),
            us_conv1x1(hid_channels, num_class)
        )
