"""
CREDIT: https://github.com/JiahuiYu/slimmable_networks/blob/master/models/slimmable_ops.py
Modified Slimmable Neural Network Components
"""
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwitchableBatchNorm2d(nn.Module):
    """
    BatchNorm2d layer that switches between different numbers of features based on width multiplier.

    Args:
        width_mult_list (List[float]): List of width multipliers available.
        num_features_list (List[int]): List of number of features corresponding to each width multiplier.
    """

    def __init__(self, width_mult_list: List[float], num_features_list: List[int]):
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

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the appropriate BatchNorm2d layer.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Batch-normalized output.
        """
        idx = self.width_mult_list.index(self.width_mult)
        return self.bn[idx](input)


class SlimmableConv2d(nn.Conv2d):
    """
    Conv2d layer that dynamically switches channels and groups based on width multiplier.

    Args:
        width_mult_list (List[float]): List of width multipliers.
        in_channels_list (List[int]): List of input channels for each multiplier.
        out_channels_list (List[int]): List of output channels for each multiplier.
        kernel_size: Size of the convolving kernel.
        stride (int | tuple): Stride of the convolution.
        padding (int | tuple): Zero-padding added to both sides.
        dilation (int | tuple): Spacing between kernel elements.
        groups_list (List[int]): List of group sizes for convolution.
        bias (bool): If True, adds a learnable bias.
    """

    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 kernel_size, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1, groups_list: List[int] = [1], bias=True):
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

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass with selected channel and group configuration.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Convolved output.
        """
        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableConvTranspose2d(nn.ConvTranspose2d):
    """
    ConvTranspose2d layer that adjusts its input/output channels and groups based on width multiplier.

    Args:
        width_mult_list (List[float]): List of width multipliers.
        in_channels_list (List[int]): Input channels corresponding to each multiplier.
        out_channels_list (List[int]): Output channels for each multiplier.
        kernel_size (int | tuple): Size of the convolving kernel.
        stride (int | tuple): Stride of the convolution.
        padding (int | tuple): Zero-padding added to both sides.
        output_padding (int | tuple): Additional size added to one side of output shape.
        dilation (int | tuple): Spacing between kernel elements.
        groups_list (List[int]): Number of blocked connections.
        bias (bool): If True, adds a learnable bias.
    """

    def __init__(self, width_mult_list: List[float], in_channels_list: List[int], out_channels_list: List[int],
                 kernel_size: int | tuple[int, ...], stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0, output_padding: int | tuple[int, ...] = 0,
                 dilation: int | tuple[int, int] = 1, groups_list: List[int] = [1], bias: bool = True):
        super(SlimmableConvTranspose2d, self).__init__(
            max(in_channels_list), max(out_channels_list), kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=max(groups_list), bias=bias, dilation=dilation)
        self.width_mult_list = width_mult_list
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        # Use the largest width multiplier as default.
        self.width_mult = max(width_mult_list)
        # Store output_padding for use in the forward pass.
        self.output_padding = output_padding

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        """
        Forward pass for transposed convolution with dynamic channel selection.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).
            output_size (Optional[List[int]]): Desired output size (unused but reserved).

        Returns:
            Tensor: Output tensor after transposed convolution.
        """
        # Select the configuration based on the current width multiplier.
        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        # For ConvTranspose2d, weight shape is:
        # (in_channels, out_channels//groups, kernel_height, kernel_width)
        weight = self.weight[:self.in_channels, :self.out_channels // self.groups, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = F.conv_transpose2d(
            input, weight, bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)
        return y


class SlimmableLinear(nn.Linear):
    """
    Linear layer that adjusts its input and output features based on the width multiplier.

    Args:
        width_mult_list (List[float]): List of width multipliers.
        in_features_list (List[int]): List of input feature counts for each width multiplier.
        out_features_list (List[int]): List of output feature counts for each width multiplier.
        bias (bool): If True, adds a learnable bias to the output.
    """

    def __init__(self, width_mult_list: List[float], in_features_list: List[int],
                 out_features_list: List[int], bias: bool = True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.width_mult_list = width_mult_list
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(width_mult_list)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass using dynamically selected weight and bias slices.

        Args:
            input (Tensor): Input tensor of shape (N, in_features).

        Returns:
            Tensor: Output tensor of shape (N, out_features).
        """
        idx = self.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


def make_divisible(v: float, divisor: int = 8, min_value: int = 1) -> int:
    """
    Ensures that the number `v` is divisible by `divisor` and greater than `min_value`.

    Args:
        v (int): Original number.
        divisor (int): Divisor to align to. Default is 8.
        min_value (int): Minimum allowed value. Default is 1.

    Returns:
        int: Adjusted number that meets divisibility constraint.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class USConv2d(nn.Conv2d):
    """
    A Conv2d layer with dynamically adjustable input/output channels for universal slimmability.

    Args:
        in_channels (int): Maximum number of input channels.
        out_channels (int): Maximum number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Zero-padding added to both sides. Default is 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        depthwise (bool, optional): If True, applies depthwise convolution. Default is False.
        bias (bool, optional): If True, adds a learnable bias. Default is True.
        us (List[bool], optional): Whether to apply slimmability to input and/or output channels. Default is [True, True].
        ratio (List[float], optional): Ratio for computing divisible channels. Default is [1, 1].
        conv_averaged (bool, optional): Whether to average output when width is reduced. Default is False.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1, groups: int = 1,
                 depthwise: bool = False, bias: bool = True,
                 us: tuple[bool, bool] = [True, True], ratio: tuple[int, int] = [1, 1], conv_averaged: bool = False):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.conv_averaged = conv_averaged
        self.us = us
        self.ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        """
        Width aware Conv2d for US Network

        * out_channels   follows the bucket dictated by width_mult
        * in_channels    whatever #channels the tensor actually carries (x.size(1))
        """

        # decide the output slice
        if self.us[1]:
            wanted = make_divisible(self.out_channels_max *
                                    self.width_mult / self.ratio[1]) * self.ratio[1]
            self.out_channels = min(wanted, self.out_channels_max)

        # take the actual input‑channel count
        real_in = x.size(1)
        self.in_channels = real_in

        # set groups
        self.groups = real_in if self.depthwise else self.groups

        # slice weights, bias and conv
        weight = self.weight[: self.out_channels, : real_in, :, :]
        bias   = None if self.bias is None else self.bias[: self.out_channels]

        return F.conv2d(x, weight, bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)


class USConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0,
                 output_padding: int | tuple[int, ...] = 0,
                 dilation: int | tuple[int, int] = 1, groups: int = 1,
                 depthwise: bool = False, bias: bool = True,
                 us: tuple[bool, bool] = [True, True], ratio: tuple[int, int] = [1, 1], conv_averaged: bool = False):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.conv_averaged = conv_averaged
        self.us = us
        self.ratio = ratio

    def forward(self, input: Tensor) -> Tensor:
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
        y = nn.functional.conv_transpose2d(
            input, weight, bias, self.stride, self.padding, self.output_padding,
            self.groups, self.dilation)
        if self.conv_averaged:
            y = y * (max(self.in_channels_list) / self.in_channels)
        return y


class USLinear(nn.Linear):
    """
    A Linear layer with input/output features adjustable according to a width multiplier.

    Args:
        in_features (int): Maximum number of input features.
        out_features (int): Maximum number of output features.
        bias (bool): If True, adds a learnable bias to the output. Default is True.
        us (None | tuple[bool, bool]): Whether to apply slimmability to input and/or output features. Default is [True, True].
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, us: tuple[bool, bool] = [True, True]):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.width_mult = None
        self.us = us

    def forward(self, input):
        """
        Forward pass using sliced weights and bias based on width multiplier.

        Args:
            input (Tensor): Input tensor of shape (N, in_features).

        Returns:
            Tensor: Output tensor of shape (N, out_features).
        """
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
    """
    A BatchNorm2d layer that dynamically adjusts the number of features based on width multiplier.

    Args: num_features (int): Maximum number of features. ratio (int): Channel alignment ratio (used with
    make_divisible). Default is 1. width_mult_list (List[float]): List of width multipliers. Specifying is
    recommended for performance and separate tracking of statistics.
    """

    def __init__(self, num_features: int, ratio: int = 1, width_mult_list: List[float] = None):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.width_mult_list = width_mult_list
        self.num_features_max = num_features
        self.ratio = ratio
        # for tracking performance during training
        #self.bn = None if width_mult_list is None else nn.ModuleList([
            #nn.BatchNorm2d(i, affine=False) for i in [
            #    make_divisible(
            #        self.num_features_max * width_mult / ratio) * ratio
            #    for width_mult in self.width_mult_list]])
        self.bn = None if width_mult_list is None else nn.ModuleList([
            nn.BatchNorm2d(
                make_divisible(self.num_features_max * width_mult / self.ratio) * self.ratio, affine=False)
            for width_mult in self.width_mult_list])

        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, x: Tensor) -> Tensor:
        c = x.size(1)                     # exactly the input channels (shape of tensor)

        # affine‑params sliced to length c
        weight =   self.weight[:c]
        bias   =   self.bias[:c]

        if self.bn is not None:
            bn = self._pick_bn(c)         # guaranteed len >= c
            running_mean = bn.running_mean[:c]
            running_var  = bn.running_var [:c]
        else:                             # single universal BN
            running_mean = self.running_mean[:c]
            running_var  = self.running_var [:c]

        return F.batch_norm(x, running_mean, running_var,
                            weight, bias, self.training,
                            self.momentum, self.eps)

    def _pick_bn(self, need_c):
        for bn in self.bn:                # the bn list is sorted from small to large
            if bn.running_mean.numel() >= need_c:
                return bn
        return self.bn[-1]


def pop_channels(autoslim_channels: List[List[int]]):
    """
    Pops the first channel count from each sublist in a list of channel lists.

    Args:
        autoslim_channels (List[List[int]]): Nested list of channel options.

    Returns:
        List[int]: List of first popped channel counts.
    """
    return [i.pop(0) for i in autoslim_channels]


def bn_calibration_init(m: nn.Module, cumulative_bn_stats: bool = False):
    """
    Initializes BatchNorm layers for calibration by resetting and setting mode.
    This is used in the calibration of Universally Slimmable Networks (only) since the
    original SSNs maintain independent batch norms for each width multiplier.

    Args:
        m (nn.Module): The module to initialize.
        cumulative_bn_stats (bool): Whether to use cumulative moving average. Default is False.
    """
    if getattr(m, 'track_running_stats', False):
        # Reset all values for post-statistics
        m.reset_running_stats()
        # Set bn in training mode to update post-statistics
        m.training = True
        # If using cumulative moving average (i.e., exact averaging per Eq. 8)
        if cumulative_bn_stats:
            m.momentum = None
