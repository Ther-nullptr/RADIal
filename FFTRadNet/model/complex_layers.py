import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair

class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution layer"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ComplexConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Real and imaginary parts of the weight
        self.weight_real = nn.Parameter(torch.randn(out_channels, in_channels, *self._pair(kernel_size)))
        self.weight_imag = nn.Parameter(torch.randn(out_channels, in_channels, *self._pair(kernel_size)))

        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_channels))
            self.bias_imag = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.reset_parameters()

    def _pair(self, x):
        if isinstance(x, int):
            return (x, x)
        return x

    def reset_parameters(self):
        # Initialize weights using Xavier initialization for complex networks
        fan_in = self.in_channels * np.prod(self._pair(self.kernel_size))
        fan_out = self.out_channels * np.prod(self._pair(self.kernel_size))
        std = np.sqrt(2.0 / (fan_in + fan_out))

        nn.init.normal_(self.weight_real, 0, std)
        nn.init.normal_(self.weight_imag, 0, std)

        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def forward(self, input_complex):
        """
        input_complex: complex tensor with shape [batch, channels, height, width]
        """
        input_real = input_complex.real
        input_imag = input_complex.imag

        # Complex convolution: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        output_real = F.conv2d(input_real, self.weight_real, self.bias_real,
                              self.stride, self.padding, self.dilation) - \
                     F.conv2d(input_imag, self.weight_imag, None,
                              self.stride, self.padding, self.dilation)

        output_imag = F.conv2d(input_real, self.weight_imag, self.bias_imag,
                              self.stride, self.padding, self.dilation) + \
                     F.conv2d(input_imag, self.weight_real, None,
                              self.stride, self.padding, self.dilation)

        return torch.complex(output_real, output_imag)


class ComplexBatchNorm2d(nn.Module):
    """Complex-valued batch normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight_real = nn.Parameter(torch.ones(num_features))
            self.weight_imag = nn.Parameter(torch.zeros(num_features))
            self.bias_real = nn.Parameter(torch.zeros(num_features))
            self.bias_imag = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight_real', None)
            self.register_parameter('weight_imag', None)
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_real', torch.zeros(num_features))
            self.register_buffer('running_mean_imag', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_real', None)
            self.register_parameter('running_mean_imag', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, input_complex):
        """
        input_complex: complex tensor with shape [batch, channels, height, width]
        """
        input_real = input_complex.real
        input_imag = input_complex.imag

        if self.training and self.track_running_stats:
            # Calculate statistics
            mean_real = input_real.mean(dim=(0, 2, 3))
            mean_imag = input_imag.mean(dim=(0, 2, 3))

            # Complex variance: E[|z - E[z]|^2]
            centered_real = input_real - mean_real.view(1, -1, 1, 1)
            centered_imag = input_imag - mean_imag.view(1, -1, 1, 1)
            var = (centered_real**2 + centered_imag**2).mean(dim=(0, 2, 3))

            # Update running statistics
            with torch.no_grad():
                self.running_mean_real.mul_(1 - self.momentum).add_(mean_real, alpha=self.momentum)
                self.running_mean_imag.mul_(1 - self.momentum).add_(mean_imag, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
                self.num_batches_tracked += 1
        else:
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
            var = self.running_var

        # Normalize
        std = torch.sqrt(var + self.eps)
        normalized_real = (input_real - mean_real.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        normalized_imag = (input_imag - mean_imag.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

        if self.affine:
            # Apply complex affine transformation
            weight_real = self.weight_real.view(1, -1, 1, 1)
            weight_imag = self.weight_imag.view(1, -1, 1, 1)
            bias_real = self.bias_real.view(1, -1, 1, 1)
            bias_imag = self.bias_imag.view(1, -1, 1, 1)

            output_real = normalized_real * weight_real - normalized_imag * weight_imag + bias_real
            output_imag = normalized_real * weight_imag + normalized_imag * weight_real + bias_imag
        else:
            output_real = normalized_real
            output_imag = normalized_imag

        return torch.complex(output_real, output_imag)


class ComplexReLU(nn.Module):
    """Complex ReLU activation: applies ReLU to both real and imaginary parts"""

    def __init__(self, inplace=False):
        super(ComplexReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input_complex):
        return torch.complex(F.relu(input_complex.real, inplace=self.inplace),
                           F.relu(input_complex.imag, inplace=self.inplace))


class ComplexModReLU(nn.Module):
    """Modulus ReLU: ReLU(|z|) * z/|z|"""

    def __init__(self, inplace=False):
        super(ComplexModReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input_complex):
        modulus = torch.abs(input_complex)
        activated_modulus = F.relu(modulus, inplace=self.inplace)

        # Avoid division by zero
        safe_modulus = torch.where(modulus > 1e-8, modulus, torch.ones_like(modulus))
        phase = input_complex / safe_modulus

        return activated_modulus.unsqueeze(-1) * torch.stack([phase.real, phase.imag], dim=-1).sum(dim=-1)


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued 2D transposed convolution layer"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, bias=True):
        super(ComplexConvTranspose2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

        # Real and imaginary parts of the weight
        self.weight_real = nn.Parameter(torch.randn(in_channels, out_channels, *self._pair(kernel_size)))
        self.weight_imag = nn.Parameter(torch.randn(in_channels, out_channels, *self._pair(kernel_size)))

        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_channels))
            self.bias_imag = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.reset_parameters()

    def _pair(self, x):
        if isinstance(x, int):
            return (x, x)
        return x

    def reset_parameters(self):
        fan_in = self.in_channels * np.prod(self._pair(self.kernel_size))
        fan_out = self.out_channels * np.prod(self._pair(self.kernel_size))
        std = np.sqrt(2.0 / (fan_in + fan_out))

        nn.init.normal_(self.weight_real, 0, std)
        nn.init.normal_(self.weight_imag, 0, std)

        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def forward(self, input_complex):
        input_real = input_complex.real
        input_imag = input_complex.imag

        # Complex transposed convolution
        output_real = F.conv_transpose2d(input_real, self.weight_real, self.bias_real,
                                       self.stride, self.padding, self.output_padding,
                                       dilation=self.dilation) - \
                     F.conv_transpose2d(input_imag, self.weight_imag, None,
                                       self.stride, self.padding, self.output_padding,
                                       dilation=self.dilation)

        output_imag = F.conv_transpose2d(input_real, self.weight_imag, self.bias_imag,
                                       self.stride, self.padding, self.output_padding,
                                       dilation=self.dilation) + \
                     F.conv_transpose2d(input_imag, self.weight_real, None,
                                       self.stride, self.padding, self.output_padding,
                                       dilation=self.dilation)

        return torch.complex(output_real, output_imag)


def complex_conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 complex convolution with padding"""
    return ComplexConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=bias)


class ComplexBasicBlock(nn.Module):
    """Complex basic block for decoder"""

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ComplexBasicBlock, self).__init__()
        self.conv1 = complex_conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = ComplexBatchNorm2d(planes)
        self.relu = ComplexReLU(inplace=True)
        self.conv2 = complex_conv3x3(planes, planes, bias=True)
        self.bn2 = ComplexBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out