import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantizedComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, bias=True, num_bits=5, tile_size=32):
        super(QuantizedComplexConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.dilation = self._pair(dilation)
        
        # 量化参数
        self.num_bits = num_bits
        self.tile_size = tile_size

        # 原始的全精度权重，作为 nn.Parameter 进行训练
        self.weight_real = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.weight_imag = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))

        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_channels))
            self.bias_imag = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        # 为量化后的权重矩阵准备存储空间 (非 nn.Parameter)
        self.q_weight_real_gemm = None
        self.q_weight_imag_gemm = None
            
        self.reset_parameters()

    def _pair(self, x):
        if isinstance(x, int):
            return (x, x)
        return x

    def reset_parameters(self):
        fan_in = self.in_channels * np.prod(self.kernel_size)
        fan_out = self.out_channels * np.prod(self.kernel_size)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(self.weight_real, 0, std)
        nn.init.normal_(self.weight_imag, 0, std)
        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def quantize_dequantize_per_tile(self, weight_matrix):
        """对权重矩阵进行分块伪量化 (Quantize-Dequantize)"""
        q_min = 0
        q_max = 2**self.num_bits - 1
        output = torch.zeros_like(weight_matrix)
        rows, cols = weight_matrix.shape

        for r_start in range(0, rows, self.tile_size):
            for c_start in range(0, cols, self.tile_size):
                r_end = min(r_start + self.tile_size, rows)
                c_end = min(c_start + self.tile_size, cols)
                tile = weight_matrix[r_start:r_end, c_start:c_end]
                
                if tile.numel() == 0:
                    continue

                max_val, min_val = tile.max(), tile.min()
                scale = (max_val - min_val) / (q_max - q_min) if max_val != min_val else 1.0
                zero_point = q_min - torch.round(min_val / (scale + 1e-8))
                zero_point = torch.clamp(zero_point, q_min, q_max)

                quantized_tile = torch.clamp(torch.round(tile / (scale + 1e-8)) + zero_point, q_min, q_max)
                dequantized_tile = (quantized_tile - zero_point) * scale
                output[r_start:r_end, c_start:c_end] = dequantized_tile
                
        return output

    def quantize_weights(self, free_memory=False):
        """
        [核心方法]
        对全精度权重执行一次性的量化/反量化，并将结果缓存以备 forward 调用。
        这个方法应该在模型训练完毕、进入推理模式后调用。
        
        :param free_memory: 若为True，则在量化后删除原始权重以节省内存。
        """
        # 将权重 reshape 成 GEMM 所需的矩阵形式 (C_out, C_in * kH * kW)
        weight_real_gemm = self.weight_real.view(self.out_channels, -1)
        weight_imag_gemm = self.weight_imag.view(self.out_channels, -1)

        # 对权重矩阵进行分块伪量化，并将结果存储
        self.q_weight_real_gemm = self.quantize_dequantize_per_tile(weight_real_gemm)
        self.q_weight_imag_gemm = self.quantize_dequantize_per_tile(weight_imag_gemm)
        
        # 确保量化后的权重在正确的设备上
        device = self.weight_real.device
        self.q_weight_real_gemm = self.q_weight_real_gemm.to(device)
        self.q_weight_imag_gemm = self.q_weight_imag_gemm.to(device)

        if free_memory:
            del self.weight_real
            del self.weight_imag
            self.weight_real = None
            self.weight_imag = None

    def forward(self, input_complex):
        """
        [高效的 forward]
        使用 im2col (F.unfold) 和 GEMM (torch.matmul) 执行卷积，
        并直接使用预先计算好的量化权重。
        """
        if self.q_weight_real_gemm is None or self.q_weight_imag_gemm is None:
            raise RuntimeError("Weights are not quantized. Call `quantize_weights()` before inference.")

        input_real = input_complex.real
        input_imag = input_complex.imag
        N, C, H, W = input_real.shape

        # 1. im2col: 使用 F.unfold 将输入图像块转换为列向量
        input_real_unf = F.unfold(input_real, self.kernel_size, self.dilation, self.padding, self.stride)
        input_imag_unf = F.unfold(input_imag, self.kernel_size, self.dilation, self.padding, self.stride)

        # 2. GEMM: 执行复数矩阵乘法
        # (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        # a,b: input_real_unf, input_imag_unf
        # c,d: q_weight_real_gemm, q_weight_imag_gemm
        output_real_unf = self.q_weight_real_gemm @ input_real_unf - self.q_weight_imag_gemm @ input_imag_unf
        output_imag_unf = self.q_weight_real_gemm @ input_imag_unf + self.q_weight_imag_gemm @ input_real_unf

        # 3. Reshape: 将结果恢复为图像格式
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        output_real = output_real_unf.view(N, self.out_channels, H_out, W_out)
        output_imag = output_imag_unf.view(N, self.out_channels, H_out, W_out)
        
        # 4. 添加偏置
        if self.bias_real is not None:
            output_real += self.bias_real.view(1, -1, 1, 1)
            output_imag += self.bias_imag.view(1, -1, 1, 1)
            
        return torch.complex(output_real, output_imag)


class QuantizedComplexConvTranspose2d(nn.Module):
    """
    使用 GEMM 和 5-bit 权重分块量化实现的复数二维转置卷积层。
    
    *** V2: 优化版本 ***
    量化过程被移出 forward pass，通过一个独立的 quantize_weights() 方法实现。
    这使得在推理过程中 forward pass 更快，避免了重复计算。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, bias=True, num_bits=5, tile_size=32):
        super(QuantizedComplexConvTranspose2d, self).__init__()

        # ... (参数初始化与之前版本相同)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.output_padding = self._pair(output_padding)
        self.dilation = self._pair(dilation)
        
        self.num_bits = num_bits
        self.tile_size = tile_size

        # 1. 原始的全精度权重仍然是 nn.Parameter
        # 这对于模型的保存、加载和可能的进一步微调非常重要
        self.weight_real = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))
        self.weight_imag = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))

        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_channels))
            self.bias_imag = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        # 2. 为量化后的权重矩阵准备存储空间 (非 nn.Parameter)
        # 这些是普通的张量，不会被 PyTorch 的自动求导机制跟踪
        self.q_weight_real_gemm = None
        self.q_weight_imag_gemm = None
            
        self.reset_parameters()

    def _pair(self, x):
        if isinstance(x, int):
            return (x, x)
        return x

    def reset_parameters(self):
        fan_in = self.in_channels * np.prod(self.kernel_size)
        gain = nn.init.calculate_gain('leaky_relu', 0.01)
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.0) * std
        nn.init.uniform_(self.weight_real, -bound, bound)
        nn.init.uniform_(self.weight_imag, -bound, bound)

        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def quantize_dequantize_per_tile(self, weight_matrix):
        """
        对权重矩阵进行分块伪量化 (Quantize-Dequantize)。
        (此函数实现与之前版本相同)
        """
        q_min = 0
        q_max = 2**self.num_bits - 1
        output = torch.zeros_like(weight_matrix)
        rows, cols = weight_matrix.shape

        for r_start in range(0, rows, self.tile_size):
            for c_start in range(0, cols, self.tile_size):
                r_end = min(r_start + self.tile_size, rows)
                c_end = min(c_start + self.tile_size, cols)
                tile = weight_matrix[r_start:r_end, c_start:c_end]
                
                if tile.numel() == 0:
                    continue

                max_val, min_val = tile.max(), tile.min()
                scale = (max_val - min_val) / (q_max - q_min) if max_val != min_val else 1.0
                zero_point = q_min - torch.round(min_val / (scale + 1e-8))
                zero_point = torch.clamp(zero_point, q_min, q_max)

                quantized_tile = torch.clamp(torch.round(tile / (scale + 1e-8)) + zero_point, q_min, q_max)
                dequantized_tile = (quantized_tile - zero_point) * scale
                output[r_start:r_end, c_start:c_end] = dequantized_tile
                
        return output
    
    def quantize_weights(self, free_memory=False):
        """
        **[核心新增方法]**
        对全精度权重执行一次性的量化/反量化，并将结果缓存以备 forward 调用。
        这个方法应该在模型训练完毕、进入推理模式后调用。
        
        :param free_memory: 如果为 True，将在量化后删除原始的全精度权重以节省内存。
                            注意：这会使模型无法再被微调或重新量化。
        """
        # 将权重 reshape 成 GEMM 所需的矩阵形式
        weight_real_gemm = self.weight_real.permute(1, 2, 3, 0).reshape(self.out_channels * np.prod(self.kernel_size), self.in_channels)
        weight_imag_gemm = self.weight_imag.permute(1, 2, 3, 0).reshape(self.out_channels * np.prod(self.kernel_size), self.in_channels)

        # 对权重矩阵进行分块伪量化，并将结果存储在 self.q_weight_* 中
        self.q_weight_real_gemm = self.quantize_dequantize_per_tile(weight_real_gemm)
        self.q_weight_imag_gemm = self.quantize_dequantize_per_tile(weight_imag_gemm)
        
        # 将量化后的权重移到与原始权重相同的设备
        self.q_weight_real_gemm = self.q_weight_real_gemm.to(self.weight_real.device)
        self.q_weight_imag_gemm = self.q_weight_imag_gemm.to(self.weight_imag.device)

        if free_memory:
            del self.weight_real
            del self.weight_imag
            self.weight_real = None
            self.weight_imag = None

    def _get_output_shape(self, input_shape):
        """计算转置卷积的输出尺寸 (与之前版本相同)"""
        N, C, H_in, W_in = input_shape
        H_out = (H_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        W_out = (W_in - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        return (H_out, W_out)

    def forward(self, input_complex):
        """
        **[核心修改]** forward pass 现在直接使用预先计算好的量化权重。
        """
        # 检查权重是否已被量化。如果没有，则抛出异常。
        if self.q_weight_real_gemm is None or self.q_weight_imag_gemm is None:
            raise RuntimeError("Weights have not been quantized. Please call `quantize_weights()` before running the forward pass.")

        input_real = input_complex.real
        input_imag = input_complex.imag
        N, C_in, H_in, W_in = input_real.shape
        
        output_shape = self._get_output_shape(input_real.shape)
        
        # 输入 reshape (与之前相同)
        input_real_reshaped = input_real.reshape(N, C_in, -1)
        input_imag_reshaped = input_imag.reshape(N, C_in, -1)

        # 执行复数矩阵乘法 (GEMM)，直接使用缓存的量化权重
        cols_rr = torch.matmul(self.q_weight_real_gemm, input_real_reshaped)
        cols_ii = torch.matmul(self.q_weight_imag_gemm, input_imag_reshaped)
        cols_ri = torch.matmul(self.q_weight_imag_gemm, input_real_reshaped)
        cols_ir = torch.matmul(self.q_weight_real_gemm, input_imag_reshaped)
        
        cols_real = cols_rr - cols_ii
        cols_imag = cols_ri + cols_ir
        
        # 使用 F.fold 恢复图像格式 (与之前相同)
        output_real = F.fold(cols_real, output_size=output_shape, kernel_size=self.kernel_size,
                             dilation=self.dilation, padding=self.padding, stride=self.stride)
        output_imag = F.fold(cols_imag, output_size=output_shape, kernel_size=self.kernel_size,
                             dilation=self.dilation, padding=self.padding, stride=self.stride)

        # 添加偏置项 (与之前相同)
        if self.bias_real is not None:
            output_real += self.bias_real.view(1, -1, 1, 1)
            output_imag += self.bias_imag.view(1, -1, 1, 1)
            
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
    

def quantized_complex_conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 complex convolution with padding"""
    return QuantizedComplexConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class ComplexBasicBlock(nn.Module):
    """Complex basic block for decoder"""

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ComplexBasicBlock, self).__init__()
        self.conv1 = quantized_complex_conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = ComplexBatchNorm2d(planes)
        self.relu = ComplexReLU(inplace=True)
        self.conv2 = quantized_complex_conv3x3(planes, planes, bias=True)
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