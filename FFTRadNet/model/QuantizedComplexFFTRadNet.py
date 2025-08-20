import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence
from .quantized_complex_layers import QuantizedComplexConv2d, ComplexBatchNorm2d, ComplexReLU, QuantizedComplexConvTranspose2d, quantized_complex_conv3x3, ComplexBasicBlock

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna


class ComplexDetection_Header(nn.Module):
    """Complex detection header - uses complex operations until final output"""

    def __init__(self, use_bn=True, reg_layer=2, input_angle_size=0):
        super(ComplexDetection_Header, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if(self.input_angle_size==224):
            self.conv1 = quantized_complex_conv3x3(256, 144, bias=bias)
            self.bn1 = ComplexBatchNorm2d(144)
            self.conv2 = quantized_complex_conv3x3(144, 96, bias=bias)
            self.bn2 = ComplexBatchNorm2d(96)
        elif(self.input_angle_size==448):
            self.conv1 = QuantizedComplexConv2d(256, 144, kernel_size=3, stride=(1,2), padding=1, bias=bias)
            self.bn1 = ComplexBatchNorm2d(144)
            self.conv2 = quantized_complex_conv3x3(144, 96, bias=bias)
            self.bn2 = ComplexBatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = QuantizedComplexConv2d(256, 144, kernel_size=3, stride=(1,2), padding=1, bias=bias)
            self.bn1 = ComplexBatchNorm2d(144)
            self.conv2 = QuantizedComplexConv2d(144, 96, kernel_size=3, stride=(1,2), padding=1, bias=bias)
            self.bn2 = ComplexBatchNorm2d(96)
        elif(self.input_angle_size==1024):
            self.conv1 = QuantizedComplexConv2d(256, 144, kernel_size=3, stride=(1,4), padding=1, bias=bias)
            self.bn1 = ComplexBatchNorm2d(144)
            self.conv2 = QuantizedComplexConv2d(144, 96, kernel_size=3, stride=(1,2), padding=1, bias=bias)
            self.bn2 = ComplexBatchNorm2d(96)
        else:
            # Default case
            self.conv1 = quantized_complex_conv3x3(256, 144, bias=bias)
            self.bn1 = ComplexBatchNorm2d(144)
            self.conv2 = quantized_complex_conv3x3(144, 96, bias=bias)
            self.bn2 = ComplexBatchNorm2d(96)
            self.need_adaptive_pool = True

        self.conv3 = quantized_complex_conv3x3(96, 96, bias=bias)
        self.bn3 = ComplexBatchNorm2d(96)
        self.conv4 = quantized_complex_conv3x3(96, 96, bias=bias)
        self.bn4 = ComplexBatchNorm2d(96)
        self.relu = ComplexReLU(inplace=True)

        # Final conversion to real for output
        self.clshead_real = nn.Conv2d(96, 1, kernel_size=3, padding=1, bias=True)
        self.reghead_real = nn.Conv2d(96, reg_layer, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # Handle both complex and real inputs
        if not x.is_complex():
            # Convert real input to complex by setting imaginary part to zero
            x = torch.complex(x, torch.zeros_like(x))

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)

        # Apply adaptive pooling if needed (convert to real temporarily)
        if hasattr(self, 'need_adaptive_pool') and self.need_adaptive_pool:
            x_mag = torch.abs(x)
            x_phase = torch.angle(x)
            x_mag_pooled = F.adaptive_avg_pool2d(x_mag, (x_mag.shape[2], 224))
            x_phase_pooled = F.adaptive_avg_pool2d(x_phase, (x_phase.shape[2], 224))
            x = x_mag_pooled * torch.exp(1j * x_phase_pooled)

        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)
        x = self.relu(x)

        # Convert to real for final classification and regression heads
        x_real = torch.abs(x)  # Use magnitude for final processing

        cls = torch.sigmoid(self.clshead_real(x_real))
        reg = self.reghead_real(x_real)

        return torch.cat([cls, reg], dim=1)


class ComplexFreespace(nn.Module):
    """Complex freespace segmentation module"""

    def __init__(self):
        super(ComplexFreespace, self).__init__()

        # Complex processing blocks
        self.block1 = ComplexBasicBlock(256, 128)
        self.block2 = ComplexBasicBlock(128, 64)

        # Final conversion to real for segmentation output
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Process with complex blocks
        x = self.block1(x)
        x = self.block2(x)

        # Convert to real and generate final segmentation map
        x_real = torch.abs(x)  # Use magnitude
        out = self.final_conv(x_real)

        return out


class ComplexBottleneck(nn.Module):
    """Complex-valued bottleneck block"""

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=4):
        super(ComplexBottleneck, self).__init__()
        self.conv1 = QuantizedComplexConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(planes)
        self.conv2 = QuantizedComplexConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(planes)
        self.conv3 = QuantizedComplexConv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = ComplexBatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = ComplexReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Complex addition
        out = out + residual
        out = self.relu(out)
        return out


class ComplexMIMO_PreEncoder(nn.Module):
    """Complex MIMO pre-encoder"""
    def __init__(self, in_layer, out_layer, kernel_size=(1,12), dilation=(1,16), use_bn=False):
        super(ComplexMIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn
        self.conv = QuantizedComplexConv2d(in_layer, out_layer, kernel_size, stride=(1, 1), padding=0, dilation=dilation, bias=(not use_bn))
        self.bn = ComplexBatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna/2)

    def forward(self, x):
        width = x.shape[-1]
        # Complex concatenation along width dimension
        x_padded = torch.cat([x[...,-self.padding:], x, x[...,:self.padding]], dim=3)
        x = self.conv(x_padded)
        x = x[..., int(x.shape[-1]/2-width/2):int(x.shape[-1]/2+width/2)]

        if self.use_bn:
            x = self.bn(x)
        return x


class ComplexFPN_BackBone(nn.Module):
    """Complex FPN backbone"""

    def __init__(self, num_block, channels, block_expansion, mimo_layer, use_bn=True):
        super(ComplexFPN_BackBone, self).__init__()

        self.block_expansion = block_expansion
        self.use_bn = use_bn

        # Pre-processing block to reorganize MIMO channels - now expects 16 complex channels
        self.pre_enc = ComplexMIMO_PreEncoder(16, mimo_layer,
                                            kernel_size=(1,NbTxAntenna),
                                            dilation=(1,NbRxAntenna),
                                            use_bn=True)

        self.in_planes = mimo_layer

        self.conv = quantized_complex_conv3x3(self.in_planes, self.in_planes)
        self.bn = ComplexBatchNorm2d(self.in_planes)
        self.relu = ComplexReLU(inplace=True)

        # Residual blocks
        self.block1 = self._make_layer(ComplexBottleneck, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(ComplexBottleneck, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(ComplexBottleneck, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(ComplexBottleneck, planes=channels[3], num_blocks=num_block[3])

    def forward(self, x):
        x = self.pre_enc(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                QuantizedComplexConv2d(self.in_planes, planes * self.block_expansion,
                             kernel_size=1, stride=2, bias=False),
                ComplexBatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = QuantizedComplexConv2d(self.in_planes, planes * self.block_expansion, kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample, expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1, expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)


class ComplexRangeAngle_Decoder(nn.Module):
    """Complex Range-Angle decoder - keeps complex operations until final output"""
    def __init__(self):
        super(ComplexRangeAngle_Decoder, self).__init__()

        # Complex top-down layers
        self.deconv4 = QuantizedComplexConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))

        self.conv_block4 = ComplexBasicBlock(48, 128)  # 16 + 32 = 48
        self.deconv3 = QuantizedComplexConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = ComplexBasicBlock(192, 256)  # 128 + 64 = 192

        # Complex channel mapping layers
        self.L3 = QuantizedComplexConv2d(192, 32, kernel_size=1, stride=1, padding=0)
        self.L2 = QuantizedComplexConv2d(160, 64, kernel_size=1, stride=1, padding=0)


    def forward(self, features):
        # Keep complex operations throughout
        # T4 shape: [4, 1024, 32, 16] -> transpose -> [4, 16, 32, 1024]
        T4 = features['x4'].transpose(1, 3)

        # Apply complex L3 first, then transpose
        T3_processed = self.L3(features['x3'])
        T3 = T3_processed.transpose(1, 3)

        # Apply complex L2 first, then transpose
        T2_processed = self.L2(features['x2'])
        T2 = T2_processed.transpose(1, 3)

        # Complex deconv4 upsampling
        deconv4_out = self.deconv4(T4)

        # Handle potential dimension mismatch with complex interpolation
        # Convert to magnitude and phase, interpolate separately, then recombine
        T3_mag = torch.abs(T3)
        T3_phase = torch.angle(T3)
        T3_mag_resized = F.interpolate(T3_mag, size=deconv4_out.shape[2:], mode='bilinear', align_corners=False)
        T3_phase_resized = F.interpolate(T3_phase, size=deconv4_out.shape[2:], mode='bilinear', align_corners=False)
        T3_resized = T3_mag_resized * torch.exp(1j * T3_phase_resized)

        S4 = torch.cat((deconv4_out, T3_resized), dim=1)
        S4 = self.conv_block4(S4)

        deconv3_out = self.deconv3(S4)

        # Handle T2 interpolation similarly
        T2_mag = torch.abs(T2)
        T2_phase = torch.angle(T2)
        T2_mag_resized = F.interpolate(T2_mag, size=deconv3_out.shape[2:], mode='bilinear', align_corners=False)
        T2_phase_resized = F.interpolate(T2_phase, size=deconv3_out.shape[2:], mode='bilinear', align_corners=False)
        T2_resized = T2_mag_resized * torch.exp(1j * T2_phase_resized)

        S43 = torch.cat((deconv3_out, T2_resized), dim=1)
        complex_out = self.conv_block3(S43)

        # Keep detection output as complex for full complex processing in detection head
        # Remove the conversion to real - let detection network handle complex input directly

        return {
            'detection': complex_out,    # Complex output for detection (preserves phase info)
            'segmentation': complex_out  # Complex output for segmentation
        }


class ComplexFFTRadNet(nn.Module):
    """Complex-valued FFTRadNet"""
    def __init__(self, mimo_layer, channels, blocks, regression_layer=2, detection_head=True, segmentation_head=True):
        super(ComplexFFTRadNet, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.FPN = ComplexFPN_BackBone(num_block=blocks, channels=channels, block_expansion=4,
                                      mimo_layer=mimo_layer, use_bn=True)
        self.RA_decoder = ComplexRangeAngle_Decoder()

        if(self.detection_head):
            self.detection_header = ComplexDetection_Header(input_angle_size=channels[3]*4, reg_layer=regression_layer)

        if(self.segmentation_head):
            self.freespace = ComplexFreespace()
            
    def quantize_weights(self):
        # Quantize weights of all quantized complex layers
        for name, module in self.named_modules():
            if isinstance(module, (QuantizedComplexConv2d, QuantizedComplexConvTranspose2d)):
                print(f"Quantizing weights of {name}")
                module.quantize_weights()

    def forward(self, x):
        out = {'Detection': [], 'Segmentation': []}

        features = self.FPN(x)
        RA_outputs = self.RA_decoder(features)  # Now returns dict with detection and segmentation outputs

        if(self.detection_head):
            # Use real output for detection
            out['Detection'] = self.detection_header(RA_outputs['detection'])

        if(self.segmentation_head):
            # Use complex output for segmentation, interpolate it first
            complex_seg_features = RA_outputs['segmentation']
            # Complex interpolation
            seg_mag = torch.abs(complex_seg_features)
            seg_phase = torch.angle(complex_seg_features)
            seg_mag_interp = F.interpolate(seg_mag, (256, 224), mode='bilinear', align_corners=False)
            seg_phase_interp = F.interpolate(seg_phase, (256, 224), mode='bilinear', align_corners=False)
            Y_complex = seg_mag_interp * torch.exp(1j * seg_phase_interp)

            out['Segmentation'] = self.freespace(Y_complex)

        return out