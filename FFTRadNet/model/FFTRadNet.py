import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Detection_Header(nn.Module):

    def __init__(self, use_bn=True,reg_layer=2,input_angle_size=0):
        super(Detection_Header, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if(self.input_angle_size==224):
            self.conv1 = conv3x3(256, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==112):  # Added support for reduced model
            self.conv1 = conv3x3(128, 72, bias=bias)  # Input from reduced RA decoder: 128 channels
            self.bn1 = nn.BatchNorm2d(72)
            self.conv2 = conv3x3(72, 48, bias=bias)  # Halved from 96
            self.bn2 = nn.BatchNorm2d(48)
        elif(self.input_angle_size==448):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias,stride=(1,2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')
            return

        # Adjust conv3 and conv4 based on input angle size
        if(self.input_angle_size==112):  # Reduced model
            self.conv3 = conv3x3(48, 48, bias=bias)  # Halved from 96
            self.bn3 = nn.BatchNorm2d(48)
            self.conv4 = conv3x3(48, 48, bias=bias)  # Halved from 96  
            self.bn4 = nn.BatchNorm2d(48)
            self.clshead = conv3x3(48, 1, bias=True)  # Use 48 instead of 96
            self.reghead = conv3x3(48, reg_layer, bias=True)  # Use 48 instead of 96
        else:  # Original sizes for other configurations
            self.conv3 = conv3x3(96, 96, bias=bias)
            self.bn3 = nn.BatchNorm2d(96)
            self.conv4 = conv3x3(96, 96, bias=bias)
            self.bn4 = nn.BatchNorm2d(96)
            self.clshead = conv3x3(96, 1, bias=True)
            self.reghead = conv3x3(96, reg_layer, bias=True)
            
    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return torch.cat([cls, reg], dim=1)


class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None,expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

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

        out = F.relu(residual + out)
        return out

class MIMO_PreEncoder(nn.Module):
    def __init__(self, in_layer,out_layer,kernel_size=(1,12),dilation=(1,16),use_bn = False):
        super(MIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn

        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size, 
                              stride=(1, 1), padding=0,dilation=dilation, bias= (not use_bn) )
     
        self.bn = nn.BatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna/2)

    def forward(self,x):
        width = x.shape[-1]
        x = torch.cat([x[...,-self.padding:],x,x[...,:self.padding]],axis=3)
        x = self.conv(x)
        x = x[...,int(x.shape[-1]/2-width/2):int(x.shape[-1]/2+width/2)]

        if self.use_bn:
            x = self.bn(x)
        return x

class FPN_BackBone(nn.Module):

    def __init__(self, num_block,channels,block_expansion,mimo_layer,use_bn=True):
        super(FPN_BackBone, self).__init__()

        self.block_expansion = block_expansion
        self.use_bn = use_bn

        # pre processing block to reorganize MIMO channels
        self.pre_enc = MIMO_PreEncoder(32,mimo_layer,
                                        kernel_size=(1,NbTxAntenna),
                                        dilation=(1,NbRxAntenna),
                                        use_bn = True)

        self.in_planes = mimo_layer

        self.conv = conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck, planes=channels[3], num_blocks=num_block[3])
                                       
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
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample,expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1,expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
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

class RangeAngle_Decoder(nn.Module):
    def __init__(self, channels):
        super(RangeAngle_Decoder, self).__init__()
        
        # Check if this is the original configuration
        is_original = (channels == [32, 40, 48, 56])
        
        # Calculate actual dimensions based on backbone output
        c2_channels = channels[1] * 4  # x2 channels
        c3_channels = channels[2] * 4  # x3 channels  
        c4_channels = channels[3] * 4  # x4 channels
        
        # The spatial dimensions after backbone are fixed regardless of channel config:
        # x4: [B, c4_channels, 16, 14] -> after transpose(1,3): [B, 14, 16, c4_channels] 
        # x3: [B, c3_channels, 32, 28] -> after L3+transpose: [B, 28, 32, L3_output]
        # x2: [B, c2_channels, 64, 56] -> after L2+transpose: [B, 56, 64, L2_output]
        
        width_x4_after_transpose = 14  # x4 width becomes channel dim
        width_x3_after_transpose = 28  # x3 width becomes channel dim
        width_x2_after_transpose = 56  # x2 width becomes channel dim
        
        if is_original:
            # Original configuration values
            l3_output_channels = 224
            l2_output_channels = 224
            conv_block4_output = 128
            conv_block3_output = 256
        else:
            # Reduced configuration - scale down
            l3_output_channels = 112  # 224/2
            l2_output_channels = 112  # 224/2  
            conv_block4_output = 64   # 128/2
            conv_block3_output = 128  # 256/2
        
        # Top-down layers  
        self.deconv4 = nn.ConvTranspose2d(width_x4_after_transpose, width_x4_after_transpose, 
                                         kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        
        # Concatenation: deconv4 + L3_after_transpose  
        self.conv_block4 = BasicBlock(width_x4_after_transpose + width_x3_after_transpose, conv_block4_output)
        self.deconv3 = nn.ConvTranspose2d(conv_block4_output, conv_block4_output, 
                                         kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        
        # Concatenation: deconv3 + L2_after_transpose
        self.conv_block3 = BasicBlock(conv_block4_output + width_x2_after_transpose, conv_block3_output)

        # Projection layers
        self.L3  = nn.Conv2d(c3_channels, l3_output_channels, kernel_size=1, stride=1, padding=0)
        self.L2  = nn.Conv2d(c2_channels, l2_output_channels, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self,features):

        T4 = features['x4'].transpose(1, 3) 
        T3 = self.L3(features['x3']).transpose(1, 3)
        T2 = self.L2(features['x2']).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)
        
        return out


class FFTRadNet(nn.Module):
    def __init__(self,mimo_layer,channels,blocks,regression_layer = 2, detection_head=True,segmentation_head=True):
        super(FFTRadNet, self).__init__()
    
        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.FPN = FPN_BackBone(num_block=blocks,channels=channels,block_expansion=4, mimo_layer = mimo_layer,use_bn = True)
        self.RA_decoder = RangeAngle_Decoder(channels)
        
        if(self.detection_head):
            self.detection_header = Detection_Header(input_angle_size=channels[3]*4,reg_layer=regression_layer)

        if(self.segmentation_head):
            # Make segmentation head configurable based on RA decoder output
            if channels == [32, 40, 48, 56]:  # Original
                self.freespace = nn.Sequential(BasicBlock(256,128),BasicBlock(128,64),nn.Conv2d(64, 1, kernel_size=1))
            else:  # Reduced
                self.freespace = nn.Sequential(BasicBlock(128,64),BasicBlock(64,32),nn.Conv2d(32, 1, kernel_size=1))

    def forward(self,x):
                       
        out = {'Detection':[],'Segmentation':[]}
        
        features= self.FPN(x)
        RA = self.RA_decoder(features)

        if(self.detection_head):
            out['Detection'] = self.detection_header(RA)

        if(self.segmentation_head):
            Y =  F.interpolate(RA, (256, 224))
            out['Segmentation'] = self.freespace(Y)
        
        return out