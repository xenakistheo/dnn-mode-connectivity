"""
ResNet models from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
adapted to format of PreResNets
"""

import math
import torch.nn as nn

import curves



__all__ = ['ResNet8', 'ResNet20', 'ResNet26', 'ResNet32', 'ResNet38', 'ResNet44', 'ResNet56', 'ResNet62', 'ResNet101', 'ResNet116',
           'ResNet164', 'ResNet272', 'ResNet326', 'ResNet650']




def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride,
                         padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv2 = conv3x3(planes, planes)
        # self.downsample = downsample
        # self.stride = stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x

        # out = self.bn1(x)
        # out = self.relu(out)
        # out = self.conv1(out)

        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.conv2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual

        # return out
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



#Changed
class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, fix_points, stride=1, downsample=None):
        super(BasicBlockCurve, self).__init__()
        self.conv1 = conv3x3curve(inplanes, planes, stride=stride, fix_points=fix_points)   
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = conv3x3curve(planes, planes, fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x, coeffs_t):
        residual = x

        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)
        
        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)
        
        if self.downsample is not None:
            residual = self.downsample(x, coeffs_t)

        out += residual
        out = self.relu(out)

        return out

#Changed
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += residual
        out = self.relu(out)

        return out

#Finished
class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, fix_points, stride=1, downsample=None):
        super(BottleneckCurve, self).__init__()
        self.conv1 = curves.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                                   fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv3 = curves.Conv2d(planes, planes * 4, kernel_size=1, bias=False,
                                   fix_points=fix_points)
        self.bn3 = curves.BatchNorm2d(planes*4, fix_points=fix_points)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        residual = x

        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)
        
        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)
        out = self.relu(out)
        
        out = self.conv3(out, coeffs_t)
        out = self.bn3(out, coeffs_t)
    
        if self.downsample is not None:
            residual = self.downsample(x, coeffs_t)

        out += residual
        out = self.relu(out)

        return out

#Changed
class ResNetBase(nn.Module):

    def __init__(self, num_classes, depth=110):
        super(ResNetBase, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class DownsampleCurve(nn.Module):
    """
    Conv1x1 + BN shortcut for curve ResNets.
    Expects to be called as downsample(x, coeffs_t).
    """
    def __init__(self, inplanes, outplanes, stride, fix_points):
        super().__init__()
        self.conv = curves.Conv2d(
            inplanes, outplanes,
            kernel_size=1, stride=stride, bias=False,
            fix_points=fix_points,
        )
        self.bn = curves.BatchNorm2d(outplanes, fix_points=fix_points)

    def forward(self, x, coeffs_t):
        x = self.conv(x, coeffs_t)
        x = self.bn(x, coeffs_t)
        return x


class ResNetCurve(nn.Module):

    def __init__(self, num_classes, fix_points, depth=110):
        super(ResNetCurve, self).__init__()

        # depth logic: same as your CIFAR ResNetBase
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = BottleneckCurve
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlockCurve

        self.inplanes = 16

        # stem: conv1 + bn1 + relu (curve versions of conv/bn)
        self.conv1 = curves.Conv2d(
            3, 16,
            kernel_size=3, padding=1, bias=False,
            fix_points=fix_points,
        )
        self.bn1 = curves.BatchNorm2d(16, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)

        # residual stages
        self.layer1 = self._make_layer(block, 16, n, fix_points=fix_points)
        self.layer2 = self._make_layer(block, 32, n, stride=2, fix_points=fix_points)
        self.layer3 = self._make_layer(block, 64, n, stride=2, fix_points=fix_points)

        # global pooling + linear head
        self.avgpool = nn.AvgPool2d(8)
        self.fc = curves.Linear(64 * block.expansion, num_classes, fix_points=fix_points)

        # weight init (same style as your original curve code)
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, f'weight_{i}').data.normal_(0, math.sqrt(2. / fan_out))
            elif isinstance(m, curves.BatchNorm2d):
                for i in range(m.num_bends):
                    getattr(m, f'weight_{i}').data.fill_(1)
                    getattr(m, f'bias_{i}').data.zero_()

    def _make_layer(self, block, planes, blocks, fix_points, stride=1):
        """
        Build one residual stage (layer1/layer2/layer3) as a ModuleList of curve blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleCurve(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
                fix_points=fix_points,
            )

        layers = []
        # first block in the stage may have stride > 1 and a downsample
        layers.append(
            block(
                self.inplanes,
                planes,
                fix_points=fix_points,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion

        # remaining blocks in the stage
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    fix_points=fix_points,
                )
            )

        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        # stem
        x = self.conv1(x, coeffs_t)
        x = self.bn1(x, coeffs_t)
        x = self.relu(x)

        # residual stages
        for block in self.layer1:  # 32x32
            x = block(x, coeffs_t)
        for block in self.layer2:  # 16x16
            x = block(x, coeffs_t)
        for block in self.layer3:  # 8x8
            x = block(x, coeffs_t)

        # no final BN+ReLU here (post-activation blocks already end with ReLU)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, coeffs_t)

        return x


class ResNet20:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 20}

class ResNet32:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 32}

class ResNet44:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 44}

class ResNet56:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 56}

class ResNet101:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 101}

class ResNet164:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 164}

class ResNet272:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 272}

class ResNet326:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 326}

class ResNet650:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 650}

#Use depth = 6n + 2   with n = number of blocks per stage

#add 8, 26, 38, 62, 116
class ResNet8:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 8}

class ResNet26:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 26}     

class ResNet38:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 38}

class ResNet62:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 62}

class ResNet65:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 65}  

class ResNet116:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'depth': 116}
