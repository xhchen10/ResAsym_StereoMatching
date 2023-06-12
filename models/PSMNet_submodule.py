from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Module):
    def __init__(self, max_disp):
        super(disparityregression, self).__init__()
        disp_range = [i for i in range(-max_disp, max_disp)]
        self.disp = Variable(torch.Tensor(np.reshape(np.array(disp_range), [1, len(disp_range), 1, 1])).cuda(),
                             requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out


class disparityregression_stereo(nn.Module):
    def __init__(self, max_disp):
        super(disparityregression_stereo, self).__init__()
        disp_range = [i for i in range(max_disp)]
        self.disp = Variable(torch.Tensor(np.reshape(np.array(disp_range), [1, len(disp_range), 1, 1])).cuda(),
                             requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 16
        self.firstconv = nn.Sequential(convbn(3, 16, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(16, 16, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(16, 16, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 16, 2, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 32, 8, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 64, 2, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(64, 16, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(64, 16, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 16, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
