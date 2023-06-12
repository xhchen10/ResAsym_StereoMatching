from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models.PSMNet_submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, is_lightfield, maxdisp):
        super(PSMNet, self).__init__()
        self.is_light_field = is_lightfield
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(32, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 3, 1, 1))

        self.dres2 = hourglass(16)

        self.classif1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        max_disp = self.maxdisp // 4
        if self.is_light_field:
            disp_range = [i for i in range(-max_disp, max_disp)]
        else:
            disp_range = [i for i in range(max_disp)]
        # matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2,
                                          len(disp_range), refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).cuda()

        for i, d in enumerate(disp_range):
            if d < 0:
                cost[:, :refimg_fea.size()[1], i, :, :d] = refimg_fea[:, :, :, :d]
                cost[:, refimg_fea.size()[1]:, i, :, :d] = targetimg_fea[:, :, :, -d:]
            elif d == 0:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
            else: # d > 0
                cost[:, :refimg_fea.size()[1], i, :, d:] = refimg_fea[:, :, :, d:]
                cost[:, refimg_fea.size()[1]:, i, :, d:] = targetimg_fea[:, :, :, :-d]
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        cost1 = self.classif1(out1)

        if self.is_light_field:
            cost1 = F.upsample(cost1, [2*self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        else:
            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost1 = torch.squeeze(cost1, 1)
        pred1 = F.softmax(cost1, dim=1)

        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        if self.is_light_field:
            pred1 = disparityregression(self.maxdisp)(pred1)
        else:
            pred1 = disparityregression_stereo(self.maxdisp)(pred1)

        return [pred1]


if __name__ == '__main__':
    model = PSMNet(False, 96).cuda()
    import torch
    inputL = torch.FloatTensor(1, 3, 400, 352).zero_().cuda()
    inputR = torch.FloatTensor(1, 3, 400, 352).zero_().cuda()
    for i in range(100):
        out = model(inputL, inputR)
        for o in out:
            print(o.shape)