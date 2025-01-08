# -*- coding: utf-8 -*-
'''
Author: Ke Xian
Email: kexian@hust.edu.cn
Date: 2019/04/09
'''

import torch
import torch.nn as nn
import torch.nn.init as init

# ==============================================================================================================
ALIGN_CORNERS = False

class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels

        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True)
        # nn.BatchNorm2d
        self.conv_branch = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(num_features=self.mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.mid, out_channels= self.mid, kernel_size=3, padding=1, stride=1, bias=True),
        )
        self.relu = nn.ReLU(inplace=True)

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.relu(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                #init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                #init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  #nn.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, size):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.size = size

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=ALIGN_CORNERS)

        self.init_params()

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        x = self.ftb2(x)
        x = nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=ALIGN_CORNERS)
        #x = self.upsample(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                #init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                #init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #nn.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, size):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.size = size

        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels//2, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(num_features=self.inchannels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.inchannels//2, out_channels=self.outchannels, kernel_size=3, padding=1, stride=1, bias=True),
            #nn.Upsample(size=self.size, mode='bilinear', align_corners=ALIGN_CORNERS),
        )

        self.init_params()

    def forward(self, x):
        x = self.adapt_conv(x)
        x = nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=ALIGN_CORNERS)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                #init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                #init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #nn.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)