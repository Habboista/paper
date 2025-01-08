# coding: utf-8

'''
Author: Ke Xian
Email: kexian@hust.edu.cn
Date: 2019/04/09
'''

import os

import torch
import torchvision
import torchvision.transforms.v2 as v2
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#from src.model.resnet import *
from src.model.layers import *

class Decoder(nn.Module):
    def __init__(
        self,
        input_size = (172, 576),
        inchannels = [256, 512, 1024, 2048],
        midchannels = [256, 256, 256, 512],
        sizes=[(86, 290), (43, 144), (22, 72), (11, 36)],
        outchannels = 1,
    ):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.sizes = sizes
        self.outchannels = outchannels

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(size=sizes[3], mode='bilinear', align_corners=ALIGN_CORNERS)

        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels=self.midchannels[2], size=sizes[2])
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels=self.midchannels[1], size=sizes[1])
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels=self.midchannels[0], size=sizes[0])

        self.outconv = AO(inchannels=self.inchannels[0], outchannels=self.outchannels, size=self.input_size)

        self._init_params()

    def _init_params(self):
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
            elif isinstance(m, nn.BatchNorm2d): #nn.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, features):
        _,_,h,w = features[3].size()
        x = self.conv(features[3])
        x = self.conv1(x)
        #x = self.upsample(x)
        x = nn.functional.interpolate(x, size=self.sizes[3], mode='bilinear', align_corners=ALIGN_CORNERS)

        x = self.ffm2(features[2], x)
        x = self.ffm1(features[1], x)
        x = self.ffm0(features[0], x)

        #-----------------------------------------
        x = self.outconv(x)

        return x

from torchvision.models.resnet import resnet50
from torchvision.models.resnet import ResNet50_Weights

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(ResNet50_Weights.IMAGENET1K_V1)

    def forward(self, x):
        features = []

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        features.append(x)
        x = self.backbone.layer2(x)
        features.append(x)
        x = self.backbone.layer3(x)
        features.append(x)
        x = self.backbone.layer4(x)
        features.append(x)

        return features

class DepthNet(nn.Module):
    #__factory = {
    #    18: resnet18,
    #    34: resnet34,
    #    50: resnet50,
    #    101: resnet101,
    #    152: resnet152
    #}
    def __init__(
        self,
        backbone='resnet',
        depth=50,
        pretrained=True,
        outchannels=1,
        input_size=(172, 576),
        share_encoder_for_confidence_prediction: bool = False,
    ):
        super(DepthNet, self).__init__()
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.backbone = backbone
        self.depth = depth
        self.pretrained = pretrained
        self.outchannels = outchannels
        self.input_size = input_size
        self.share_encoder_for_confidence_prediction = share_encoder_for_confidence_prediction

        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

        # Build model
        #if self.depth not in DepthNet.__factory:
        #    raise KeyError("Unsupported depth:", self.depth)
        self.encoder = Encoder() #DepthNet.__factory[depth](pretrained=pretrained)
        if not share_encoder_for_confidence_prediction:
            self.confidence_encoder = Encoder() #DepthNet.__factory[depth](pretrained=pretrained)
        else:
            self.confidence_encoder = None

        sample_input = torch.randn((1, 3, *input_size))
        features: list[torch.Tensor] = self.encoder(sample_input)
        self.inchannels = [
            features[0].shape[1],
            features[1].shape[1],
            features[2].shape[1],
            features[3].shape[1],
        ]
        print(f"DepthNet with ResNet{depth} and input size {tuple(input_size)} produces the following feature pyramid:")
        for f in features:
            print(tuple(f.shape))
        print()
        self.midchannels = [
            self.inchannels[0],
            self.inchannels[0],
            self.inchannels[0],
            self.inchannels[1],
        ]
        self.sizes = [
            (features[0].shape[2] * 2, features[0].shape[3] * 2),
            features[0].shape[-2:],
            features[1].shape[-2:],
            features[2].shape[-2:],
        ]
        self.decoder = Decoder(
            input_size=self.input_size,
            inchannels=self.inchannels,
            midchannels=self.midchannels,
            sizes=self.sizes,
            outchannels=self.outchannels,
        )

    def forward(self, x):
        x = self.normalize(x)
        
        if self.share_encoder_for_confidence_prediction:
            x = self.encoder(x)
            confidence = self.confidence_head(x[-1])
        else:
            confidence = self.confidence_head(self.confidence_encoder(x)[-1])
            x = self.encoder(x)

        x = self.decoder(x)

        return x, confidence