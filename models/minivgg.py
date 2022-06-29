###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

import torch
import torch.nn as nn
import ai8x
import distiller.apputils as apputils

'''
Simple model for Cats and Dogs
'''

class CatsAndDogsClassifier(nn.Module):
    def __init__(self, num_classes=2,num_channels=3,dimensions=(128,128),bias=True,**kwargs):
        super().__init__()

        # 3x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = ai8x.FusedConv2dReLU(3, 8, 3, stride=1, padding=1,
                                            bias=False, **kwargs)
        
        # 8x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv2 = ai8x.FusedConv2dReLU(8, 8, 3, stride=1, padding=1,
                                            bias=False, **kwargs)
        
        # 8x128x128 --> 8x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(8, 16, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=False, **kwargs)
        bias=True
        # 16x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv4 = ai8x.FusedConv2dBNReLU(16, 16, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        
        # 16x64x64 --> 16x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 32x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv6 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 32x32x32 --> 32x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # 64x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv8 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 64x16x16 --> 64x8x8 (padding by 1 so same dimension)
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
                                                   
        
        # 64x8x8 --> 64x4x4 (padding by 1 so same dimension)
        self.conv10 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # flatten to fully connected layer
        self.fc1 = ai8x.FusedLinearReLU(32*4*4, 64, bias=True, **kwargs)
        self.do1 = torch.nn.Dropout(p=0.5)
        self.fc2 = ai8x.Linear(64, 2, bias=True, wide=True, **kwargs)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.do1(x)
        x = self.fc2(x)

        return x


def catdognet(pretrained=False, **kwargs):
    """
    Constructs a small CNN model for binary classification.
    """
    assert not pretrained
    return CatsAndDogsClassifier(**kwargs)


class Imagenet10Classifier(nn.Module):
    def __init__(self, num_classes=2,num_channels=3,dimensions=(128,128),bias=True,**kwargs):
        super().__init__()

        # 3x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv1 = ai8x.FusedConv2dReLU(3, 8, 3, stride=1, padding=1,
                                            bias=False, **kwargs)
        
        # 8x128x128 --> 8x128x128 (padding by 1 so same dimension)
        self.conv2 = ai8x.FusedConv2dReLU(8, 8, 3, stride=1, padding=1,
                                            bias=False, **kwargs)
        
        # 8x128x128 --> 8x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(8, 16, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=False, **kwargs)
        bias=True
        # 16x64x64 --> 16x64x64 (padding by 1 so same dimension)
        self.conv4 = ai8x.FusedConv2dBNReLU(16, 16, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        
        # 16x64x64 --> 16x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 32x32x32 --> 32x32x32 (padding by 1 so same dimension)
        self.conv6 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 32x32x32 --> 32x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # 64x16x16 --> 64x16x16 (padding by 1 so same dimension)
        self.conv8 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                                                   bias=True,batchnorm='Affine', **kwargs)
        
        # 64x16x16 --> 64x8x8 (padding by 1 so same dimension)
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
                                                   
        
        # 64x8x8 --> 64x4x4 (padding by 1 so same dimension)
        self.conv10 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 3, stride=1, padding=1, pool_size=2, pool_stride=2,
                                                   bias=True,batchnorm='Affine',**kwargs)
        
        # flatten to fully connected layer
        self.fc1 = ai8x.FusedLinearReLU(32*4*4, 128, bias=True, **kwargs)
        self.do1 = torch.nn.Dropout(p=0.5)
        self.fc2 = ai8x.Linear(128, 10, bias=True, wide=True, **kwargs)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.do1(x)
        x = self.fc2(x)

        return x


def imagenet10(pretrained=False, **kwargs):
    """
    Constructs a small CNN model for binary classification.
    """
    assert not pretrained
    return Imagenet10Classifier(**kwargs)




class CatsAndDogsDCD(nn.Module):
    def __init__(self, num_classes=2,num_channels=3,dimensions=(128,128),bias=True,**kwargs):
        super().__init__()
        
        # flatten to fully connected layer
        self.fc1 = ai8x.FusedLinearReLU(128,64, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(64, 4, bias=True, wide=True, **kwargs)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class OfficeDCD(nn.Module):
    def __init__(self, num_classes=2,num_channels=3,dimensions=(128,128),bias=True,**kwargs):
        super().__init__()
        
        # flatten to fully connected layer
        self.fc1 = ai8x.FusedLinearReLU(128,64, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(64, 4, bias=True, wide=True, **kwargs)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class OfficeClassifier(nn.Module):
    def __init__(self, num_classes=2,num_channels=3,dimensions=(128,128),bias=True,device='cpu',**kwargs):
        super().__init__()
        
        load_model_path = "jupyter_logging/imagenet10_long-train___2022.06.28-172810/imagenet10_qat_best.pth.tar"

        self.feature_extractor = Imagenet10Classifier()                       
        checkpoint = torch.load(load_model_path, map_location=lambda storage, loc: storage)
        ai8x.fuse_bn_layers(self.feature_extractor)
        self.feature_extractor = apputils.load_lean_checkpoint(self.feature_extractor, load_model_path, model_device=device)
        ai8x.update_model(self.feature_extractor)
        
        # freeze the weights except for last conv and fc
        # ct = 0
        # for child in self.feature_extractor.children():
        #     ct += 1
        #     if ct < 8:
        #         for param in child.parameters():
        #             param.requires_grad = False
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
            
        # retrain the last layer to detect a bounding box and classes
        self.feature_extractor.fc2 = ai8x.FusedLinearReLU(128, 64, bias=True, **kwargs)
        self.feature_extractor.fc3 = ai8x.Linear(64, 5, bias=True, wide=True, **kwargs)

        self.do1 = torch.nn.Dropout(p=0.5)
            
        # add a fully connected layer for bounding box detection after the conv10
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.conv2(x)
        x = self.feature_extractor.conv3(x)
        x = self.feature_extractor.conv4(x)
        x = self.feature_extractor.conv5(x)
        x = self.feature_extractor.conv6(x)
        x = self.feature_extractor.conv7(x)
        x = self.feature_extractor.conv8(x)
        x = self.feature_extractor.conv9(x)
        x = self.feature_extractor.conv10(x)
        x = x.view(x.size(0), -1)
        
        # output layers
        x1 = self.feature_extractor.fc1(x)
        #x1 = self.do1(x1)
        x1 = self.feature_extractor.fc2(x1)
        x1 = self.feature_extractor.fc3(x1)

        return x1


def catdogdcdnet(pretrained=False, **kwargs):
    """
    Constructs a small CNN model for binary classification.
    """
    assert not pretrained
    return CatsAndDogsDCD(**kwargs)

models = [
    {
        'name': 'catdognet',
        'min_input': 1,
        'dim': 2,
    }
]