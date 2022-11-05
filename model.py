#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import warnings
warnings.simplefilter('ignore')

import torchvision.models as M
import torchvision.models.quantization as QM

MODELS = [
#  'googlenet',

#  'inception_v3',

#  'mobilenet_v2',
#  'mobilenet_v3_small',
#  'mobilenet_v3_large',

  'resnet18',
#  'resnet34',
  'resnet50',
#  'resnet101',
#  'resnet152',
  'resnext50_32x4d',
#  'resnext101_32x8d',
#  'resnext101_64x4d',
  'wide_resnet50_2',
#  'wide_resnet101_2',

#  'shufflenet_v2_x0_5',
#  'shufflenet_v2_x1_0',
#  'shufflenet_v2_x1_5',
#  'shufflenet_v2_x2_0',
]


def get_model(name, qt=False):
  if qt:
    model = getattr(QM, name)(pretrained=True, quantize=True)
  else:
    model = getattr(M, name)(pretrained=True)
  
  return model
