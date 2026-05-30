from statistics import mode
from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4
from .imagenet import ImageEncoder, ImageUpSample, OneImageUpSample, ImageUpSampleNearest, OneImageUpSampleDummy, DepthEncoder
from .PointViT import PointTransformer

def build_backbone(config):

    return ResNetFPN_8_2(config['resnetfpn'], 3 if config['rgb'] else 1)
       