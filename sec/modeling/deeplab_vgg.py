import torch
import torch.nn as nn

__all__ = [
    'build_deeplab_vgg16_backbone',
]

from detectron2.layers import ShapeSpec

from detectron2.modeling import BACKBONE_REGISTRY, Backbone


arch_mapping = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
              512, 512, 512, 'N', 512, 512, 512, 'N', 'A'],
}


class DeepLabVGG(Backbone):
    def __init__(self, depth=16, out_features=None):
        super(DeepLabVGG, self).__init__()
        vgg_arch = "VGG" + str(depth)
        self.features = make_layers(arch_mapping[vgg_arch])

        self.dim_out = 512
        self.spatial_scale = 1. / 8.

        if out_features is None:
            out_features = ["conv5"]
        self._out_features = out_features
        assert len(self._out_features)

    def forward(self, x):
        x = self.features(x)
        return {"conv5": x}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=512, stride=8
            )
            for name in self._out_features
        }


def make_layers(cfg):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i >= 14:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


@BACKBONE_REGISTRY.register()
def build_deeplab_vgg16_backbone(cfg, input_shape):
    """
    Create a VGG16 instance from config.

    Returns:
        VGG16: a :class:`VGG16` instance.
    """
    return DeepLabVGG()
