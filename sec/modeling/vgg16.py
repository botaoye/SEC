import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec

from detectron2.modeling import Backbone, BACKBONE_REGISTRY


class VGG16(Backbone):
    def __init__(self, freeze_at=0, out_features=None):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True))

        self.dim_out = 512
        self.spatial_scale = 1. / 8.

        if out_features is None:
            out_features = ["conv5"]
        self._out_features = out_features
        self._freeze_at = freeze_at
        assert len(self._out_features)

        self._init_modules()

    def _init_modules(self):
        assert self._freeze_at in [0, 2, 3, 4, 5]
        for i in range(1, self._freeze_at + 1):
            freeze_params(getattr(self, 'conv%d' % i))

    def train(self, mode=True):
        # Override
        self.training = mode
        for i in range(self._freeze_at + 1, 6):
            getattr(self, 'conv%d' % i).train(mode)

    def forward(self, x):
        for i in range(1, 6):
            x = getattr(self, 'conv%d' % i)(x)
        return {"conv5": x}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=512, stride=8
            )
            for name in self._out_features
        }


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


@BACKBONE_REGISTRY.register()
def build_vgg16_backbone(cfg, input_shape):
    """
    Create a VGG16 instance from config.

    Returns:
        VGG16: a :class:`VGG16` instance.
    """
    # fmt: off
    freez_at = cfg.MODEL.VGG.FREEZE_AT
    # fmt: on
    return VGG16(freez_at)
