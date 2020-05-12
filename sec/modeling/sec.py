# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, SEM_SEG_HEADS_REGISTRY, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.events import get_event_storage
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry


from sec.modeling.sec_layers import seed_loss_layer, expand_loss_layer, crf_layer, constrain_loss_layer, softmax_layer
from sec.torch_utils import vis_segmentation


@META_ARCH_REGISTRY.register()
class SEC(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.vis_period = cfg.VIS_PERIOD

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model, used in inference.
                     See :meth:`postprocess` for details.

        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor of the output resolution that represents the
              per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None

        pred_mask = self.sem_seg_head(features)

        if self.training:
            # TODO figure out why float is necessary
            labels = [x["label"].to(self.device) for x in batched_inputs]
            labels = torch.stack(labels, dim=0).to(torch.float)

            # get downscaled images
            downscaled_images = [x["image"] for x in batched_inputs]
            downscaled_images = ImageList.from_tensors(downscaled_images, self.backbone.size_divisibility)
            downscaled_images = F.interpolate(downscaled_images.tensor.float(),
                                              size=(pred_mask.shape[-2], pred_mask.shape[-1]),
                                              mode='bilinear', align_corners=False)

            # # from bgr to rgb
            # downscaled_images = downscaled_images.numpy()
            # downscaled_images = downscaled_images[:, ::-1, :, :]
            # downscaled_images = torch.from_numpy(np.ascontiguousarray(downscaled_images))
            # from bchw to bhwc
            downscaled_images = downscaled_images.permute(0, 2, 3, 1).contiguous().to(torch.uint8)
            fc8_sec_softmax = softmax_layer(pred_mask)

            # vis label
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    for i in range(labels.shape[0]):
                        vis_segmentation(targets[i].cpu().numpy().astype("int64"), np.ascontiguousarray(downscaled_images[i]))

            # cal loss
            loss_s = seed_loss_layer(fc8_sec_softmax, targets)
            loss_e = expand_loss_layer(fc8_sec_softmax, labels,
                                       fc8_sec_softmax.shape[-2], fc8_sec_softmax.shape[-1],
                                       self.num_classes + 1)
            crf_result = crf_layer(fc8_sec_softmax, downscaled_images, 10)
            loss_c = constrain_loss_layer(fc8_sec_softmax, crf_result)
            return {"loss_s": loss_s, "loss_e": loss_e, "loss_c": loss_c}
            # return {"loss": torch.tensor(0.0, requires_grad=True)}

        processed_results = []
        for result, input_per_image, image_size in zip(pred_mask, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results


@SEM_SEG_HEADS_REGISTRY.register()
class SECSemSegHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                # if feature_strides[in_feature] != self.common_stride:
                #     head_ops.append(
                #         nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                #     )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        # x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        return x


@SEM_SEG_HEADS_REGISTRY.register()
class DeepLabLargeFOVHead(nn.Module):
    """
    A semantic segmentation head described DeepLabLargeFOV
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=12, dilation=12)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout2d(0.5)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(1024, num_classes + 1, kernel_size=1)

        weight_init.c2_msra_fill(self.fc6)
        weight_init.c2_msra_fill(self.fc7)
        weight_init.c2_msra_fill(self.fc8)

    def forward(self, features):
        x = features["conv5"]
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)

        return x
