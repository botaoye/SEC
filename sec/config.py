# -*- coding = utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_sec_config(cfg):
    """
    Add config for sec.
    """
    _C = cfg

    _C.MODEL.VGG = CN()
    _C.MODEL.VGG.FREEZE_AT = 2
