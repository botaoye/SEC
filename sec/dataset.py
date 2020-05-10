# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import pickle
import numpy as np
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

__all__ = ["register_pascal_voc"]

# fmt: off
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
# fmt: on


def load_voc_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    with PathManager.open(os.path.join(dirname, "ImageSets", "Segmentation", split + ".txt")) as f:
        seg_fileids = np.loadtxt(f, dtype=np.str)

    cls_labels_file = os.path.join(dirname, "cls_labels.npy")
    cls_labels_dict = np.load(cls_labels_file).item()

    dicts = []
    count = 0
    for fileid in fileids:
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        if fileid in seg_fileids:
            seg_file = os.path.join(dirname, "SegmentationClass", fileid + ".png")
        else:
            seg_file = "no_seg_label"

        im = Image.open(jpeg_file)
        width, height = im.size

        localization_cues_path = os.path.join(dirname, "weak-localization/localization_cues.pickle")
        with open(localization_cues_path, "rb") as f:
            localization_cues = pickle.load(f)

        r = {
            "file_name": jpeg_file,
            "seg_file": seg_file,
            "image_id": fileid,
            "height": height,
            "width": width,
            "label": cls_labels_dict[fileid],
            "localization_cues": localization_cues,
        }
        dicts.append(r)
        count += 1
    # print(count)
    return dicts


def register_pascal_voc(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )


register_pascal_voc("ws_voc_2012_train", "./datasets/VOC2012", "train", 2012)
register_pascal_voc("ws_voc_2012_val", "./datasets/VOC2012", "val", 2012)
