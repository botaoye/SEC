# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.structures import Instances, BoxMode, Boxes
from detectron2.data import detection_utils as utils, MetadataCatalog
from detectron2.data import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.training_datasets = cfg.DATASETS.TRAIN
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        # self.load_proposals = True
        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            transform_proposals_mataws(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        # transfer label to Pytorch tensor
        # dataset_dict["label"] = torch.tensor(dataset_dict["label"], dtype=torch.int64)

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                obj
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances_metaws(annos, image_shape)
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt

        # add localization cues
        # TODO change hard code of h, w here
        localization_cues = MetadataCatalog.get(self.training_datasets[0]).localization_cues
        label, sem_seg_gt = get_localization_cues(localization_cues,
                                                  str(dataset_dict["image_order"]),
                                                  41,
                                                  41,
                                                  dataset_dict["label"].shape[-1])
        dataset_dict["label"] = torch.tensor(label, dtype=torch.int64)
        # sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("float"))
        dataset_dict["sem_seg"] = sem_seg_gt

        return dataset_dict


def get_localization_cues(localization_cues, image_id, height, width, num_class):
    label_seed = np.zeros((1, 1, num_class + 1))
    label_seed[0, 0, localization_cues["%s_labels" % image_id]] = 1.

    # method 1
    cues = np.zeros([num_class + 1, height, width])
    cues_i = localization_cues["%s_cues" % image_id]
    cues[cues_i[0], cues_i[1], cues_i[2]] = 1.

    # # method 2
    # gt_temp = np.zeros((height, height))
    # H = gt_temp.shape[0]
    # W = gt_temp.shape[1]
    # cues_i = localization_cues["%s_cues" % image_id]
    # for idx in range(len(cues_i[0])):
    #     if cues_i[1, idx] < H and cues_i[2, idx] < W:
    #         gt_temp[cues_i[1, idx], cues_i[2, idx]] = cues_i[0, idx]
    # gt_temp2 = np.zeros((num_class + 1, height, width))
    # for lab in localization_cues["%s_labels" % image_id]:
    #     gt_temp2[lab, :, :] = (gt_temp == lab).astype('float')
    #
    # assert (cues == gt_temp2).all()
    return label_seed.astype(np.float32), cues.astype(np.float32)


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    # tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    # resize transform
    tfm_gens.append(T.Resize((321, 321)))
    if is_train:
        # tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def annotations_to_instances_metaws(annos, image_size):
    target = Instances(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    return target


def transform_proposals_mataws(dataset_dict, image_shape, transforms, min_box_side_len, proposal_topk):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        min_box_side_len (int): keep proposals with at least this size
        proposal_topk (int): only keep top-K scoring proposals

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_side_len)
        boxes = boxes[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        dataset_dict["proposals"] = proposals
    else:
        return
