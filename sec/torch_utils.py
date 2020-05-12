import cv2
import numpy as np
import torch
from torchvision.utils import save_image, make_grid

from detectron2.utils.events import get_event_storage


def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, num_class, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    _, num_class, _, _ = mask.shape
    heatmaps = []
    results = []
    for i in range(num_class):
        mask_i = mask[:, i, ...]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_i.squeeze().cpu()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b])

        result = heatmap + img.cpu()
        result = result.div(result.max()).squeeze()

        heatmaps.append(heatmap)
        results.append(result)

    return heatmaps, results


def create_pascal_label_colormap():
    """
    PASCAL VOC 分割数据集的类别标签颜色映射label colormap

    返回:
        可视化分割结果的颜色映射Colormap
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """
    添加颜色到图片，根据数据集标签的颜色映射 label colormap

    参数:
        label: 整数类型的 2D 数组array, 保存了分割的类别标签 label

    返回:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(seg_map, image, ori_mask=None):
    """
    Args:
        seg_map (np.array): h * w
        image : RGB image h * w * 3

    Returns:
        weights (Tensor): weight of every proposal
    """
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    alpha = 0.3
    beta = 1 - alpha
    gamma = 0
    seg_image_overlay = cv2.addWeighted(image, alpha, seg_image, beta, gamma)

    result = np.concatenate([image, seg_image, seg_image_overlay], axis=0)
    result = result.transpose((2, 0, 1))
    storage = get_event_storage()
    storage.put_image("crf label", result)
