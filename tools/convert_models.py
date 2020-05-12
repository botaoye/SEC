from collections import OrderedDict

import torch


path = "../output/models/vgg16_20M.pth"
checkpoint = torch.load(path)
# checkpoint = checkpoint['model']
print(checkpoint.keys())

map = {
    "0.weight": "backbone.features.0.weight",
    "2.weight": "backbone.features.2.weight",
    "5.weight": "backbone.features.5.weight",
    "7.weight": "backbone.features.7.weight",
    "10.weight": "backbone.features.10.weight",
    "12.weight": "backbone.features.12.weight",
    "14.weight": "backbone.features.14.weight",
    "17.weight": "backbone.features.17.weight",
    "19.weight": "backbone.features.19.weight",
    "21.weight": "backbone.features.21.weight",
    "24.weight": "backbone.features.24.weight",
    "26.weight": "backbone.features.26.weight",
    "28.weight": "backbone.features.28.weight",
    "31.weight": "sem_seg_head.fc6.weight",
    "34.weight": "sem_seg_head.fc7.weight",
    "37.weight": "roi_heads.box_head.fc2.weight",

    "0.bias": "backbone.features.0.bias",
    "2.bias": "backbone.features.2.bias",
    "5.bias": "backbone.features.5.bias",
    "7.bias": "backbone.features.7.bias",
    "10.bias": "backbone.features.10.bias",
    "12.bias": "backbone.features.12.bias",
    "14.bias": "backbone.features.14.bias",
    "17.bias": "backbone.features.17.bias",
    "19.bias": "backbone.features.19.bias",
    "21.bias": "backbone.features.21.bias",
    "24.bias": "backbone.features.24.bias",
    "26.bias": "backbone.features.26.bias",
    "28.bias": "backbone.features.28.bias",
    "31.bias": "sem_seg_head.fc6.bias",
    "34.bias": "sem_seg_head.fc7.bias",
    "37.bias": "roi_heads.box_head.fc2.bias",
}

new_state_dict = OrderedDict()

for k, v in checkpoint.items():
    new_state_dict[map[k]] = v

new_path = "../output/models/vgg_new.pth"
torch.save(new_state_dict, new_path)
