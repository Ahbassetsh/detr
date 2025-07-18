# datasets/custom.py

import os
from .coco import make_coco_transforms
from .coco import CocoDetection
from util.misc import nested_tensor_from_tensor_list

from pathlib import Path
import torchvision

def build(image_set, args):
    # Path to annotations and images
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} does not exist'

    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"instances_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"instances_val2017.json"),
        "test": (root / "test2017", root / "annotations" / f"instances_test2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset
