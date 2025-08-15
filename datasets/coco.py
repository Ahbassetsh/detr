import os
from pathlib import Path
import torch
import torchvision
from .coco import make_coco_transforms
from .coco import CocoDetection
from util.misc import nested_tensor_from_tensor_list

class CocoDetectionWithSize(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Get width & height from annotation (RoboFlow COCO export includes this)
        if "width" in target and "height" in target:
            w, h = target["width"], target["height"]
        else:
            w, h = img.size  # fallback

        # Convert xywh → xyxy
        anno = target["annotations"]
        boxes = []
        labels = []
        for obj in anno:
            if 'bbox' in obj:
                x, y, bw, bh = obj["bbox"]
                x2 = x + bw
                y2 = y + bh
                boxes.append([x, y, x2, y2])
                labels.append(obj["category_id"])
        
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Clamp to image size
            boxes[:, 0::2].clamp_(0, w)
            boxes[:, 1::2].clamp_(0, h)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target_out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([target["image_id"]])
        }

        return img, target_out


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} does not exist'

    PATHS = {
        "train": (root / "train2017", root / "annotations" / "instances_train2017.json"),
        "val":   (root / "val2017",   root / "annotations" / "instances_val2017.json"),
        "test":  (root / "test2017",  root / "annotations" / "instances_test2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetectionWithSize(
        img_folder, ann_file, 
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks
    )
    return dataset
