import os
from pathlib import Path
import torch
from .coco import make_coco_transforms, CocoDetection
from util.misc import nested_tensor_from_tensor_list


class CocoDetectionWithSize(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Image width/height
        if "width" in target and "height" in target:
            w, h = target["width"], target["height"]
        else:
            w, h = img.size  # fallback to PIL.Image size

        # Convert xywh → xyxy
        boxes = []
        labels = []
        for obj in target["annotations"]:
            if "bbox" in obj:
                x, y, bw, bh = obj["bbox"]
                x2, y2 = x + bw, y + bh
                boxes.append([x, y, x2, y2])
                labels.append(obj["category_id"])

        # Convert to tensor
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Clamp to image bounds
            boxes[:, 0::2].clamp_(0, w)
            boxes[:, 1::2].clamp_(0, h)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare target in DETR expected format
        new_target = {
            "boxes": boxes,                        # absolute coords
            "labels": labels,
            "image_id": torch.tensor([target["image_id"]]),
            "size": torch.tensor([h, w])            # (H, W)
        }

        # Apply transforms (this step keeps colors correct & boxes in sync)
        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)

        return img, new_target


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
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks
    )
    return dataset
