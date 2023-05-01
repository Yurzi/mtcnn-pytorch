import random
from typing import Generator, Tuple

import torch
import torchvision.transforms.functional as VF

from mtcnn.utils.evaluation import IoU


def RandomHarverster(
    img: torch.Tensor, anchor_size: int, num: int = -1
) -> Generator[Tuple[torch.Tensor, Tuple[float, float, float, float]], None, None]:
    img_width = img.shape[2]
    img_height = img.shape[1]
    width_range = img_width - anchor_size
    height_range = img_height - anchor_size
    while num > 0 or num == -1:
        # get random anchor
        x = random.randint(0, width_range - 1)
        y = random.randint(0, height_range - 1)
        cropd_img = VF.crop(img, y, x, anchor_size, anchor_size)
        yield cropd_img, (
            x / img_width,
            y / img_height,
            anchor_size / img_width,
            anchor_size / img_height,
        )


def get_mean_anchor_size(dataset) -> float:
    sum_width, sum_height = 0, 0

    for img, bbox, _ in dataset:
        if bbox is None:
            continue

        img_width = img.shape[2]
        img_height = img.shape[1]

        sum_width += bbox[2] * img_width
        sum_height += bbox[3] * img_height

    total_num = len(dataset)
    mean_width = sum_width / total_num
    mean_height = sum_height / total_num

    return (mean_width + mean_height) / 2


def harverst_train_set_frow_raw(raw_dataset, crop_size: Tuple[int, int]):
    """
    generate train dataset from raw dataset. Generally, the function just try to crop crop_size images from raw images.
    some properties will be applied to control this process[todo]
    """
    num: int = 25
    anchor_scale = 0.65
    anchor_size = int(anchor_scale * get_mean_anchor_size(raw_dataset))
    iou_threshold_1 = 0.55
    iou_threshold_2 = 0.75

    for img, bbox, landmark in raw_dataset:
        # crop image by iou threshold and anchor size
        harverseter = RandomHarverster(img, anchor_size, num)
        for cropd_img, (x, y, width, height) in harverseter:
            # resize cropd_img to crop_size
            cropd_img = VF.resize(cropd_img, list(crop_size), antialias=True)
            # reassign bbox
            reassigned_bbox = torch.zeros(4)
            if bbox is not None:
                reassigned_bbox = bbox.clone().detach()
                reassigned_bbox[0] = reassigned_bbox[0] - x
                reassigned_bbox[1] = reassigned_bbox[1] - y
            reassigned_landmark = torch.zeros(10) if landmark is None else landmark.clone().detach()

            # get cls_label by iou
            cls_label = 0

            if bbox is not None:
                iou = IoU(bbox, (x, y, width, height))
                if iou > iou_threshold_2:
                    cls_label = 1
                elif iou > iou_threshold_1:
                    cls_label = 2

            if landmark is None:
                if cls_label == 1:
                    cls_label = 2

            yield cropd_img, cls_label, reassigned_bbox, reassigned_landmark
