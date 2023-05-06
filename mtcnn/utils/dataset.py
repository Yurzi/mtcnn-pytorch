from collections import OrderedDict
import os
from collections.abc import Callable
from typing import Generator, List

import torch
import torchvision.transforms.functional as VF
from torchvision import transforms

from .filesystem import check_and_reset
from .harverster import RandomHarvester
from .logger import ConsoleLogWriter, DebugLogger
from .parser import write_anno_file

from tqdm import tqdm

logger = DebugLogger(__name__, ConsoleLogWriter())


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


def construct_image_pyramid(
    original: torch.Tensor, step_fn: Generator[float, None, None]
) -> List[torch.Tensor]:
    """
    construct image pyramid from original image
    Input:
        original: original image
        step_fn: a generator that generate a scale factor

    Output:
        list of image [biggest size -> smallset size]
    """
    res: List[torch.Tensor] = list()
    ori_width = original.shape[2]
    ori_height = original.shape[1]

    for scale_factor in step_fn:
        transform = transforms.Compose(
            [
                transforms.Resize((ori_height * scale_factor, ori_width * scale_factor)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        res.append(transform(original))

    return res


def generate_train_set_from_raw(
    raw_dataset,
    perfix: str,
    task_type: str,
    target_size,
    anchor_fn: Callable | int,
    harverster: Callable | None = None,
    config=None,
    reset: bool = False
):
    if task_type not in ["train", "eval", "test"]:
        raise ValueError("task_type must be one of train, eval, test")

    image_dir_perfix = "images"
    image_dir = os.path.join(perfix, image_dir_perfix)
    annotation_basename = task_type + ".txt"
    annotation_path = os.path.join(perfix, annotation_basename)
    # check and init dir
    if reset:
        check_and_reset(image_dir)
        check_and_reset(annotation_path, is_file=True)
    # properties
    neg_num = 25
    iou_threshold_1 = 0.3
    part_num = 25
    iou_threshold_2 = 0.7
    pos_num = 75

    if config is not None:
        neg_num = config.negative_num
        iou_threshold_1 = config.iou_threshold[0]
        part_num = config.part_num
        iou_threshold_2 = config.iou_threshold[1]
        pos_num = config.positive_num

    logger.info("trying to generate "+ task_type +" set in " + perfix)

    counter = 0
    total_num = len(raw_dataset) * (neg_num + part_num + pos_num)
    zfill_len = len(str(total_num))


    anchor_size = target_size
    if isinstance(anchor_fn, Callable):
        anchor_size = anchor_fn(raw_dataset)
    else:
        anchor_size = anchor_fn

    anchor_size = int(anchor_size)

    logger.info("anchor_size: " + str(anchor_size))

    pbar = tqdm(total=total_num, desc=f"{task_type} set", mininterval=0.3)
    annotations = list()


    def process_one(counter, cropped_img, anchor_pos, gt_bbox, gt_landmark, cls_label):
        reassign_bbox = torch.zeros(4)
        if bbox is not None:
            reassign_bbox[0] = gt_bbox[0] - anchor_pos[0]
            reassign_bbox[1] = gt_bbox[1] - anchor_pos[1]
            reassign_bbox[2] = gt_bbox[2]
            reassign_bbox[3] = gt_bbox[3]
        reassign_landmark = torch.zeros(10)
        if landmark is not None:
            reassign_landmark = gt_landmark

        image_name = task_type + str(counter).zfill(zfill_len) + ".jpg"

        annotations.append((image_name, cls_label, reassign_bbox, reassign_landmark))
        # save image
        image_path = os.path.join(image_dir, image_name)
        resized_img = VF.resize(cropped_img, list(target_size), antialias=True)
        pil_image = VF.to_pil_image(resized_img)
        pil_image.save(image_path)
        pbar.set_postfix(OrderedDict({'img_name': image_name}))
        pbar.update(1)


    for img, bbox, landmark in raw_dataset:
        if harverster is None:
            harverster = RandomHarvester(img, bbox, anchor_size)
        # generate negative samples
        for _ in range(neg_num):
            cropped_img, anchor_pos = harverster((0, iou_threshold_1))
            process_one(counter, cropped_img, anchor_pos, bbox, landmark, 0)
            counter += 1
        for _ in range(part_num):
            cropped_img, anchor_pos = harverster((iou_threshold_1, iou_threshold_2))
            process_one(counter, cropped_img, anchor_pos, bbox, landmark, 2)
            counter += 1
        for _ in range(pos_num):
            cropped_img, anchor_pos = harverster((iou_threshold_2, 1))
            process_one(counter, cropped_img, anchor_pos, bbox, landmark, 1)
            counter += 1

    # save annotations
    write_anno_file(annotation_path, annotations)
    pbar.close()
    logger.info("finished")
