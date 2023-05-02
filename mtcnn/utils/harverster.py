import random
from typing import Generator, Tuple, Union

import torch
import torchvision.transforms.functional as VF

from mtcnn.utils.evaluation import IoU
from mtcnn.utils.functional import get_abs_bbox
from mtcnn.utils.logger import ConsoleLogWriter, DebugLogger

logger = DebugLogger(__name__, ConsoleLogWriter())


class RandomHarvester:
    def __init__(self, original: torch.Tensor, gt_bbox: torch.Tensor, anchor_size: int):
        """
        Input:
            original: original image
            gt_bbox: relative bounding box (x, y, w, h)
            size: crop size
        """
        self.image = original
        # image_size -> (width, height)
        self.image_width, self.image_height = (original.shape[2], original.shape[1])
        self.anchor_size = int(anchor_size)
        if gt_bbox is not None:
            self.gt_bbox = gt_bbox
            self.gt_bbox_width = (gt_bbox[2] * self.image_width).item()
            self.gt_bbox_height = (gt_bbox[3] * self.image_height).item()
            # acutually area
            gt_bbox_area = self.gt_bbox_width * self.gt_bbox_height
            anchor_area = anchor_size**2
            self.max_iou = min(gt_bbox_area, anchor_area) / max(gt_bbox_area, anchor_area)
        else:
            self.gt_bbox = torch.zeros(4)
            self.gt_bbox_width = 0
            self.gt_bbox_height = 0
            self.max_iou = 0

    def __call__(self, iou_threshold: Tuple[float, float]):
        """
        Output:
            cropped image
            cropped anchor box
        """

        # check iou_threshold is valid
        iou_min, iou_max = iou_threshold
        iou_min = min(iou_min, iou_max)
        iou_max = max(iou_min, iou_max)

        if iou_min > self.max_iou:
            logger.warn(
                f"now min iou is over than max_iou can get, iou_threshold: ({iou_min},{iou_max}), max_iou: {self.max_iou}, fallback to max_iou"
            )
            iou_min = self.max_iou
        if iou_max > 1:
            iou_max = 1

        # get anchor position
        anchor_pos_x, anchor_pos_y = self.get_anchor_pos((iou_min, iou_max))
        # get cropped image
        cropped_image = VF.crop(
            self.image, anchor_pos_y, anchor_pos_x, self.anchor_size, self.anchor_size
        )

        relative_anchor_pos = (anchor_pos_x / self.image_width, anchor_pos_y / self.image_height)

        return cropped_image, relative_anchor_pos

    def get_anchor_pos(self, iou_threshold: Tuple[float, float]) -> Tuple:
        """
        method to get anchor position[abs_x, abs_y]
        """

        # randomly select an anchor pos
        abs_gt_bbox = get_abs_bbox((self.image_width, self.image_height), self.gt_bbox)
        abs_gt_bbox_x1, abs_gt_bbox_y1, abs_gt_bbox_x2, abs_gt_bbox_y2 = abs_gt_bbox

        iou_min, iou_max = iou_threshold

        # loop until get a valid anchor pos
        while True:
            # if iou_min is 0, all image is under considered
            range_x = (0, self.image_width - self.anchor_size)
            range_y = (0, self.image_height - self.anchor_size)
            if iou_min > 0:
                range_x = (
                    int(max(0, abs_gt_bbox_x1 - self.anchor_size)),
                    int(
                        min(self.image_width - self.anchor_size, abs_gt_bbox_x2 + self.anchor_size)
                    ),
                )
                range_y = (
                    int(max(0, abs_gt_bbox_y1 - self.anchor_size)),
                    int(
                        min(self.image_height - self.anchor_size, abs_gt_bbox_y2 + self.anchor_size)
                    ),
                )

            # random select anchor pos
            anchor_pos_x = random.randint(range_x[0], range_x[1] - 1)
            anchor_pos_y = random.randint(range_y[0], range_y[1] - 1)

            # calculate iou
            anchor_bbox = (anchor_pos_x, anchor_pos_y, self.anchor_size, self.anchor_size)
            gt_bbox = (abs_gt_bbox_x1, abs_gt_bbox_y1, self.gt_bbox_width, self.gt_bbox_height)
            iou = IoU(anchor_bbox, gt_bbox)

            if iou_min <= iou < iou_max:
                return anchor_pos_x, anchor_pos_y
