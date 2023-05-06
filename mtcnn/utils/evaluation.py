from typing import Any
import torch

def IoU(bbox_a, bbox_b) -> float:
    """
    Input:
        bbox : [top_left_x,top_left_y,width, height]
    """
    # check input
    assert len(bbox_a) == 4 and len(bbox_b) == 4, "bbox_a and bbox_b must have 4 elements"
    res: float = 0
    bbox_1 = {
        "left_x": bbox_a[0],
        "left_y": bbox_a[1],
        "right_x": bbox_a[0] + bbox_a[2],
        "right_y": bbox_a[1] + bbox_a[3],
    }
    bbox_1_area = bbox_a[2] * bbox_a[3]

    bbox_2 = {
        "left_x": bbox_b[0],
        "left_y": bbox_b[1],
        "right_x": bbox_b[0] + bbox_b[2],
        "right_y": bbox_b[1] + bbox_b[3],
    }
    bbox_2_area = bbox_b[2] * bbox_b[3]

    mid_width = min(bbox_1["right_x"], bbox_2["right_x"]) - max(bbox_1["left_x"], bbox_2["left_x"])
    mid_height = min(bbox_1["right_y"], bbox_2["right_y"]) - max(bbox_1["left_y"], bbox_2["left_y"])

    # if has zero or negative value
    if mid_width <= 0 or mid_height <= 0:
        return 0

    mid_area = mid_width * mid_height
    res = mid_area / (bbox_1_area + bbox_2_area - mid_area)
    return res


def get_cls_accuarcy(pred:torch.Tensor, target:torch.Tensor, type_indicator:torch.Tensor) -> float:
    """
    Input:
        pred: [N, 2]
        target: [N]
        type_indicator: [N, 1]
    """
    assert pred.shape[0] == target.shape[0] == type_indicator.shape[0], "pred and target must have same number of elements"
    assert pred.shape[1] == 2, "pred must have 2 elements in the second dimension"
    assert len(target.shape) == 1, "target must be 1D tensor"

    pred = pred.argmax(dim=1)
    type_indicator = type_indicator.argmax(dim=1) # a trick to view
    correct = pred.eq(target).eq(type_indicator).sum().item()
    return correct / pred.shape[0]

def get_bbox_accuarcy(pred:torch.Tensor, target:torch.Tensor, type_indicator:torch.Tensor, iou_threshold) -> float:
    """
    Input:
        pred: [N, 4]
        target: [N, 4]
        type_indicator: [N, 1]
    """
    assert pred.shape[0] == target.shape[0] == type_indicator.shape[0], "pred and target must have same number of elements"
    assert pred.shape[1] == 4, "pred must have 4 elements in the second dimension"
    assert target.shape[1] == 4, "target must have 4 elements in the second dimension"

    correct = 0;
    total = pred.shape[0]

    for i in range(pred.shape[0]):
        pred_bbox = pred[i]
        target_bbox = target[i]
        iou = IoU(pred_bbox, target_bbox)
        if iou > iou_threshold and type_indicator[i] == 1:
            correct += 1

    return correct / total


def get_landmark_accuarcy(pred:torch.Tensor, target:torch.Tensor, type_indicator:torch.Tensor, threshold:torch.Tensor) -> float:
    """
    Input:
        pred: [N, 10]
        target: [N, 10]
        type_indicator: [N, 1]
    """
    assert pred.shape[0] == target.shape[0] == type_indicator.shape[0], "pred and target must have same number of elements"
    assert pred.shape[1] == 10, "pred must have 10 elements in the second dimension"
    assert target.shape[1] == 10, "target must have 10 elements in the second dimension"
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.Tensor(threshold)

    correct = 0;
    total = pred.shape[0]

    for i in range(pred.shape[0]):
        pred_landmark = pred[i]
        target_landmark = target[i]
        dist = pred_landmark - target_landmark
        dist = dist.abs()
        res = torch.le(dist, threshold)
        if res.sum() == 10 and type_indicator[i] == 1:
            correct += 1

    return correct / total


class MTCNNMultiTaskAcc():
    def __init__(self, cls_weight, bbox_weight, landmark_weight, iou_threshold, ldmk_threshold) -> None:
        """
        cls_weight: weight of classification loss
        bbox_weight: weight of bounding box regression loss
        landmark_weight: weight of landmark regression loss
        iou_threshold[float]: threshold of bbox regression
        ldmk_threshold[Tensor[10]]: threshold of landmark
        """
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.landmark_weight = landmark_weight
        self.iou_threshold = iou_threshold
        self.ldmk_threshold = ldmk_threshold

    def __call__(self, pred, target, type_indicator):
        """
        input : a tuple of three tensors (cls_prob, bbox_pred, landmark_pred)
        target: a tuple of three tensors (cls_target, bbox_target, landmark_target)
        type_indicator: a tuple of three 0/1 indicator tensors (cls_type_indicator, bbox_type_indicator, landmark_type_indicator)
        """

        cls_prob, bbox_pred, landmark_pred = pred
        cls_target, bbox_target, landmark_target = target
        cls_type_indicator, bbox_type_indicator, landmark_type_indicator = type_indicator

        cls_acc = get_cls_accuarcy(cls_prob, cls_target, cls_type_indicator)
        bbox_acc = get_bbox_accuarcy(bbox_pred, bbox_target, bbox_type_indicator, self.iou_threshold)
        landmark_acc = get_landmark_accuarcy(landmark_pred, landmark_target, landmark_type_indicator, self.ldmk_threshold)

        general_acc = self.cls_weight * cls_acc + self.bbox_weight * bbox_acc + self.landmark_weight * landmark_acc
        general_acc = general_acc / (self.cls_weight + self.bbox_weight + self.landmark_weight)

        return general_acc, cls_acc, bbox_acc, landmark_acc


