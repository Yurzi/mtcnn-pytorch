import torch.nn.functional as F


class MTCNNMultiSourceSLoss:
    def __init__(self, cls_weight, bbox_weight, landmark_weight, ohem_rate=1.0) -> None:
        """
        cls_weight: weight of classification loss
        bbox_weight: weight of bounding box regression loss
        landmark_weight: weight of landmark regression loss
        ohem_rate: the rate of top K (K=batch_size * ohem_rate) loss to be used in training
        """
        assert 0 <= ohem_rate <= 1.0, "ohem_rate must be in [0, 1]"

        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.landmark_weight = landmark_weight
        self.ohem_rate = ohem_rate

    def __call__(self, input, target, type_indicator):
        """
        input : a tuple of three tensors (cls_prob, bbox_pred, landmark_pred)
        target: a tuple of three tensors (cls_target, bbox_target, landmark_target)
        type_indicator: a tuple of three 0/1 indicator tensors (cls_type_indicator, bbox_type_indicator, landmark_type_indicator)
        """

        cls_prob, bbox_pred, landmark_pred = input
        cls_target, bbox_target, landmark_target = target
        cls_type_indicator, bbox_type_indicator, landmark_type_indicator = type_indicator

        cls_loss = F.cross_entropy(cls_prob, cls_target, reduction="none").unsqueeze(1)
        bbox_loss = F.mse_loss(bbox_pred, bbox_target, reduction="none").sum(dim=1, keepdim=True)
        landmark_loss = F.mse_loss(landmark_pred, landmark_target, reduction="none").sum(
            dim=1, keepdim=True
        )

        weighted_cls_loss = self.cls_weight * cls_type_indicator * cls_loss
        weighted_bbox_loss = self.bbox_weight * bbox_type_indicator * bbox_loss
        weighted_landmark_loss = self.landmark_weight * landmark_type_indicator * landmark_loss

        loss_list = weighted_cls_loss + weighted_bbox_loss + weighted_landmark_loss

        if self.ohem_rate == 1:
            return loss_list.mean()

        # use ohem to select top K hard examples
        batch_size = loss_list.size(0)
        num_hard = int(batch_size * self.ohem_rate + 0.5)
        # get top K loss mean
        topk_loss_list, _ = loss_list.topk(num_hard, dim=0)
        loss = topk_loss_list.mean()

        return loss
