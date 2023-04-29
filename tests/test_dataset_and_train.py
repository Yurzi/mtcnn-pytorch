import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from mtcnn.backbones.functional import MTCNNMultiSourceSLoss
from mtcnn.backbones.net import PNet


class TestDataset(Dataset):
    def __init__(self, size: int) -> None:
        super(TestDataset, self).__init__()

        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> Any:
        # return image, (cls_label, bbox, landmark), (cls_type_indicator, bbox_type_indicator, landmark_type_indicator)
        return (
            torch.randn(3, 12, 12),
            (
                random.randint(0, 1),
                torch.randn(
                    4,
                ),
                torch.randn(
                    10,
                ),
            ),
            (torch.randint(0, 2, (1,)), torch.randint(0, 2, (1,)), torch.randint(0, 2, (1,))),
        )


dataset = TestDataset(256)

model = PNet()
train_loss = MTCNNMultiSourceSLoss(1, 0.5, 0.5, ohem_rate=0.7)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

if __name__ == "__main__":
    step = 0

    for i, (image, target, type_indicator) in enumerate(train_loader):
        step += 1

        output = model(image)

        loss = train_loss(output, target, type_indicator)
        print("[loss]:", loss, "[step]:", step)

        model.zero_grad()

        loss.backward()
