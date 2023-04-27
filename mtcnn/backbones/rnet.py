import torch.nn as nn
import torch.nn.functional as F


class RNet(nn.Module):
    def __init__(self) -> None:
        super(RNet, self).__init__()

        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 28, 3)
        # PReLU(num_parameters)
        self.prelu1 = nn.PReLU(28)
        # MaxPool2d(kernel_size, stride, padding, ceil_mode)
        self.maxpool1 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(28, 48, 3)
        self.prelu2 = nn.PReLU(48)
        self.maxpool2 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(48, 64, 2)
        self.prelu3 = nn.PReLU(64)

        self.fc1 = nn.Linear(3 * 3 * 64, 128)
        self.prelu4 = nn.PReLU(128)

        self.fc2_1 = nn.Linear(128, 2)
        self.fc2_2 = nn.Linear(128, 4)
        self.fc2_3 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Input:
            x: a tensor with shape [batch_size, 3, H, W]
        Output:
            prob    : a tensor with shape [batch_size, 2, H', W']
            offset  : a tensor with shape [batch_size, 4, H', W']
            landmark: a tensor with shape [batch_size, 10, H', W']
        """
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.prelu3(x)

        x = self.fc1(x)
        feature = self.prelu4(x)

        prob = F.softmax(self.fc2_1(feature), dim=1)
        offset = self.fc2_2(feature)
        landmark = self.fc2_3(feature)

        return prob, offset, landmark
