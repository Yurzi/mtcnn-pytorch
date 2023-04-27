import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self) -> None:
        super(PNet, self).__init__()

        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 10, 3)
        # PReLU(num_parameters)
        self.prelu1 = nn.PReLU(10)
        # MaxPool2d(kernel_size, stride, padding, ceil_mode)
        self.maxpool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(10, 16, 3)
        self.prelu2 = nn.PReLU(16)

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.prelu3 = nn.PReLU(32)

        self.conv4_1 = nn.Conv2d(32, 2, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1)
        self.conv4_3 = nn.Conv2d(32, 10, 1)

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

        x = self.conv3(x)
        feature = self.prelu3(x)

        prob = F.softmax(self.conv4_1(feature), dim=1)
        offset = self.conv4_2(feature)
        landmark = self.conv4_3(feature)

        return prob, offset, landmark
