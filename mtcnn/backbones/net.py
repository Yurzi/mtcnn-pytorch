import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self) -> None:
        super(PNet, self).__init__()
        # store feature_map for other usage
        self.feature_map = None

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

        self.flat = nn.Flatten()

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
        # store feature_map
        self.feature_map = feature

        prob = self.flat(self.conv4_1(feature))
        offset = self.flat(self.conv4_2(feature))
        landmark = self.flat(self.conv4_3(feature))

        if not self.training:
            prob = F.softmax(prob, dim=1)

        return prob, offset, landmark


class RNet(nn.Module):
    def __init__(self) -> None:
        super(RNet, self).__init__()
        # store feature_map for other usage
        self.feature_map = None

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

        self.flat = nn.Flatten()

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

        x = self.flat(x)

        x = self.fc1(x)
        feature = self.prelu4(x)
        # store feature_map
        self.feature_map = feature

        prob = self.fc2_1(feature)
        offset = self.fc2_2(feature)
        landmark = self.fc2_3(feature)

        if not self.training:
            prob = F.softmax(prob, dim=1)

        return prob, offset, landmark


class ONet(nn.Module):
    def __init__(self) -> None:
        super(ONet, self).__init__()
        # store feature_map for other usage
        self.feature_map = None

        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 32, 3)
        # PReLU(num_parameters)
        self.prelu1 = nn.PReLU(32)
        # MaxPool2d(kernel_size, stride, padding, ceil_mode)
        self.maxpool1 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.prelu2 = nn.PReLU(64)
        self.maxpool2 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3)
        self.prelu3 = nn.PReLU(64)
        self.maxpool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 128, 2)
        self.prelu4 = nn.PReLU(128)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(3 * 3 * 128, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)

        self.prelu5 = nn.PReLU(256)

        self.fc2_1 = nn.Linear(256, 2)
        self.fc2_2 = nn.Linear(256, 4)
        self.fc2_3 = nn.Linear(256, 10)

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
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.prelu4(x)

        x = self.flat(x)

        x = self.fc1(x)
        x = self.batch_norm1(x)
        feature = self.prelu5(x)
        # store feature_map
        self.feature_map = feature

        prob = self.fc2_1(feature)
        offset = self.fc2_2(feature)
        landmark = self.fc2_3(feature)

        if not self.training:
            prob = F.softmax(prob, dim=1)

        return prob, offset, landmark
