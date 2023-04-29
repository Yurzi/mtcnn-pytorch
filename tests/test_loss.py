import torch

from mtcnn.backbones.functional import MTCNNMultiSourceSLoss
from mtcnn.backbones.net import PNet, RNet

p_net = PNet()
p_loss = MTCNNMultiSourceSLoss(1, 0.5, 0.5, ohem_rate=0.7)

if __name__ == "__main__":
    input = torch.randn(4, 3, 12, 12, requires_grad=True)
    output = p_net(input)

    target = (torch.argmax(torch.randn(4, 2), dim=1), torch.randn(4, 4), torch.randn(4, 10))
    type_indicator = (torch.ones(4, 1), torch.ones(4, 1), torch.ones(4, 1))

    loss = p_loss(output, target, type_indicator)

    print("[loss]:", loss)
    print("[grad]:", input.grad)

    loss.backward()
    print("[grad]:", input.grad)
