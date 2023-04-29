import torch

from mtcnn.backbones.net import ONet, PNet, RNet

p_net = PNet()
r_net = RNet()
o_net = ONet()

if __name__ == "__main__":
    p_input = torch.randn(2, 3, 12, 12)
    r_input = torch.randn(2, 3, 24, 24)
    o_input = torch.randn(2, 3, 48, 48)

    p_prob, p_offset, p_landmark = p_net(p_input)
    r_prob, r_offset, r_landmark = r_net(r_input)
    o_prob, o_offset, o_landmark = o_net(o_input)

    print("[PNet]", p_prob, p_offset, p_landmark)
    print("[RNet]", r_prob, r_offset, r_landmark)
    print("[ONet]", o_prob, o_offset, o_landmark)

    pass
