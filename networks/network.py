import torch.nn as nn
import torch.nn.functional as F
from . import madlanet

BACKBONE = {
    "madlanet": madlanet.MultiAbstractionNet
}

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.W1 = nn.Linear(in_dim, in_dim)
        self.BN1 = nn.BatchNorm1d(in_dim)
        self.ReLU = nn.ReLU()
        self.W2 = nn.Linear(in_dim, out_dim)
        self.BN2 = nn.BatchNorm1d(out_dim)
        self.apply(init_weights)

    def forward(self, x):
        out = self.BN2(self.W2(self.ReLU(self.BN1(self.W1(x)))))
        out = F.normalize(out, p=2, dim=1)
        return out

class Network(nn.Module):
    def __init__(self, backbone_type="madlanet", backbone_kwargs={"type": "dla34"}, out_dim=128):
        super(Network, self).__init__()
        self.backbone = BACKBONE[backbone_type](**backbone_kwargs)
        self.proj_head = ProjectionHead(self.backbone.backbone_dim, out_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.proj_head(x)
        return x

if __name__ == "__main__":
    import torch
    net = Network().cuda()
    imgs = torch.randn(2, 3, 256, 256).cuda()
    out = net(imgs)
    print(out.shape)
    print(f"Model parameters: {sum(p.numel() for p in net.parameters())/1e6}M")