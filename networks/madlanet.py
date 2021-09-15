import math
import numpy as np
from torch import nn
import torch.nn.functional as F
import timm
from .DCNv2_latest.dcn_v2 import DCN
import torch

BN_MOMENTUM = 0.1

class DLA(nn.Module):
    BACKBONE = {
        "dla34": [16, 32, 64, 128, 256, 512],
    }
    def __init__(self, type="dla34"):
        super(DLA, self).__init__()
        dla = timm.create_model(type, pretrained=True)
        self.channels = self.BACKBONE[type]
        dla_layers = list(dla.children())[:-1]
        
        self.pre_conv = nn.Sequential(*dla_layers[0:3])
        self.layer_1 = dla_layers[3]
        self.layer_2 = dla_layers[4]
        self.layer_3 = dla_layers[5]
        self.layer_4 = dla_layers[6]
    
    def forward(self, x):
        x = self.pre_conv(x)
        layer1 = self.layer_1(x)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)
        return [None, None, layer1, layer2, layer3, layer4]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_dim, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
        )
        self.conv = DCN(
            in_dim,
            out_dim,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, out_dim, layer_fdims, up_factor):
        super(IDAUp, self).__init__()
        for i in range(1, len(layer_fdims)):
            c = layer_fdims[i]
            f = int(up_factor[i])
            proj = DeformConv(c, out_dim)
            node = DeformConv(out_dim, out_dim)

            up = nn.ConvTranspose2d(
                out_dim,
                out_dim,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=out_dim,
                bias=False,
            )
            fill_up_weights(up)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

    def forward(self, layers, start, end):
        for i in range(start + 1, end):
            upsample = getattr(self, "up_" + str(i - start))
            project = getattr(self, "proj_" + str(i - start))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - start))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, start, dims, spat_dims, in_dims=None):
        super(DLAUp, self).__init__()
        self.start = start
        if in_dims is None:
            in_dims = dims
        self.dims = dims
        dims = list(dims)
        spat_dims = np.array(spat_dims, dtype=int)
        for i in range(len(dims) - 1):
            j = -i - 2
            setattr(
                self,
                "ida_{}".format(i),
                IDAUp(dims[j], in_dims[j:], spat_dims[j:] // spat_dims[j]),
            )
            spat_dims[j + 1 :] = spat_dims[j]
            in_dims[j + 1 :] = [dims[j] for _ in dims[j + 1 :]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.start - 1):
            ida = getattr(self, "ida_{}".format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale, mode=self.mode, align_corners=False
        )
        return x

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_dim, out_dim, 1)
    
    def forward(self, inp1, inp2):
        return inp1 * torch.sigmoid(self.conv1x1(inp2))

class DLABottom(nn.Module):
    BACKBONE = {
        "dla34": [16, 32, 64, 128, 256, 512],
    }
    def __init__(self, type="dla34"):
        super(DLABottom, self).__init__()
        dla = timm.create_model(type, pretrained=True)
        self.channels = self.BACKBONE[type]
        dla_layers = list(dla.children())[:-1]
        
        self.layer_2 = dla_layers[4]
        self.glu_2 = GLU(64, self.channels[-3])
        self.layer_3 = dla_layers[5]
        self.glu_3 = GLU(64, self.channels[-2])
        self.layer_4 = dla_layers[6]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, prev):
        x = self.layer_2(x)
        x = self.glu_2(x, self.pool(prev[-2]))
        x = self.layer_3(x)
        x = self.glu_3(x, self.pool(self.pool(prev[-3])))
        x = self.layer_4(x)
        return x

class MultiAbstractionNet(nn.Module):
    def __init__(self, type):
        super(MultiAbstractionNet, self).__init__()
        self.down1 = DLA(type)
        channels = self.down1.channels
        spat_dims = [2 ** i for i in range(2, 6)]
        self.dla_up = DLAUp(2, channels[2:], spat_dims)
        self.ida_up = IDAUp(
            channels[2],
            channels[2:5],
            [2 ** i for i in range(3)],
        )
        self.down2 = DLABottom(type)
        self.backbone_dim = channels[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.down1(x)
        
        x = self.dla_up(x)
        y = []
        for i in range(3):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        x = self.down2(y[-1], y)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

