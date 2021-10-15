import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models


class DownResNet(nn.Module):
    BACKBONES = {
        "resnet18": (models.resnet18, [64, 128, 256, 512]),
        "resnet34": (models.resnet34, [64, 128, 256, 512]),
        "resnet50": (models.resnet50, [256, 512, 1024, 2048]),
        "resnet101": (models.resnet101, [256, 512, 1024, 2048]),
    }

    def __init__(self, type="resnet18"):
        super(DownResNet, self).__init__()
        pretrained, self.channels = self.BACKBONES[type]
        pretrained = pretrained(pretrained=True)
        pretrained = list(pretrained.children())

        self.pre_conv = nn.Sequential(*pretrained[0:4])
        self.layer1 = pretrained[4]
        self.layer2 = pretrained[5]
        self.layer3 = pretrained[6]
        self.layer4 = pretrained[7]

    def forward(self, x):
        x = self.pre_conv(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return [out1, out2, out3, out4]


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.0]))

    def forward(self, inp1, inp2):
        return inp1 * torch.sigmoid(self.w * inp2)


class Skip(nn.Module):
    def __init__(self, dim):
        super(Skip, self).__init__()
        self.conv1x1 = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, inp1, inp2):
        x = torch.cat([inp1, inp2], dim=1)
        x = self.conv1x1(x)
        return x


class UpBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, up_factor=1, skip=False):
        super(UpBasicBlock, self).__init__()
        if up_factor == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.gate = None
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=True),
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            )
            if skip:
                self.gate = Skip(planes)
            else:
                self.gate = Swish()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if up_factor != 1 or in_planes != int(self.expansion * planes):
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    in_planes, int(self.expansion * planes), kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(int(self.expansion * planes)),
            )

    def forward(self, x):
        if self.gate is not None:
            x, gate = x
        out = F.relu(self.bn1(self.conv1(x)))
        if self.gate is not None:
            out = self.gate(out, gate)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UpBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, up_factor=1, skip=False):
        super(UpBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if up_factor == 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.gate = None
        else:
            self.conv2 = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=True),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            )
            if skip:
                self.gate = Skip(planes)
            else:
                self.gate = Swish()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(self.expansion * planes))

        self.shortcut = nn.Sequential()
        if up_factor != 1 or in_planes != int(self.expansion * planes):
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    in_planes, int(self.expansion * planes), kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(int(self.expansion * planes)),
            )

    def forward(self, x):
        if self.gate is not None:
            x, gate = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.gate is not None:
            out = self.gate(out, gate)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UpResNet(nn.Module):
    BACKBONES = {
        "resnet18": (UpBasicBlock, [2, 2, 2, 2]),
        "resnet34": (UpBasicBlock, [3, 4, 6, 3]),
        "resnet50": (UpBottleneck, [3, 4, 6, 3]),
        "resnet101": (UpBottleneck, [3, 4, 23, 3]),
    }

    def __init__(self, type="resnet18", skip=False):
        super(UpResNet, self).__init__()
        block, num_blocks = self.BACKBONES[type]
        self.in_planes = 512
        self.channels = [int(512 * 0.5 ** f * block.expansion) for f in range(4)]

        block.expansion = 1 / block.expansion
        self.layer1 = self._make_layer(
            block, self.channels[0], num_blocks[0], up_factor=2, skip=skip
        )
        self.layer2 = self._make_layer(
            block, self.channels[1], num_blocks[1], up_factor=2, skip=skip
        )
        self.layer3 = self._make_layer(
            block, self.channels[2], num_blocks[2], up_factor=2, skip=skip
        )
        self.layer4 = self._make_layer(
            block, self.channels[3], num_blocks[3], up_factor=2, skip=skip
        )

    def _make_layer(self, block, planes, num_blocks, up_factor=1, skip=False):
        up_factors = [up_factor] + [1] * (num_blocks - 1)
        layers = []
        for up_factor in up_factors:
            layers.append(block(self.in_planes, planes, up_factor, skip))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x, prev_levels):
        out1 = self.layer1([x, prev_levels[-1]])
        out2 = self.layer2([out1, prev_levels[-2]])
        out3 = self.layer3([out2, prev_levels[-3]])
        out4 = self.layer4([out3, prev_levels[-4]])
        return [out1, out2, out3, out4]


def init_fpn_weights(module):
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class FPN(nn.Module):
    def __init__(self, in_dims, out_dim):
        super(FPN, self).__init__()
        self.init_conv = nn.Conv2d(in_dims[-1], out_dim, kernel_size=1, stride=1, padding=0)

        self.output1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.output2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.output3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.lateral1 = nn.Conv2d(in_dims[-2], out_dim, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(in_dims[-3], out_dim, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(in_dims[-4], out_dim, kernel_size=1, stride=1, padding=0)

        init_fpn_weights(self.output1)
        init_fpn_weights(self.output2)
        init_fpn_weights(self.output3)
        init_fpn_weights(self.lateral1)
        init_fpn_weights(self.lateral2)
        init_fpn_weights(self.lateral3)

    def _downsample_add(self, x, y):
        return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True) + y

    def forward(self, x):
        p3 = self._downsample_add(self.init_conv(x[-1]), self.lateral1(x[-2]))
        p2 = self._downsample_add(p3, self.lateral2(x[-3]))
        p1 = self._downsample_add(p2, self.lateral3(x[-4]))
        p3 = self.output1(p3)
        p2 = self.output2(p2)
        p1 = self.output3(p1)

        _, _, out_h, out_w = p1.shape
        out = [p3, p2, p1]
        out = [
            F.interpolate(o, size=(out_h, out_w))
            if (out_w != o.shape[3]) or (out_h != o.shape[2])
            else o
            for o in out
        ]
        out = torch.cat([out.unsqueeze(-1) for out in out], dim=-1)
        out = (out * F.softmax(out, dim=-1)).sum(dim=-1)
        return out


def proj_init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.type = type
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )
        self.apply(proj_init_weights)

    def forward(self, x):
        out = self.fc(x)
        out = F.normalize(out, p=2, dim=1)
        return out


class Network(nn.Module):
    def __init__(self, type="resnet18", skip=False, out_dim=256, proj_dim=256):
        super().__init__()
        self.down1 = DownResNet(type)
        self.down1_mlp = ProjectionHead(self.down1.channels[-1], proj_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_up = nn.Sequential(
            nn.Linear(self.down1.channels[-1], self.down1.channels[-1]),
            nn.BatchNorm1d(self.down1.channels[-1]),
            nn.ReLU(),
            nn.Linear(self.down1.channels[-1], 16 * self.down1.channels[-1]),
        )

        self.up1 = UpResNet(type, skip)

        self.down2 = FPN(self.up1.channels, out_dim)
        self.down2_mlp = ProjectionHead(out_dim, proj_dim)

    def forward(self, x):
        out = self.down1(x)
        x = torch.flatten(self.avg_pool(out[-1]), 1)
        mlp_fv1 = self.down1_mlp(x)

        x = self.init_up(x).view(x.shape[0], x.shape[1], 4, 4)
        out = self.up1(x, out)

        out = self.down2(out)
        x = torch.flatten(self.avg_pool(out), 1)
        mlp_fv2 = self.down2_mlp(x)

        return mlp_fv1, mlp_fv2


@torch.no_grad()
def gather_ddp(t):
    t_g = [torch.ones_like(t) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(t_g, t, async_op=False)
    out = torch.cat(t_g, dim=0)
    return out


class UnsupervisedWrapper(nn.Module):
    def __init__(self, model, proj_dim=256, q_size=65536, m=0.999, temp=0.07):
        super(UnsupervisedWrapper, self).__init__()
        self.m = m
        self.temp = temp
        self.q_size = q_size

        self.model_q = model
        self.model_k = copy.deepcopy(self.model_q)

        for param_k in self.model_k.parameters():
            param_k.requires_grad = False

        q1 = torch.randn(proj_dim, q_size)
        q1 = F.normalize(q1, p=2, dim=0)

        q2 = torch.randn(proj_dim, q_size)
        q2 = F.normalize(q2, p=2, dim=0)

        self.register_buffer(f"q1", q1)
        self.register_buffer(f"q2", q2)
        self.register_buffer("q_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_model_k(self):
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        bs_this = x.shape[0]
        x_g = gather_ddp(x)
        bs_all = x_g.shape[0]

        n_devices = bs_all // bs_this
        indx_shuffle = torch.randperm(bs_all).cuda()

        torch.distributed.broadcast(indx_shuffle, src=0)

        indx_unshuffle = torch.argsort(indx_shuffle)

        device_indx = torch.distributed.get_rank()
        indx_this = indx_shuffle.view(n_devices, -1)[device_indx]

        return x_g[indx_this], indx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(
        self, mlp_fv1, mlp_fv2, indx_unshuffle
    ):
        bs_this = mlp_fv1.shape[0]
        mlp_fv1_g = gather_ddp(mlp_fv1)
        mlp_fv2_g = gather_ddp(mlp_fv2)

        bs_all = mlp_fv1_g.shape[0]

        n_devices = bs_all // bs_this

        device_indx = torch.distributed.get_rank()
        indx_this = indx_unshuffle.view(n_devices, -1)[device_indx]

        return (
            mlp_fv1_g[indx_this],
            mlp_fv2_g[indx_this],
        )

    @torch.no_grad()
    def _update_queue(self, keys1, keys2):
        batch_size = keys1.shape[0]
        ptr = int(self.q_ptr)
        assert self.q_size % batch_size == 0

        self.q1[:, ptr : ptr + batch_size] = keys1.T
        self.q2[:, ptr : ptr + batch_size] = keys2.T

        self.q_ptr[0] = ptr

    def compute_mlp_logits(self, q, k, queue):
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, queue])
        logit_mlp = torch.cat([l_pos, l_neg], dim=1)
        logit_mlp /= self.temp
        label_mlp = torch.zeros(logit_mlp.shape[0], dtype=torch.long, device=q.device)
        return logit_mlp, label_mlp

    def forward(self, img_q, img_k, dist=False):

        mlp_fv1_q, mlp_fv2_q = self.model_q(img_q)

        with torch.no_grad():
            self._update_model_k()
            if dist:
                img_k, indx_unshuffle = self._batch_shuffle_ddp(img_k)
            mlp_fv1_k, mlp_fv2_k = self.model_k(img_k)
            if dist:
                (
                    mlp_fv1_k,
                    mlp_fv2_k,
                ) = self._batch_unshuffle_ddp(
                    mlp_fv1_k,
                    mlp_fv2_k,
                    indx_unshuffle,
                )

        logit_mlp1, label_mlp1 = self.compute_mlp_logits(
            mlp_fv1_q, mlp_fv1_k, self.q1.clone().detach()
        )
        logit_mlp2, label_mlp2 = self.compute_mlp_logits(
            mlp_fv2_q, mlp_fv2_k, self.q2.clone().detach()
        )

        if dist:
            mlp_fv1_k = gather_ddp(mlp_fv1_k)
            mlp_fv2_k = gather_ddp(mlp_fv2_k)
        self._update_queue(mlp_fv1_k, mlp_fv2_k)

        return (
            logit_mlp1,
            label_mlp1,
            logit_mlp2,
            label_mlp2,
        )


if __name__ == "__main__":
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    net = Network()
    net = UnsupervisedWrapper(net).cuda()
    out = net(a, b)
    print([o.shape for o in out])
