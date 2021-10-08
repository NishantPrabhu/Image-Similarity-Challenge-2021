import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, input1, input2):
        return input1 * torch.sigmoid(self.weight * input2)


class Skip(nn.Module):
    def __init__(self, dim):
        super(Skip, self).__init__()
        self.conv1x1 = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], dim=1)
        x = self.conv1x1(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down=True, skip=False):
        super(BasicBlock, self).__init__()
        if down:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True),
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
        if stride != 1 or in_planes != int(self.expansion * planes):
            if down:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        int(self.expansion * planes),
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(int(self.expansion * planes)),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True),
                    nn.Conv2d(
                        in_planes,
                        int(self.expansion * planes),
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(int(self.expansion * planes)),
                )

    def forward(self, x):
        if isinstance(x, list):
            x, gate = x
        else:
            gate = None
        out = F.relu(self.bn1(self.conv1(x)))
        if gate is not None:
            out = self.gate(out, gate)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down=True, skip=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if down:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True),
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
        if stride != 1 or in_planes != self.expansion * planes:
            if down:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        int(self.expansion * planes),
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(int(self.expansion * planes)),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True),
                    nn.Conv2d(
                        in_planes,
                        int(self.expansion * planes),
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(int(self.expansion * planes)),
                )
        self.swish = Swish()

    def forward(self, x):
        if isinstance(x, list):
            x, gate = x
        else:
            gate = None
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if gate is not None:
            out = self.gate(out, gate)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DownResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(DownResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.channels = [int(f * block.expansion) for f in [64, 128, 256, 512]]

    def _make_layer(self, block, planes, num_blocks, stride=1, down=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, down))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return [out1, out2, out3, out4]


class UpResNet(nn.Module):
    def __init__(self, block, num_blocks, skip=False):
        super(UpResNet, self).__init__()
        self.in_planes = 512
        block.expansion = 1 / block.expansion
        self.layer1 = self._make_layer(block, 512, num_blocks[0], stride=2, down=False, skip=skip)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2, down=False, skip=skip)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2, down=False, skip=skip)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2, down=False, skip=skip)
        self.channels = [int(f * block.expansion) for f in [512, 256, 128, 64]]

    def _make_layer(self, block, planes, num_blocks, stride=1, down=True, skip=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, down, skip))
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
        return p3, p2, p1


def proj_init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class ProjectionHead(nn.Module):
    def __init__(self, type, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.type = type
        if type == "mlp":
            self.fc = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim),
            )
        elif type == "dense":
            self.fc = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True),
            )
        self.apply(proj_init_weights)

    def forward(self, x):
        out = self.fc(x)
        if self.type == "mlp":
            out = F.normalize(out, p=2, dim=1)
        else:
            out = out.view(out.shape[0], out.shape[1], -1)
            out = F.normalize(out, p=2, dim=1)
        return out


class Network(nn.Module):
    TYPE = {"resnet18": (BasicBlock, [2, 2, 2, 2])}

    def __init__(self, type="resnet18", skip=False, out_dim=256, proj_dim=256):
        super().__init__()
        block, num_blocks = self.TYPE[type]

        self.down1 = DownResNet(block, num_blocks)
        self.down1_mlp = ProjectionHead("mlp", self.down1.channels[-1], proj_dim)
        self.down1_dense = ProjectionHead("dense", self.down1.channels[-1], proj_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_up = nn.Sequential(
            nn.Linear(self.down1.channels[-1], self.down1.channels[-1]),
            nn.BatchNorm1d(self.down1.channels[-1]),
            nn.ReLU(),
            nn.Linear(self.down1.channels[-1], 16 * self.down1.channels[-1]),
        )

        self.up1 = UpResNet(block, num_blocks, skip)

        self.down2 = FPN(self.up1.channels, out_dim)
        self.down2_mlp = ProjectionHead("mlp", out_dim, proj_dim)
        self.down2_dense = ProjectionHead("dense", out_dim, proj_dim)

    def forward(self, x):
        outputs = []
        out = self.down1(x)
        dense_fv1 = self.down1_dense(out[-1])
        x = torch.flatten(self.avg_pool(out[-1]), 1)
        mlp_fv1 = self.down1_mlp(x)
        outputs.extend(
            [
                F.normalize(out[-1].view(out[-1].shape[0], out[-1].shape[1], -1), p=2, dim=1),
                dense_fv1,
                mlp_fv1,
            ]
        )

        x = self.init_up(x).view(x.shape[0], x.shape[1], 4, 4)
        out = self.up1(x, out)
        outputs.append(out[-1])

        out = self.down2(out)
        dense_fv2 = self.down2_dense(out[-1])
        x = torch.flatten(self.avg_pool(out[-1]), 1)
        mlp_fv2 = self.down2_mlp(x)
        outputs.extend(
            [F.normalize(out[-1].view(out[-1].shape[0], out[-1].shape[1], -1)), dense_fv2, mlp_fv2]
        )

        return outputs


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

        q1_dense = torch.randn(proj_dim, q_size)
        q1_dense = F.normalize(q1_dense, p=2, dim=0)

        q2_dense = torch.randn(proj_dim, q_size)
        q2_dense = F.normalize(q2_dense, p=2, dim=0)

        self.register_buffer(f"q1", q1)
        self.register_buffer(f"q2", q2)
        self.register_buffer(f"q1_dense", q1_dense)
        self.register_buffer(f"q2_dense", q2_dense)
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
        self, x1, dense_fv1, mlp_fv1, rec, x2, dense_fv2, mlp_fv2, indx_unshuffle
    ):
        bs_this = x1.shape[0]
        x1_g = gather_ddp(x1)
        dense_fv1_g = gather_ddp(dense_fv1)
        mlp_fv1_g = gather_ddp(mlp_fv1)
        rec_g = gather_ddp(rec)
        x2_g = gather_ddp(x2)
        dense_fv2_g = gather_ddp(dense_fv2)
        mlp_fv2_g = gather_ddp(mlp_fv2)

        bs_all = x1_g.shape[0]

        n_devices = bs_all // bs_this

        device_indx = torch.distributed.get_rank()
        indx_this = indx_unshuffle.view(n_devices, -1)[device_indx]

        return (
            x1_g[indx_this],
            dense_fv1_g[indx_this],
            mlp_fv1_g[indx_this],
            rec_g[indx_this],
            x2_g[indx_this],
            dense_fv2_g[indx_this],
            mlp_fv2_g[indx_this],
        )

    @torch.no_grad()
    def _update_queue(self, keys1, dense_keys1, keys2, dense_keys2):
        batch_size = keys1.shape[0]
        ptr = int(self.q_ptr)
        assert self.q_size % batch_size == 0

        self.q1[:, ptr : ptr + batch_size] = keys1.T
        self.q2[:, ptr : ptr + batch_size] = keys2.T
        self.q1_dense[:, ptr : ptr + batch_size] = dense_keys1.T
        self.q2_dense[:, ptr : ptr + batch_size] = dense_keys2.T

        self.q_ptr[0] = ptr

    def compute_mlp_logits(self, q, k, queue):
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, queue])
        logit_mlp = torch.cat([l_pos, l_neg], dim=1)
        logit_mlp /= self.temp
        label_mlp = torch.zeros(logit_mlp.shape[0], dtype=torch.long, device=q.device)
        return logit_mlp, label_mlp

    def compute_dense_logits(self, dense_q, dense_k, queue_dense):
        d_pos = torch.einsum("ncm,ncm->nm", dense_q, dense_k).unsqueeze(1)
        d_neg = torch.einsum("ncm,ck->nkm", dense_q, queue_dense)
        logit_dense = torch.cat([d_pos, d_neg], dim=1)
        logit_dense /= self.temp
        label_dense = torch.zeros(
            (logit_dense.shape[0], logit_dense.shape[-1]), dtype=torch.long, device=dense_q.device
        )
        return logit_dense, label_dense

    def forward(self, img_q, img_k, dist=False):

        x1_q, dense_fv1_q, mlp_fv1_q, rec_q, x2_q, dense_fv2_q, mlp_fv2_q = self.model_q(img_q)

        with torch.no_grad():
            self._update_model_k()
            if dist:
                img_k, indx_unshuffle = self._batch_shuffle_ddp(img_k)
            x1_k, dense_fv1_k, mlp_fv1_k, rec_k, x2_k, dense_fv2_k, mlp_fv2_k = self.model_k(img_k)
            if dist:
                (
                    x1_k,
                    dense_fv1_k,
                    mlp_fv1_k,
                    rec_k,
                    x2_k,
                    dense_fv2_k,
                    mlp_fv2_k,
                ) = self._batch_unshuffle_ddp(
                    x1_k,
                    dense_fv1_k,
                    mlp_fv1_k,
                    rec_k,
                    x2_k,
                    dense_fv2_k,
                    mlp_fv2_k,
                    indx_unshuffle,
                )
            cosine = torch.einsum("nca,ncb->nab", x1_q, x1_k)
            pos_indx = cosine.argmax(dim=-1)
            dense_fv1_k = dense_fv1_k.gather(
                2, pos_indx.unsqueeze(1).expand(-1, dense_fv1_k.shape[1], -1)
            )

            cosine = torch.einsum("nca,ncb->nab", x2_q, x2_k)
            pos_indx = cosine.argmax(dim=-1)
            dense_fv2_k = dense_fv2_k.gather(
                2, pos_indx.unsqueeze(1).expand(-1, dense_fv2_k.shape[1], -1)
            )

        logit_mlp1, label_mlp1 = self.compute_mlp_logits(
            mlp_fv1_q, mlp_fv1_k, self.q1.clone().detach()
        )
        logit_mlp2, label_mlp2 = self.compute_mlp_logits(
            mlp_fv2_q, mlp_fv2_k, self.q2.clone().detach()
        )

        logit_dense1, label_dense1 = self.compute_dense_logits(
            dense_fv1_q, dense_fv1_k, self.q1_dense.clone().detach()
        )
        logit_dense2, label_dense2 = self.compute_dense_logits(
            dense_fv2_q, dense_fv2_k, self.q2_dense.clone().detach()
        )

        if dist:
            mlp_fv1_k = gather_ddp(mlp_fv1_k)
            mlp_fv2_k = gather_ddp(mlp_fv2_k)
            dense_fv1_k = gather_ddp(dense_fv1_k.mean(dim=2))
            dense_fv2_k = gather_ddp(dense_fv2_k.mean(dim=2))
        else:
            dense_fv1_k = dense_fv1_k.mean(dim=2)
            dense_fv2_k = dense_fv2_k.mean(dim=2)
        self._update_queue(mlp_fv1_k, dense_fv1_k, mlp_fv2_k, dense_fv2_k)

        return (
            logit_mlp1,
            label_mlp1,
            logit_mlp2,
            label_mlp2,
            logit_dense1,
            label_dense1,
            logit_dense2,
            label_dense2,
            rec_q,
            rec_k,
        )


if __name__ == "__main__":
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    net = Network()
    net = UnsupervisedWrapper(net).cuda()
    out = net(a, b)
    print([o.shape for o in out])
