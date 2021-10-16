import torch.nn as nn
import torch
import math
from copy import deepcopy

class Scale(nn.Module):
    def __init__(self, mode="down"):
        super(Scale, self).__init__()
        if mode == "down":
            self.scale = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.scale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        cls_token = x[:, [0]]
        rem_tokens = x[:, 1:]
        s = int(rem_tokens.shape[1]**0.5)
        rem_tokens = rem_tokens.view(rem_tokens.shape[0], s, s, rem_tokens.shape[-1])
        rem_tokens = rem_tokens.permute(0, 3, 1, 2)
        rem_tokens = self.scale(rem_tokens)
        rem_tokens = rem_tokens.view(rem_tokens.shape[0], rem_tokens.shape[1], -1)
        rem_tokens = rem_tokens.permute(0, 2, 1)
        return torch.cat((cls_token, rem_tokens), dim=1)

class Network(nn.Module):
    TYPES = [
        "vits16",
        "vits8",
        "vitb16",
        "vitb8",
    ]
    def __init__(self, type="vits8"):
        super(Network, self).__init__()
        assert type in self.TYPES
        pretrained = torch.hub.load("facebookresearch/dino:main", f"dino_{type}")

        self.patch_embed = pretrained.patch_embed
        self.cls_token = pretrained.cls_token
        self.pos_embed = pretrained.pos_embed
        self.pos_drop = pretrained.pos_drop
        self.embed_dim = self.cls_token.shape[-1]

        blocks = list(pretrained.blocks.children())

        self.block1 = blocks[0]
        self.block2 = blocks[1]
        self.block3 = blocks[2]
        self.block4 = blocks[3]
        self.block5 = blocks[4]
        self.block6 = blocks[5]
        self.block7 = blocks[6]
        self.block8 = blocks[7]
        self.block9 = blocks[8]
        self.block10 = blocks[9]
        self.block11 = blocks[10]
        self.block12 = blocks[11]

        self.down = Scale(mode="down")
        self.up = Scale(mode="up")

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        
        # down
        o1 = self.block1(x)
        o2 = self.block2(self.down(o1))
        o3 = self.block3(self.down(o2))
        o4 = self.block4(self.down(o3))

        # up
        o5 = self.block5(o4)
        o6 = self.block6(self.up(o5) + o3)
        o7 = self.block7(self.up(o6) + o2)
        o8 = self.block8(self.up(o7) + o1)

        # down
        o9 = self.block9(o8)
        o10 = self.block10(self.down(o9) + o7)
        o11 = self.block11(self.down(o10) + o6)
        o12 = self.block12(self.down(o11) + o5)

        g1, g2, g3 = o4[:, 0], o8[:, 0], o12[:, 0]
        l1, l2, l3 = o4[:, 1:], o8[:, 1:], o12[:, 1:]
        return l1, g1, l2, g2, l3, g3


def proj_init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.02)
        m.bias.data.zero_()

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )
        self.apply(proj_init_weights)

    def forward(self, x):
        out = self.fc(x)
        out = nn.functional.normalize(out, p=2, dim=-1)
        return out

class Unsupervised(nn.Module):
    def __init__(self, model, proj_dim=256):
        super(Unsupervised, self).__init__()
        self.model = model
        self.g1_fc = ProjectionHead(self.model.embed_dim, proj_dim)
        self.l1_fc = ProjectionHead(self.model.embed_dim, proj_dim)
        self.g3_fc = ProjectionHead(self.model.embed_dim, proj_dim)
        self.l3_fc = ProjectionHead(self.model.embed_dim, proj_dim)

    def forward(self, x):
        x1, g1, x2, g2, x3, g3 = self.model(x)
        g1 = self.g1_fc(g1)
        g3 = self.g3_fc(g3)
        l1 = self.l1_fc(x1).permute(0, 2, 1)
        l3 = self.l3_fc(x3).permute(0, 2, 1)

        x1 = nn.functional.normalize(x1, p=2, dim=1).permute(0, 2, 1)
        x3 = nn.functional.normalize(x3, p=2, dim=1).permute(0, 2, 1)

        return x1, l1, g1, x2, x3, l3, g3


@torch.no_grad()
def gather_ddp(t):
    t_g = [torch.ones_like(t) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(t_g, t, async_op=False)
    out = torch.cat(t_g, dim=0)
    return out

class Regulariser(nn.Module):

    def __init__(self):
        super(Regulariser, self).__init__()
        self.pdist = nn.PairwiseDistance(2)
    
    def forward(self, x):

        I = pairwise_NNs_inner(x)
        distances = self.pdist(x, x[I])
        loss_uniform = - torch.log(distances).mean()
        return loss_uniform


def pairwise_NNs_inner(x):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I


class UnsupervisedWrapper(nn.Module):
    def __init__(self, model = 'vits8', proj_dim=256, q_size=65536, m=0.999, temp=0.07, margin = 0.5):
        super(UnsupervisedWrapper, self).__init__()
        self.margin = margin
        self.m = m
        self.temp = temp
        self.q_size = q_size

        self.model_q = torch.hub.load('facebookresearch/dino:main', 'dino_' + model).cuda()
        self.model_k = deepcopy(self.model_q)

        for param_k in self.model_k.parameters():
            param_k.requires_grad = False

        q1 = torch.randn(proj_dim, q_size)
        q1 = nn.functional.normalize(q1, p=2, dim=0)

        q2 = torch.randn(proj_dim, q_size)
        q2 = nn.functional.normalize(q2, p=2, dim=0)

        q1_dense = torch.randn(proj_dim, q_size)
        q1_dense = nn.functional.normalize(q1_dense, p=2, dim=0)

        q2_dense = torch.randn(proj_dim, q_size)
        q2_dense = nn.functional.normalize(q2_dense, p=2, dim=0)

        self.register_buffer(f"q1", q1)
        self.register_buffer("q_ptr", torch.zeros(1, dtype=torch.long))

        self.regulariser = Regulariser()

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
    def _update_queue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.q_ptr)
        assert self.q_size % batch_size == 0

        self.q1[:, ptr : ptr + batch_size] = keys.T
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

        q_global = self.model_q(img_q)

        n = q_global.shape[0]

        with torch.no_grad():
            self._update_model_k()
            if dist:
                img_k, indx_unshuffle = self._batch_shuffle_ddp(img_k)
            k_global = self.model_k(img_k)
            if dist:
                k_global = self._batch_unshuffle_ddp(
                    k_global,
                    indx_unshuffle
                )
            
        sim_mat = torch.matmul(q_global, k_global.t())
        epsilon = 1e-5

        loss = list()

        neg_count = list()

        for i in range(n):

        
            
            pos_pair_ = torch.masked_select(sim_mat[i], sim_mat[i] < 1 - epsilon)
            
            neg_pair = torch.masked_select(sim_mat[i], sim_mat[i] > self.margin)

            pos_loss = torch.sum(-pos_pair_ + 1)

            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))

            else:
                neg_loss = 0
            
            loss.append(pos_loss + neg_loss)

        loss = sum(loss)/n

        if dist:
            k_global = gather_ddp(k_global)
            
        self._update_queue(k_global)

        reg = self.regulariser(q_global)

        return (
            loss, 
            reg        
        )


if __name__ == "__main__":
    
    net = UnsupervisedWrapper().cuda()

    img1 = torch.randn(2, 3, 64, 64).cuda()
    img2 = torch.randn(2, 3, 64, 64).cuda()
    out = net(img1, img2)
    print([o.shape for o in out])