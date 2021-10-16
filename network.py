import torch.nn as nn
import torch
import math
from copy import deepcopy

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
        self, x, indx_unshuffle
    ):
        bs_this = x.shape[0]
        x_g = gather_ddp(x)

        bs_all = x_g.shape[0]

        n_devices = bs_all // bs_this

        device_indx = torch.distributed.get_rank()
        indx_this = indx_unshuffle.view(n_devices, -1)[device_indx]

        return (
            x_g[indx_this],
        )

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