import argparse
import torch
import random
import numpy as np
import utils
from torchvision import transforms
import dataset
from torch.utils.data import DataLoader
import network
import os
import json
from utils import pbar
import faiss
from argparse import Namespace
from PIL import Image
import torch.nn.functional as F


class InferenceModel(torch.nn.Module):
    def __init__(self, model):
        super(InferenceModel, self).__init__()
        self.down1 = model.down1
        self.avg_pool = model.avg_pool
        self.init_up = model.init_up
        self.up1 = model.up1
        self.down2 = model.down2

    def forward(self, x):
        out = self.down1(x)
        x = torch.flatten(self.avg_pool(out[-1]), 1)
        x = self.init_up(x).view(x.shape[0], x.shape[1], 4, 4)
        out = self.up1(x, out)
        out = self.down2(out)
        x = torch.flatten(self.avg_pool(out), 1)
        x = F.normalize(x, p=2, dim=1)
        return x


class Infer:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA Enabled device not found, Exiting!!")

        self.args = args
        self.out_dir = args.output_dir

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.device, local_rank = utils.setup_device(args.dist)
        print(f"Setting up device: {self.device}")

        transform = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if args.split == "test":
            query_dset = dataset.ISC(
                root=args.data_root,
                split="query_all",
                q_transform=transform,
                k_transform=transform,
            )
            ref_dset = dataset.ISC(
                root=args.data_root,
                split="reference_all",
                q_transform=transform,
                k_transform=transform,
            )
            self.query_loader = DataLoader(
                query_dset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
                drop_last=True,
            )
            self.ref_loader = DataLoader(
                ref_dset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
                drop_last=True,
            )
            print(
                f"Loaded q dset with {len(query_dset)} images and r dset with {len(ref_dset)} images"
            )
        else:
            val_dset = dataset.ISC(
                root=args.data_root,
                split="query_val",
                q_transform=transform,
                k_transform=transform,
            )
            self.val_loader = DataLoader(
                val_dset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
                drop_last=True,
            )
            print(
                f"Loaded q dset with {len(val_dset)} images and r dset with {len(val_dset)} images"
            )

        model = network.Network(
            type=args.type,
            skip=args.skip,
            out_dim=args.out_dim,
            proj_dim=args.proj_dim,
        )
        model = network.UnsupervisedWrapper(
            model=model, proj_dim=args.proj_dim, q_size=args.q_size, m=args.m, temp=args.temp
        )
        ckpt = torch.load(args.ckpt, map_location=self.device)
        if "last" in args.ckpt:
            model_dict = ckpt["model"]
        else:
            model_dict = ckpt
        if "module" in list(model_dict.keys())[0]:
            model_dict = {key.replace("module.", ""): value for key, value in model_dict.items()}
        model.load_state_dict(model_dict)
        self.model = InferenceModel(model.model_q).to(self.device)

        self.metric_meter = utils.AvgMeter()

    @torch.no_grad()
    def compute_fvecs(self, name, data_indx=0):
        self.model.eval()
        self.metric_meter.reset()

        loader = getattr(self, name)
        img_list = [tup[data_indx] for tup in loader.dataset.data]
        fvecs = []
        for indx, data in enumerate(loader):
            img = data[data_indx].to(self.device)
            out = self.model(img)

            fvecs.extend(out.detach().cpu().numpy())
            pbar(indx / len(loader))
        pbar(1)
        return img_list, np.array(fvecs)

    @staticmethod
    def find_neighbors(q_fvecs, ref_fvecs, topk=10):
        index = faiss.IndexFlatIP(ref_fvecs.shape[1])
        index.add(ref_fvecs)
        _, indices = index.search(q_fvecs, topk)
        return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="path to config file")
    parser.add_argument("--split", default="val", type=str, help="split choice")
    parser.add_argument("--ckpt", required=True, type=str, help="path to checkpoint file")
    parser.add_argument(
        "--data_root",
        default="/home/sneezygiraffe/data/ISC",
        type=str,
        help="path to data directory root",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="inference batch size")
    parser.add_argument("--topk", default=10, type=int, help="topk elements to retrieve")
    parser.add_argument(
        "--vis", action="store_true", help="visualise query and retrieved samples."
    )
    parser.add_argument("--save_csv", action="store_true", help="create csv submission file.")
    args_new = parser.parse_args()

    cfg = json.load(open(args_new.cfg, "r"))
    args = Namespace(cfg=args_new.cfg, ckpt=args_new.ckpt, split=args_new.split, **cfg)
    args.dist = False
    args.data_root = args_new.data_root
    args.batch_size = args_new.batch_size
    infer = Infer(args)

    if args_new.split == "test":
        q_imgs, q_fvecs = infer.compute_fvecs("query_loader")
        ref_imgs, ref_fvecs = infer.compute_fvecs("ref_loader")
        indices = Infer.find_neighbors(q_fvecs, ref_fvecs, args_new.topk)
    else:
        q_imgs, q_fvecs = infer.compute_fvecs("val_loader", 0)
        ref_imgs, ref_fvecs = infer.compute_fvecs("val_loader", 1)
        indices = Infer.find_neighbors(q_fvecs, ref_fvecs, args_new.topk)

    if args_new.vis:
        for indx, q_img in enumerate(q_imgs):
            img = Image.open(q_img)
            q_dir = os.path.join(args.output_dir, os.path.basename(q_img)[:-4])
            os.makedirs(q_dir, exist_ok=True)
            img = img.save(os.path.join(q_dir, os.path.basename(q_img)))
            for topk, ref_indx in enumerate(indices[indx]):
                img = Image.open(ref_imgs[ref_indx])
                img = img.save(os.path.join(q_dir, f"{topk}.jpg"))
            if args_new.split == "val":
                img = Image.open(ref_imgs[indx])
                img = img.save(os.path.join(q_dir, "GT.jpg"))
            pbar(indx / len(q_imgs))
