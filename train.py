import argparse
import torch
import random
import numpy as np
import helpers
import augment
from torchvision import transforms
import dataset
from torch.utils.data import DataLoader, DistributedSampler
import network
import os
from torch.nn.parallel import DistributedDataParallel
import wandb
import json
from helpers import pbar


class Trainer:
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

        self.device, local_rank = helpers.setup_device(args.dist)
        print(f"Setting up device: {self.device}")

        if args.rand_aug:
            t = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
                augment.RandomAugment(args.n_rand_aug),
                augment.Cutout(args.cut_len),
            ]
        elif args.auto_aug:
            t = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
                augment.Policy(args.policy_num, args.fill_color),
            ]
        elif args.custom_aug:
            t = [
                transforms.Resize((args.img_size, args.img_size)),
                augment.ToNumpy(),
                augment.CustomAugment.augment_image,
                transforms.ToPILImage(),
            ]
        else:
            t = [
                transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            0.8 * args.color_jitter_strength,
                            0.8 * args.color_jitter_strength,
                            0.8 * args.color_jitter_strength,
                            0.2 * args.color_jitter_strength,
                        )
                    ],
                    p=args.color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=args.gray_prob),
                transforms.RandomApply([augment.GaussianBlur(args.blur_sigma)], p=args.blur_prob),
                transforms.RandomHorizontalFlip(),
            ]
        if args.augly_aug:
            t.insert(0, transforms.RandomApply([augment.AuglyTransforms()], p=args.augly_aug_prob))
        train_transform = transforms.Compose(
            [
                *t,
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        train_dset = dataset.ISC(
            root=args.data_root,
            split="train",
            q_transform=train_transform,
            k_transform=train_transform,
        )
        val_dset = dataset.ISC(
            root=args.data_root,
            split="query_val",
            q_transform=val_transform,
            k_transform=val_transform,
        )
        if args.dist:
            train_sampler = DistributedSampler(train_dset)
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.n_workers,
            )
        else:
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_workers,
                drop_last=True,
            )

        self.val_loader = DataLoader(
            val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            drop_last=True,
        )

        model = network.Network(
            type=args.type
        )
        self.model = network.UnsupervisedWrapper(
            model=model, proj_dim=args.proj_dim, q_size=args.q_size, m=args.m, temp=args.temp
        )
        if args.dist:
            torch.set_num_threads(1)
            self.model = DistributedDataParallel(
                self.model.to(self.device),
                device_ids=[local_rank],
                output_device=local_rank,
            )
            self.main_thread = True if int(os.environ.get("RANK")) == 0 else False
        else:
            self.model = self.model.to(self.device)
            self.main_thread = True

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            args.lr,
            weight_decay=args.weight_decay,
        )
        if self.args.lr_step_mode == "epoch":
            total_steps = args.epochs - args.warmup_epochs
        else:
            total_steps = int((args.epochs - args.warmup_epochs) * len(self.train_loader))
        if args.warmup_epochs > 0:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 / args.warmup_epochs * group["lr"]
        self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, total_steps)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        self.log_wandb = False
        self.metric_meter = helpers.AvgMeter()
        if self.main_thread:
            print(
                f"Loaded {train_dset.__class__.__name__} dataset, train: {len(train_dset)}, val: {len(val_dset)}"
            )
            print(f"# of Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6}M")
            if args.wandb:
                self.log_wandb = True
                run = wandb.init()
                print(f"Started wandb logging @ {run.get_url()}")

        if os.path.exists(os.path.join(self.out_dir, "last.ckpt")):
            if args.resume == False:
                raise KeyError(
                    f"Directory {self.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(os.path.join(self.out_dir, "last.ckpt"), map_location=self.device)
            model_dict = ckpt["model"]
            if "module" in list(model_dict.keys())[0] and args.dist == False:
                model_dict = {
                    key.replace("module.", ""): value for key, value in model_dict.items()
                }
            self.model.load_state_dict(model_dict)
            self.optim.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            self.start_epoch = ckpt["epoch"] + 1
            if self.main_thread:
                print(
                    f"=> Loaded checkpoint, resuming training expt from {self.start_epoch} to {args.epochs} epochs."
                )
        else:
            if args.resume == True:
                raise KeyError(
                    f"Resume training args are true but no checkpoint found in {self.out_dir}"
                )
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, "args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
            self.start_epoch = 0
            if self.main_thread:
                print(f"=> Starting fresh training expt for {args.epochs} epochs.")
        self.train_steps = self.start_epoch * len(self.train_loader)

    def train_epoch(self):
        self.model.train()
        self.metric_meter.reset()

        for (indx, (img_q, img_k)) in enumerate(self.train_loader):
            img_q, img_k = img_q.to(self.device), img_k.to(self.device)
            (
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
            ) = self.model(img_q, img_k, self.args.dist)

            mlp1_loss = self.cross_entropy(logit_mlp1, label_mlp1)
            mlp2_loss = self.cross_entropy(logit_mlp2, label_mlp2)
            dense1_loss = self.cross_entropy(logit_dense1, label_dense1)
            dense2_loss = self.cross_entropy(logit_dense2, label_dense2)
            rec_loss = self.mse_loss(rec_q, rec_k)
            down1_loss = (mlp1_loss + dense1_loss) / 2
            down2_loss = (mlp2_loss + dense2_loss) / 2

            w1, w2, w3 = self.args.loss_weights
            loss = w1 * down1_loss + w2 * rec_loss + w3 * down2_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            metrics = {
                "train total loss": loss.item(),
                "train mlp1 loss": mlp1_loss.item(),
                "train dense1 loss": dense1_loss.item(),
                "train down1 loss": down1_loss.item(),
                "train rec loss": rec_loss.item(),
                "train mlp2 loss": mlp2_loss.item(),
                "train dense2 loss": dense2_loss.item(),
                "train down2 loss": down2_loss.item(),
            }
            self.metric_meter.add(metrics)

            if self.main_thread:
                if self.log_wandb:
                    metrics["train step"] = self.train_steps
                    wandb.log(metrics)
                pbar(indx / len(self.train_loader), msg=self.metric_meter.msg())

            if self.args.lr_step_mode == "step":
                if self.train_steps < self.args.warmup_epochs * len(self.train_loader):
                    self.optim.param_groups[0]["lr"] = (
                        self.train_steps
                        / (self.args.warmup_epochs * len(self.train_loader))
                        * self.args.lr
                    )
                else:
                    self.lr_sched.step()

            self.train_steps += 1
        if self.main_thread:
            pbar(1, msg=self.metric_meter.msg())

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        self.metric_meter.reset()

        for (indx, (img_q, img_k)) in enumerate(self.val_loader):
            img_q, img_k = img_q.to(self.device), img_k.to(self.device)
            (
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
            ) = self.model(img_q, img_k, self.args.dist)

            mlp1_loss = self.cross_entropy(logit_mlp1, label_mlp1)
            mlp2_loss = self.cross_entropy(logit_mlp2, label_mlp2)
            dense1_loss = self.cross_entropy(logit_dense1, label_dense1)
            dense2_loss = self.cross_entropy(logit_dense2, label_dense2)
            rec_loss = self.mse_loss(rec_q, rec_k)
            down1_loss = (mlp1_loss + dense1_loss) / 2
            down2_loss = (mlp2_loss + dense2_loss) / 2

            w1, w2, w3 = self.args.loss_weights
            loss = w1 * down1_loss + w2 * rec_loss + w3 * down2_loss

            metrics = {
                "val total loss": loss.item(),
                "val mlp1 loss": mlp1_loss.item(),
                "val dense1 loss": dense1_loss.item(),
                "val down1 loss": down1_loss.item(),
                "val rec loss": rec_loss.item(),
                "val mlp2 loss": mlp2_loss.item(),
                "val dense2 loss": dense2_loss.item(),
                "val down2 loss": down2_loss.item(),
            }
            self.metric_meter.add(metrics)

            if self.main_thread:
                pbar(indx / len(self.val_loader), msg=self.metric_meter.msg())

        if self.main_thread:
            pbar(1, msg=self.metric_meter.msg())

    def run(self):
        best_train, best_val = float("inf"), float("inf")
        for epoch in range(self.start_epoch, self.args.epochs):
            if self.main_thread:
                print(f"Epoch: {epoch}")
                print("---------------")

            self.train_epoch()

            if self.main_thread:
                train_metrics = self.metric_meter.get()
                train_loss = train_metrics["train total loss"]
                if train_loss < best_train:
                    print(
                        "\x1b[34m"
                        + f"train loss improved from {round(best_train, 5)} to {round(train_loss, 5)}"
                        + "\033[0m"
                    )
                    best_train = train_loss

                if (epoch + 1) % self.args.val_every == 0:
                    self.eval()
                    val_metrics = self.metric_meter.get()
                    val_loss = val_metrics["val total loss"]
                    if val_loss < best_val:
                        print(
                            "\x1b[33m"
                            + f"val loss improved from {round(best_val, 5)} to {round(val_loss, 5)}"
                            + "\033[0m"
                        )
                        best_val = val_loss
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.out_dir, "best.ckpt"),
                        )

                    if self.log_wandb:
                        train_metrics = {"epoch " + key: value for key, value in train_metrics.items()}
                        val_metrics = {"epoch " + key: value for key, value in val_metrics.items()}
                        wandb.log(
                            {
                                "epoch": epoch,
                                **train_metrics,
                                **val_metrics,
                                "lr": self.optim.param_groups[0]["lr"],
                            }
                        )
                else:
                    if self.log_wandb:
                        train_metrics = {"epoch " + key: value for key, value in train_metrics.items()}
                        wandb.log(
                            {
                                "epoch": epoch,
                                **train_metrics,
                                "lr": self.optim.param_groups[0]["lr"],
                            }
                        )

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        "lr_sched": self.lr_sched.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.out_dir, "last.ckpt"),
                )

            if self.args.lr_step_mode == "epoch":
                if epoch < self.args.warmup_epochs:
                    self.optim.param_groups[0]["lr"] = (
                        epoch / self.args.warmup_epochs * self.args.lr
                    )
                else:
                    self.lr_sched.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = helpers.add_args(parser)
    args = parser.parse_args()

    helpers.print_args(args)

    trainer = Trainer(args)
    trainer.run()

    if args.dist:
        torch.distributed.destroy_process_group()
