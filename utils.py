import torch
import numpy as np
import os
from datetime import datetime


def print_args(args):
    print("\n---- Experiment Configuration ----")
    args_ = vars(args)
    for arg, value in args_.items():
        print(f" * {arg} => {value}")
    print("----------------------------------")


def add_args(parser):
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [default: year-month-date_hour-minute].",
    )
    parser.add_argument("--wandb", action="store_true", help="start wandb logging.")
    parser.add_argument("--dist", action="store_true", help="start distributed training.")
    parser.add_argument("--seed", type=int, default=42, help="set experiment seed.")
    parser.add_argument("--rand_aug", action="store_true", help="start random augmentation.")
    parser.add_argument(
        "--auto_aug",
        action="store_true",
        help="start auto augmentation using identified best policies.",
    )
    parser.add_argument("--img_size", type=int, default=256, help="set image size.")
    parser.add_argument(
        "--rand_aug_n", type=int, default=4, help="set number of sequential random augs."
    )
    parser.add_argument("--cut_len", type=int, default=128, help="cutout size.")
    parser.add_argument("--policy_num", type=int, default=1, help="auto augment policy number.")
    parser.add_argument(
        "--fill_color",
        type=tuple,
        default=(128, 128, 128),
        help="fill color size for auto augment.",
    )
    parser.add_argument(
        "--color_jitter_strength", type=float, default=0.5, help="color jitter strength."
    )
    parser.add_argument(
        "--color_jitter_prob", type=float, default=0.8, help="color jitter probability."
    )
    parser.add_argument("--gray_prob", type=float, default=0.2, help="gray probability.")
    parser.add_argument("--blur_sigma", type=list, default=[0.1, 2], help="blur sigma.")
    parser.add_argument("--blur_prob", type=float, default=0.5, help="blur probability.")
    parser.add_argument("--data_root", type=str, required=True, help="dataset directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of workers for dataloading."
    )
    parser.add_argument("--type", type=str, default="resnet18", help="network backbone type.")
    parser.add_argument(
        "--skip",
        type=bool,
        default=False,
        help="use skip connections between first down and upblocks.",
    )
    parser.add_argument("--out_dim", type=int, default=256, help="final feature vector dimension.")
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=256,
        help="projection feature dimension for self supervised learning.",
    )
    parser.add_argument("--q_size", type=int, default=65536, help="queue size.")
    parser.add_argument("--m", type=float, default=0.999, help="momentum update.")
    parser.add_argument("--temp", type=float, default=0.07, help="contrastive loss temperature.")
    parser.add_argument("--lr", type=float, default=1, help="sgd learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd optimizer momentum.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="sgd optimizer weight decay."
    )
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="number of warmup epochs.")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint.")
    parser.add_argument(
        "--loss_weights", type=list, default=[1, 1, 1], help="loss weights (down1, up1, down2)."
    )
    parser.add_argument(
        "--lr_step_mode",
        type=str,
        default="epoch",
        help="choose lr step mode, choose one of [epoch, step]",
    )
    return parser


def setup_device(dist):
    if dist:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = None
        device = torch.device("cuda:0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device, local_rank


def pbar(p=0, msg="", bar_len=20):
    msg = msg.ljust(50)
    block = int(round(bar_len * p))
    text = "\rProgress: [{}] {}% {}".format(
        "\x1b[32m" + "=" * (block - 1) + ">" + "\033[0m" + "-" * (bar_len - block),
        round(p * 100, 2),
        msg,
    )
    print(text, end="")
    if p == 1:
        print()


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, batch_metrics):
        if self.metrics == {}:
            for key, value in batch_metrics.items():
                self.metrics[key] = [value]
        else:
            for key, value in batch_metrics.items():
                self.metrics[key].append(value)

    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        avg_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])
