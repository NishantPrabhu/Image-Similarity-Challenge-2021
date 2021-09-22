
import os 
import argparse 
import trainers
from utils import data_utils
from datetime import datetime as dt


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=False, type=str, help="Path to configuration file")
    ap.add_argument("-t", "--task", required=True, type=str, choices=["train", "val_aug"], help="Task to perform")
    ap.add_argument("-o", "--output", default=dt.now().strftime("%d-%m-%Y_%H-%M"), type=str, help="Path to directory where logs and checkpoints are stored")
    ap.add_argument("-l", "--load", default=None, type=str, help="Path to directory containing model checkpoint to load (for inference only)")
    ap.add_argument("-r", "--resume", default=None, type=str, help="Path to directory containing saved state to load (for resuming training)")
    args = vars(ap.parse_args())
    
    if args["task"] == "train":
        trainer = trainers.Trainer(args)
        trainer.train() 
    
    elif args["task"] == "val_aug":
        data_utils.augment_val_set(dir="data/mini/val/", emoji_dir="data/mini/meta/", num_augs=4)