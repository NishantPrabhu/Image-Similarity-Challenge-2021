
import os 
import torch 
import wandb
import pickle
import numpy as np
import torch.nn as nn 
from networks import densenet
from utils import common, train_utils, eval_utils, data_utils


class Trainer:
    
    def __init__(self, args):
        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root="outputs/densenet")
        self.train_loader, self.val_loader = data_utils.get_mini_loaders(**self.config["data"])                        
        
        self.model = densenet.SimilarityDensenetModel(**self.config["model"]).to(self.device)
        self.optim = train_utils.get_optimizer(self.config["optim"], self.model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({**self.config["scheduler"], "epochs": self.config["epochs"]}, self.optim)
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optim"]["lr"] - 1e-12) / self.warmup_epochs
        
        run = wandb.init(project="isc-2021")
        self.logger.write(f"WandB run: {run.get_url()}", mode="info")
        self.loss_fn = nn.TripletMarginLoss(**self.config["loss_fn"])
        self.start_epoch = 1
        self.best_metric = 0
        
        with open(self.config["val_labels"], "rb") as f:
            self.val_labels = pickle.load(f)
        
        if args["resume"] is not None:
            self.load_state(args["resume"])
        if args["load"] is not None:
            self.load_checkpoint(args["load"])
            
    def save_state(self, epoch):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, "last_state.pth"))
    
    def load_state(self, state_dir):
        if os.path.exists(os.path.join(state_dir, "last_state.pth")):
            state = torch.load(os.path.join(state_dir, "last_state.pth"), map_location=self.device)
            self.start_epoch = state["epoch"] + 1
            self.model.load_state_dict(state["model"])
            self.optim.load_state_dict(state["optim"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            self.logger.print("Successfully loaded saved state", mode="info")
        else:
            raise FileNotFoundError("Could not find last_state.pth in specified directory")
        
    def save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pth"))
        
    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "best_model.pth")):
            state = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.logger.print("Successfully loaded model checkpoint", mode="info")
        else:
            raise FileNotFoundError("Could not find best_model.pth in specified directory")
        
    def adjust_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 + self.warmup_rate * epoch
        elif self.scheduler is not None:
            self.scheduler.step()
        else:
            pass
        
    def train_on_batch(self, batch):
        anchor, pos, neg = batch["anchor"].to(self.device), batch["pos"].to(self.device), batch["neg"].to(self.device)
        anchor_logits, pos_logits, neg_logits = self.model(anchor), self.model(pos), self.model(neg)
        loss = self.loss_fn(anchor_logits, pos_logits, neg_logits)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def evaluate(self):
        features = {}
        for step in range(len(self.val_loader)):
            batch = self.val_loader.get()
            imgs, paths = batch["img"].to(self.device), batch["path"] 
            fvecs = self.model(imgs).detach().cpu().numpy()
            features.update({path: np.expand_dims(vec, 0) for path, vec in zip(paths, fvecs)})
            common.progress_bar(progress=(step+1)/len(self.val_loader), desc="Generating features", status="")
        print()
        iou_score = eval_utils.compute_neighbor_accuracy(features, self.val_labels)
        return iou_score     
    
    def train(self):
        for epoch in range(0, self.config["epochs"]+1):
            desc = "[TRAIN] Epoch {:4d}/{:4d}".format(epoch, self.config["epochs"])
            avg_meter = common.AverageMeter()
            self.model.train()
            
            for step, batch in enumerate(self.train_loader):
                outputs = self.train_on_batch(batch)
                avg_meter.add(outputs)
                wandb.log({"Train loss": outputs["loss"]})
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc=desc, status=avg_meter.return_msg())
                break
            print()
            self.logger.write("Epoch {:4d}/{:4d} {}".format(epoch, self.config["epochs"], avg_meter.return_msg()), mode="train")
            self.adjust_lr(epoch)
            self.save_state(epoch)
            
            if epoch % self.config["eval_every"] == 0:
                iou = self.evaluate()
                self.logger.write("Epoch {:4d}/{:4d} [IoU score] {:.4f}".format(epoch, self.config["epochs"], iou), mode="val")
                wandb.log({"Val IoU": iou, "Epoch": epoch})                           
                
                if iou > self.best_metric:
                    self.best_metric = iou
                    self.save_checkpoint()
        print()
        self.logger.record("Completed training.", mode="info")