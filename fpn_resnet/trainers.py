
import os 
import torch 
import wandb
import pickle
import common
import losses
import eval_utils
import data_utils
import train_utils
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from networks import SimilarityResNet


class MemoryBank:

    def __init__(self, queue_size, feature_size):
        self.bank = torch.FloatTensor(queue_size, feature_size).zero_()
        self.bank = F.normalize(self.bank, dim=-1, p=2)
        self.size = queue_size
        self.ptr = 0 
        
    def add_batch(self, batch):
        for row in batch:
            self.bank[self.ptr] = F.normalize(row, dim=-1, p=2) 
            self.ptr += 1
            if self.ptr >= self.size:
                self.ptr = 0

    def get_vectors(self):
        return self.bank


class Trainer:
    
    def __init__(self, args):
        self.config, self.output_dir, self.logger, self.device = common.initialize_experiment(args, output_root="outputs/resnet_fpn")
        self.train_loader, self.query_loader, self.ref_loader = data_utils.get_loaders(**self.config["data"])                        
        
        self.q_model = SimilarityResNet(**self.config["model"]).to(self.device)
        self.k_model = SimilarityResNet(**self.config["model"]).to(self.device)
        self.optim = train_utils.get_optimizer(self.config["optim"], self.q_model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler({**self.config["scheduler"], "epochs": self.config["epochs"]}, self.optim)
        self.momentum = self.config.get("momentum", 0.999)
        self.memory_bank = MemoryBank(**self.config["memory_bank"])
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optim"]["lr"] - 1e-12) / self.warmup_epochs
        
        for p in self.k_model.parameters():
            p.requires_grad = False
        
        self.start_epoch = 1
        self.best_metric = 0
        run = wandb.init(project="isc-2021")
        self.logger.write(f"WandB run: {run.get_url()}", mode="info")
        self.moco_loss = losses.MocoLoss(**self.config["loss_fn"])
        self.upsample_loss = losses.UpsampleLoss(**self.config["loss_fn"])
        self.query_ref_map = data_utils.process_ground_truth(self.config["public_ground_truth"])
        self.logger.record(f"Model param count: {common.count_parameters(self.model)}", mode="info")
        
        if args["resume"] is not None:
            self.load_state(args["resume"])
        if args["load"] is not None:
            self.load_checkpoint(args["load"])
            
    def save_state(self, epoch):
        state = {
            "epoch": epoch,
            "q_model": self.q_model.state_dict(),
            "k_model": self.k_model.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, "last_state.pth"))
    
    def load_state(self, state_dir):
        if os.path.exists(os.path.join(state_dir, "last_state.pth")):
            state = torch.load(os.path.join(state_dir, "last_state.pth"), map_location=self.device)
            self.start_epoch = state["epoch"] + 1
            self.q_model.load_state_dict(state["q_model"])
            self.k_model.load_state_dict(state["k_model"])
            self.optim.load_state_dict(state["optim"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            self.logger.print("Successfully loaded saved state", mode="info")
        else:
            raise FileNotFoundError("Could not find last_state.pth in specified directory")
        
    def save_checkpoint(self):
        torch.save(self.q_model.state_dict(), os.path.join(self.output_dir, "best_model.pth"))
        
    def load_checkpoint(self, ckpt_dir):
        if os.path.exists(os.path.join(ckpt_dir, "best_model.pth")):
            state = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location=self.device)
            self.q_model.load_state_dict(state)
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
        
    def momentum_update(self):
        for q_param, k_param in zip(self.q_model.parameters(), self.k_model.parameters()):
            k_param.data = self.m * k_param.data + (1.0 - self.m) * q_param.data
        
    def train_on_batch(self, batch):
        aug_1, aug_2 = batch["aug_1"].to(self.device), batch["aug_2"].to(self.device)
        (up_out1, q_logits_1), (up_out2, q_logits_2) = self.q_model(aug_1).values(), self.q_model(aug_2).values()
        k_logits_1, k_logits_2 = self.k_model(aug_1)["features"], self.k_model(aug_2)["features"] 
        loss1 = self.moco_loss(q_logits_1, k_logits_2, self.memory_bank.get_vectors().to(self.device))
        loss2 = self.moco_loss(q_logits_2, k_logits_1, self.memory_bank.get_vectors().to(self.device))
        upsample_loss = self.upsample_loss(up_out1, up_out2)
        loss = loss1 + loss2 + self.config["upsample_loss_lambda"] * upsample_loss
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        self.memory_bank.add_batch(torch.cat([k_logits_1, k_logits_2], 0))
        self.momentum_update()
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def evaluate(self):
        query_features, ref_features = {}, {}
        for step, batch in enumerate(self.query_loader):
            imgs, paths = batch["img"].to(self.device), batch["path"] 
            fvecs = self.q_model(imgs)["features"].detach().cpu()
            fvecs = F.normalize(fvecs, p=2, dim=-1).numpy()
            query_features.update({path: np.expand_dims(vec, 0) for path, vec in zip(paths, fvecs)})
            common.progress_bar(progress=(step+1)/len(self.query_loader), desc="Query features", status="") 
        print()
        for step, batch in enumerate(self.ref_loader):
            imgs, paths = batch["img"].to(self.device), batch["path"] 
            fvecs = self.q_model(imgs)["features"].detach().cpu()
            fvecs = F.normalize(fvecs, p=2, dim=-1).numpy()
            query_features.update({path: np.expand_dims(vec, 0) for path, vec in zip(paths, fvecs)})
            common.progress_bar(progress=(step+1)/len(self.ref_loader), desc="Reference features", status="") 
        print()
        accuracy = eval_utils.compute_neighbor_accuracy(query_features, ref_features, self.query_ref_map)
        return accuracy
    
    def train(self):
        for epoch in range(self.start_epoch, self.config["epochs"]+1):
            desc = "[TRAIN] Epoch {:4d}/{:4d}".format(epoch, self.config["epochs"])
            avg_meter = common.AverageMeter()
            self.model.train()
            
            for step, batch in enumerate(self.train_loader):
                outputs = self.train_on_batch(batch)
                avg_meter.add(outputs)
                wandb.log({"Train loss": outputs["loss"]})
                common.progress_bar(progress=(step+1)/len(self.train_loader), desc=desc, status=avg_meter.return_msg())
            print()
            self.logger.write("Epoch {:4d}/{:4d} {}".format(epoch, self.config["epochs"], avg_meter.return_msg()), mode="train")
            self.adjust_lr(epoch)
            self.save_state(epoch)
            
            # if epoch % self.config["eval_every"] == 0:
            #     accuracy = self.evaluate()
            #     self.logger.record("Epoch {:4d}/{:4d} [Accuracy] {:.4f}".format(epoch, self.config["epochs"], accuracy), mode="val")
            #     wandb.log({"Val accuracy": accuracy, "Epoch": epoch})                           
                
            #     if accuracy > self.best_metric:
            #         self.best_metric = accuracy
            #         self.save_checkpoint()
        print()
        self.logger.record("Completed training.", mode="info")