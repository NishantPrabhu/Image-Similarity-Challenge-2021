
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_optimizer(config, params):
    """
    Initializes an optimizer with provided configuration.
    """
    name = config.get("name", "sgd")
    if name == "sgd":
        return optim.SGD(
            params=params, lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9, nesterov=True)
    elif name == "adam":
        return optim.Adam(
            params=params, lr=config["lr"], weight_decay=config["weight_decay"], eps=config.get("epsilon", 1e-06), amsgrad=config.get("amsgrad", False))
    elif name == "adamw":
        return optim.AdamW(
            params=params, lr=config["lr"], weight_decay=config["weight_decay"], eps=config.get("epsilon", 1e-06), amsgrad=config.get("amsgrad", False))
    else:
        raise NotImplementedError(f"Invalid optimizer {name}") 

def get_scheduler(config, optimizer):
    """
    Initializes a scheduler with provided configuration.
    """
    name = config.get("name", None)
    warmup_epochs = config.get("warmup_epochs", 0)

    if warmup_epochs > 0:
        max_lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = 1e-12 + max_lr / warmup_epochs 

    if name is not None:
        if name == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"] - warmup_epochs, eta_min=0.0, last_epoch=-1)
        elif name == "multistep":
            scheduler = lr_scheduler.MultiStepLR(optimizer, config["milestones"], config["gamma"])
        else:
            raise NotImplementedError(f"Invalid scheduler {name}")
        return scheduler, warmup_epochs

    else:
        return None, warmup_epochs