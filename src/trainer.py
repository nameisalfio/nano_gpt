import torch
import logging
import wandb
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn
from collections import OrderedDict
from torch.optim import Optimizer

from .dataset import get_batch
from .utils import save_checkpoint

logger = logging.getLogger(__name__)

@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, config: DictConfig, device: str) -> dict:
    """Estimate the average loss on the two datasets."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, config.batch_size, config.block_size)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def run_training(
    model: nn.Module, 
    optimizer: Optimizer, 
    train_data: torch.Tensor, 
    val_data: torch.Tensor, 
    config: DictConfig, 
    start_iter: int = 0,
    device: str = "cpu"  
):
    """Training loop."""
    logger.info(f"Starting training from iteration {start_iter} on {device}")
    
    if config.use_wandb:
        wandb_cfg = OmegaConf.to_container(config, resolve=True)
        wandb.init(project=config.project_name, name=config.run_name, config=wandb_cfg)

    for iter in range(start_iter, config.max_iters):

        # Periodic evaluation
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            metrics = estimate_loss(model, train_data, val_data, config, device)
            
            log_data = OrderedDict([
                ("iter", iter),
                ("train_loss", metrics['train']),
                ("val_loss", metrics['val']),
                ("lr", optimizer.param_groups[0]['lr'])
            ])
            
            msg = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log_data.items()])
            logger.info(msg)
            
            if config.use_wandb:
                wandb.log(dict(log_data))

        # Checkpointing 
        if iter > 0 and iter % config.checkpoint_interval == 0:
            ckpt_path = save_checkpoint(model, optimizer, iter, metrics['val'], config.checkpoint_dir, f"ckpt_iter_{iter}")
            logger.info(f"Checkpoint saved at {ckpt_path}")

        # Final save
        if iter == config.max_iters - 1:
            ckpt_path = save_checkpoint(model, optimizer, iter, metrics['val'], config.checkpoint_dir, "last_checkpoint")
            logger.info(f"Last checkpoint reached. Saved at {ckpt_path}")

        # Training Step
        xb, yb = get_batch(train_data, config.batch_size, config.block_size)
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    if config.use_wandb:
        wandb.finish()