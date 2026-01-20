import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.optim import Optimizer
from typing import Union 

def set_seed(seed: int) -> None:
    """Ensures reproducibility of experiments."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(
    model: nn.Module, 
    optimizer: Optimizer, 
    iter: int, 
    loss: Union[float, torch.Tensor], 
    ckpt_dir: Union[str, Path], 
    name: str
) -> Path:
    """Saves the state of the model and optimizer."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{name}.pt"
    
    torch.save({
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
    }, checkpoint_path)
    
    return checkpoint_path

def load_checkpoint(
    path: str | Path, 
    model: nn.Module, 
    optimizer: Optimizer, 
    device: str
) -> int:
    """
    Loads the state of the model and optimizer from a file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at: {path}")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iter']