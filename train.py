import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset import load_data, get_vocab_info, prepare_data
from src.model import BigramLanguageModel
from src.trainer import run_training
from src.utils import set_seed, load_checkpoint

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    
    logger.info("Loaded configuration:\n" + OmegaConf.to_yaml(cfg))
    
    # 1. Data Loading 
    logger.info(f"Loading data from: {cfg.dataset.path}")
    text = load_data(cfg.dataset.path)
    vocab_size, encode, decode = get_vocab_info(text)
    train_data, val_data = prepare_data(text, encode, train_split=cfg.dataset.train_split)
    
    logger.info(f"Vocabulary: {vocab_size} characters.")
    logger.info(f"Split: {len(train_data)} train, {len(val_data)} val.")

    # 2. Model Initialization 
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        dropout=cfg.model.dropout,
        block_size=cfg.model.block_size,
        device=cfg.device
    ).to(cfg.device)

    # 3. Optimizer 
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.trainer.learning_rate, 
        weight_decay=cfg.trainer.get('weight_decay', 0.1)
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model instantiated. Total parameters: {n_params}")

    # 4. Resume Logic 
    start_iter = 0
    if cfg.trainer.resume_path:
        try:
            logger.info(f"Attempting to resume from: {cfg.trainer.resume_path}")
            start_iter = load_checkpoint(
                cfg.trainer.resume_path, 
                model, 
                optimizer, 
                cfg.device
            )
            start_iter += 1 
            logger.info(f"Resumed from iteration {start_iter}")
        except Exception as e:
            logger.error(f"Resume error: {e}. Starting from scratch.")

    # 5. Training Execution
    run_training(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        config=cfg.trainer, 
        start_iter=start_iter,
        device=cfg.device
    )

if __name__ == "__main__":
    main()