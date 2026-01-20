import torch
import logging

from src.config import Config
from src.dataset import load_data, get_vocab_info, prepare_data
from src.model import BigramLanguageModel
from src.trainer import run_training
from src.utils import set_seed, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    set_seed(Config.seed)
    
    logger.info("Loading data...")
    text = load_data("data/tiny_shakespeare.txt")
    vocab_size, encode, decode = get_vocab_info(text)
    train_data, val_data = prepare_data(text, encode)
    logger.info(f"Vocabulary created: {vocab_size} unique characters.")
    logger.info(f"Dataset split: {len(train_data)} tokens for training, {len(val_data)} tokens for validation.")

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=Config.n_embd,
        n_head=Config.n_head,
        n_layer=Config.n_layer,
        dropout=Config.dropout,
        block_size=Config.block_size,
        device=Config.device
    ).to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model instantiated. Total parameters: {n_params}")

    start_iter = 0
    if hasattr(Config, 'resume_path') and Config.resume_path:
        try:
            logger.info(f"Attempting to resume from checkpoint: {Config.resume_path}")
            start_iter = load_checkpoint(
                Config.resume_path, 
                model, 
                optimizer, 
                Config.device
            )
            start_iter += 1 
            logger.info(f"Resume completed. Resuming from iteration {start_iter}")
        except Exception as e:
            logger.error(f"Error during resume: {e}. Starting from scratch.")

    # Avvio training con lo start_iter calcolato
    run_training(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        config=Config,
        start_iter=start_iter
    )

if __name__ == "__main__":
    main()