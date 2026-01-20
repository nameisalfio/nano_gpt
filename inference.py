import torch
import logging
import argparse
from src.config import Config
from src.dataset import load_data, get_vocab_info
from src.model import BigramLanguageModel
from src.utils import load_checkpoint

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

@torch.no_grad()
def predict(model, idx, max_new_tokens, temperature=1.0):
    """
    Advanced version of predict with temperature support.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def main():
    parser = argparse.ArgumentParser(description="Inference script for Nano-GPT")
    parser.add_argument("--prompt", type=str, default="\n", help="Initial text for generation")
    parser.add_argument("--max_tokens", type=int, default=500, help="Number of tokens to generate")
    parser.add_argument("--ckpt", type=str, default="checkpoints/ckpt_iter_9999.pt", help="Path to the checkpoint")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (0.1 deterministic, 1.5 creative)")
    args = parser.parse_args()

    # 1. Vocabulary preparation
    text = load_data("data/tiny_shakespeare.txt")
    vocab_size, encode, decode = get_vocab_info(text)

    # 2. Model initialization with GPT architecture
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=Config.n_embd,
        n_head=Config.n_head,
        n_layer=Config.n_layer,
        block_size=Config.block_size,
        dropout=Config.dropout,
        device=Config.device
    ).to(Config.device)

    # 3. Loading weights from the checkpoint
    try:
        logger.info(f"Loading checkpoint from: {args.ckpt}")
        # We use a dummy optimizer because load_checkpoint requires it,
        # but here we are only interested in the model weights
        dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        load_checkpoint(args.ckpt, model, dummy_optimizer, Config.device)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return

    # 4. Generation
    logger.info(f"Generating (Temp: {args.temp})...")
    
    # Encode the prompt
    start_ids = encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=Config.device).unsqueeze(0) # (1, T)

    # Perform generation (using sampling logic with temperature)
    generated_indices = predict(model, x, args.max_tokens, args.temp)
    
    print("\n" + "="*50)
    print(decode(generated_indices[0].tolist()))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()