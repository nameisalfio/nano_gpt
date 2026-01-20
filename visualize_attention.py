import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hydra
from omegaconf import DictConfig
from typing import List, Tuple

from src.model import BigramLanguageModel
from src.dataset import load_data, get_vocab_info

def get_attention_matrix(model: BigramLanguageModel, idx: torch.Tensor, device: str) -> np.ndarray:
    """
    Extracts the attention weights from the first Head of the last Block.
    """
    model.eval()
    B, T = idx.shape
    
    with torch.no_grad():
        # Use the explicitly passed device
        x = model.token_embedding_table(idx) + model.position_embedding_table(torch.arange(T, device=device))
        
        # Recursive access to model modules
        last_block = model.blocks[-1]
        head = last_block.sa.heads[0] 
        
        q = head.query(last_block.ln1(x))
        k = head.key(last_block.ln1(x))
        
        # Compute weights (Scaled Dot-Product Attention)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1) 
        
    return wei[0].cpu().numpy()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # Load data and vocabulary info
    text: str = load_data(config.dataset.path)
    vocab_size: int
    encode: callable
    decode: callable
    vocab_size, encode, decode = get_vocab_info(text)
    
    # Initialize model 
    model = BigramLanguageModel(
        vocab_size=vocab_size, 
        n_embd=config.model.n_embd, 
        n_head=config.model.n_head,
        n_layer=config.model.n_layer, 
        block_size=config.model.block_size, 
        dropout=config.model.dropout, 
        device=config.device
    ).to(config.device)
    
    # Load weights 
    ckpt_path: str = config.trainer.resume_path or "checkpoints/last_checkpoints.pt"
    print(f"Loading weights from: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Prepare Prompt
    prompt: str = "ROMEO: Shall I"
    idx: torch.Tensor = torch.tensor([encode(prompt)], dtype=torch.long, device=config.device)
    tokens: List[str] = [decode([t]) for t in encode(prompt)]

    attn_matrix: np.ndarray = get_attention_matrix(model, idx, config.device)

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=False)
    plt.title(f"Attention Map - {config.model.name} (Head 0)")
    plt.xlabel("Key (Source Tokens)")
    plt.ylabel("Query (Target Tokens)")
    plt.show()

if __name__ == "__main__":
    main()