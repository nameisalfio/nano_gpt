import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import BigramLanguageModel
from src.config import Config
from src.dataset import load_data, get_vocab_info

def get_attention_matrix(model, idx):
    """
    Extracts the attention weights from the first Head of the last Block.
    """
    model.eval()
    B, T = idx.shape
    
    with torch.no_grad():
        x = model.token_embedding_table(idx) + model.position_embedding_table(torch.arange(T, device=Config.device))
        
        # Take the last block (the last in the self.blocks sequence)
        last_block = model.blocks[-1]
        head = last_block.sa.heads[0]  # First head of the last block
        q = head.query(last_block.ln1(x))
        k = head.key(last_block.ln1(x))
        
        # Compute weights as in the model
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1) # (B, T, T)
        
    return wei[0].cpu().numpy()

def main():
    text = load_data("data/tiny_shakespeare.txt")
    vocab_size, encode, decode = get_vocab_info(text)
    
    model = BigramLanguageModel(
        vocab_size=vocab_size, n_embd=Config.n_embd, n_head=Config.n_head,
        n_layer=Config.n_layer, block_size=Config.block_size, 
        dropout=Config.dropout, device=Config.device
    ).to(Config.device)
    
    # Load the last checkpoint
    checkpoint = torch.load("checkpoints/ckpt_iter_9999.pt", map_location=Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    prompt = "ROMEO: Shall I"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=Config.device)
    tokens = [decode([t]) for t in encode(prompt)]

    # Get the attention matrix
    attn_matrix = get_attention_matrix(model, idx)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=False)
    plt.title("Attention Map - Last Block (Head 0)")
    plt.xlabel("Key (Source Tokens)")
    plt.ylabel("Query (Target Tokens)")
    plt.show()

if __name__ == "__main__":
    main()