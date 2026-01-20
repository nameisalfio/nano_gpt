import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size: int, n_embd: int, block_size: int) -> None:
        """ 
        Initialize a single head of Self-Attention. 
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (C**-0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # masking to avoid attending to future tokens
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # Aggregation of values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v    # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out # (B, T, head_size)
    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, device: str) -> None:
        """
        Initialize the Bigram Language Model with Self-Attention.
        """
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size=n_embd, n_embd=n_embd, block_size=block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        """
        B, T = idx.shape

        # Compute embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd) where T is the sequence length
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, n_embd)
        
        # Sum of content and position to perform broadcasting (that is )
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Apply attention
        x = self.sa_head(x) # (B, T, n_embd)
        
        # Compute final logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def predict(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx