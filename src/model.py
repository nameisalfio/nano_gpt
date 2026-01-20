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
        
        # Aggregazione dei valori
        v = self.value(x) # (B, T, head_size)
        out = wei @ v    # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out # (B, T, head_size)
    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        """
        Initializes the BigramLanguageModel.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model. Computes logits and optionally the loss.
        """
        # Compute logits using the embedding table
        logits = self.token_embedding_table(idx)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            # Reshape logits and targets to compute cross-entropy loss
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def predict(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates a sequence of new tokens starting from an initial context.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, V)
            probs = F.softmax(logits, dim=-1)  # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx # (B, T + max_new_tokens)
    
