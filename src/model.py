import torch
import torch.nn as nn
from torch.nn import functional as F

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
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
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