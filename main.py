import torch
from src.dataset import load_data, get_vocab_info, prepare_data, get_batch
from src.model import BigramLanguageModel

text = load_data("data/tiny_shakespeare.txt")
vocab_size, encode, decode = get_vocab_info(text)
train_data, val_data = prepare_data(text, encode)

model = BigramLanguageModel(vocab_size)
xb, yb = get_batch(train_data, batch_size=32, block_size=8)
logits, loss = model(xb, yb)
print(f"Logits dimensions (B*T, V): {logits.shape}")
print(f"Initial Loss (expected around {torch.log(torch.tensor(vocab_size)).item():.4f}): {loss.item():.4f}")

context = torch.zeros((1, 1), dtype=torch.long) 
generated_indices = model.generate(context, max_new_tokens=100)[0].tolist()
print("\n--- GENERATION TEST (UNTRAINED MODEL) ---")
print(decode(generated_indices))
