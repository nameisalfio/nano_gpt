import torch
from sklearn.model_selection import train_test_split

def load_data(path: str="data/tiny_shakespeare.txt") -> str:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_vocab_info(text: str) -> tuple[int, callable, callable]:
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # take a string as input and outputs a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # take a list of integer as input and outputs a string
    return vocab_size, encode, decode

def prepare_data(text, encode_func, train_split: int = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.tensor(encode_func(text), dtype=torch.long)
    train_data, val_data = train_test_split(data, test_size=train_split, random_state=42, shuffle=False)
    return train_data, val_data

def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = torch.randint(0, len(data) - block_size, (batch_size,))
    x_batch = torch.stack([data[i:i+block_size] for i in start_indices])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in start_indices])
    return x_batch, y_batch