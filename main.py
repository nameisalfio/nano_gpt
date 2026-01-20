from src.dataset import load_data, get_vocab_info, prepare_data, get_batch

text_name = "data/tiny_shakespeare.txt"
text = load_data()
print(f"Loaded text {text_name}:\n{text[:500]}\n")
print("..."*10 + "\n")

vocab_size, encode, decode = get_vocab_info(text)
print(f"Vocabulary size: {vocab_size}\n")

train_data, val_data = prepare_data(text, encode)
print(f"Train data: {len(train_data)}\t Val data: {len(val_data)}\n")

x_batch, y_batch = get_batch(train_data, batch_size=4, block_size=8)
print("Batch obtained")
print(f"X: {len(x_batch)}, shape: {x_batch.shape}\nY: {len(y_batch)}, shape: {y_batch.shape}\n")
print(f"X: {x_batch[:50]}\nY: {y_batch[:50]}")
