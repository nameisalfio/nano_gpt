from src.dataset import load_data, get_vocab_info, prepare_data, get_batch

text_name = "data/tiny_shakespeare.txt"
text = load_data()
print(f"Loaded text {text_name}:\n{text[:500]}")

vocab_size, encode, decode = get_vocab_info(text)
print(f"Vocabulary size: {vocab_size}")

train_data, val_data = prepare_data(text, encode)
print(f"Train data: {len(train_data)}\t Val data: {len(val_data)}")

x_batch, y_batch = get_batch(train_data, batch_size=4, block_size=8)
print("Batch obtained")
print(f"X: {len(train_data)}\t Y: {len(val_data)}")
print(f"X: {train_data[:50]}\nY: {val_data[:50]}")
