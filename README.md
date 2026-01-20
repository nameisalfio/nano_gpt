# Nano-GPT 

This project implements a generative language model based on the GPT (Generative Pre-trained Transformer) architecture, trained on the Tiny Shakespeare dataset. The model was developed from the basics (Bigram) to include modern Self-Attention mechanisms.

---

## Model Structure

The architecture follows the "decoder-only" philosophy of GPT models, optimized for Deep Learning on modern hardware (Apple Silicon MPS/NVIDIA CUDA).

### 1. Embedding System

Unlike classical statistical models, this model operates in a continuous vector space:

- **Token Embedding**: Each character in the vocabulary is mapped to a dense vector of dimension `n_embd`.
- **Positional Embedding**: Since Self-Attention is agnostic to token order, positional vectors are added to provide the model with a sense of sequence.

---

### 2. The Self-Attention Mechanism

The core of the model is the `Head` class, which implements **Scaled Dot-Product Attention**. Each token emits three vectors:

- **Query (Q)**: What the current token is looking for.
- **Key (K)**: What the token offers to be selected.
- **Value (V)**: The information the token communicates if selected.

The attention between tokens is calculated using the formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Masking (Tril)** ensures that during training, each token can only look at the past, maintaining the model's autoregressive nature.

---

### 3. Multi-Head Attention & FeedForward

- **Multi-Head Attention**: Multiple heads are used in parallel to allow the model to learn different relationships simultaneously (e.g., one head for punctuation, another for grammar).
- **FeedForward Network (FFWD)**: After the communication phase (Attention), each token processes the information independently through a two-layer linear neural network with ReLU activation.

---

### 4. Stability and Depth

To enable the training of deep networks (multiple `Block`), the following were implemented:

- **Residual Connections (Skip Connections)**: The input is added to the output of each block to facilitate gradient flow.
- **Layer Normalization**: Applied before each operation (Pre-norm) to stabilize activations.
- **Dropout**: Used to prevent overfitting during training.

---

## Technical Specifications of the Checkpoint (Iteration 10000)

The latest trained model has the following parameters:

- **Total Parameters**: ~1.2 Million.
- **Context Window (Block Size)**: 256 tokens.
- **Embedding Dimension**: 128.
- **Transformer Layers**: 6.
- **Attention Heads**: 4.
- **Final Loss**: ~1.16 (Train) / ~1.47 (Val).

---

## Results Interpretation

### Attention Visualization

The model is no longer a "black box." Through the visualization script, it is possible to observe how the model focuses attention on specific tokens (such as the character `:` in dialogues or the initial letters of names) to maintain the structural coherence of Shakespearean text.

---

## Usage

### Training

```bash
python train.py
```

The training process automatically handles checkpoint saving and resumption.

### Generation (Inference)

```bash
python inference.py --prompt "ROMEO: " --temp 0.8 --max_tokens 500
```

- The `--temp` parameter controls the model's creativity:
    - Low values (e.g., `0.5`) produce safer and more deterministic texts.
    - High values (e.g., `1.2`) increase variety at the expense of coherence.
