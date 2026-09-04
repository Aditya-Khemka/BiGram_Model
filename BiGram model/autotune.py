"""
Autotuning script for a GPT-style language model.

This script automatically searches for optimal hyperparameters such as:
- batch size
- number of transformer layers
- embedding dimension
- number of attention heads
- learning rate

It performs short "probe" training runs to estimate validation loss
without doing full expensive training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import mmap
import itertools
import math
import gc

# -------------------------------------------------
# 1. GLOBAL CONFIGURATION
# -------------------------------------------------

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset parameters
block_size = 128           # context length
vocab_size = 65            # update to your tokenizer vocab size

# Training probe parameters
probe_steps = 300          # short training for autotune
eval_batches = 20          # number of val batches for scoring

# -------------------------------------------------
# 2. HYPERPARAMETER SEARCH SPACE
# -------------------------------------------------

BATCH_SIZES = [8, 16, 32]
N_LAYERS = [4, 6, 8]
N_EMBD = [64, 128, 256]
N_HEADS = [4, 8]
LEARNING_RATES = [3e-4, 2e-4, 1e-4]

# -------------------------------------------------
# 3. TOKENIZER (PLACEHOLDER)
# -------------------------------------------------
# Replace encode() with your actual tokenizer

def encode(text):
    return [ord(c) % vocab_size for c in text]

# -------------------------------------------------
# 4. MEMORY-MAPPED DATA LOADING
# -------------------------------------------------

def get_random_chunk(split, batch_size):
    """
    Reads a random chunk of text from a large file using mmap.
    This allows training on huge datasets without loading them into RAM.
    """
    filename = "openwebtext/output_train.txt" if split == "train" else "openwebtext/output_val.txt"

    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)

            # Random start position in the file
            start_pos = random.randint(0, file_size - block_size * batch_size)
            mm.seek(start_pos)

            # Read enough bytes for batch sampling
            raw_bytes = mm.read(block_size * batch_size - 1)

    text = raw_bytes.decode("utf-8", errors="ignore").replace("\r", "")
    tokens = torch.tensor(encode(text), dtype=torch.long)

    return tokens


def get_batch(split, batch_size):
    """
    Creates a batch of (input, target) token sequences for training.
    """
    data = get_random_chunk(split, batch_size)

    # Random offsets inside the chunk
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

# -------------------------------------------------
# 5. TRANSFORMER MODEL DEFINITION
# -------------------------------------------------

class Head(nn.Module):
    """Single self-attention head"""

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask to prevent looking ahead
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Scaled dot-product attention
        wei = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer decoder block"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """GPT-style decoder-only language model"""

    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# -------------------------------------------------
# 6. GPU MEMORY SAFETY CHECK
# -------------------------------------------------

def can_fit_on_gpu(model):
    """
    Checks if a model fits in GPU memory by running a dummy forward pass.
    """
    try:
        dummy = torch.zeros((1, block_size), dtype=torch.long, device=device)
        model(dummy)
        del dummy
        torch.cuda.empty_cache()
        return True
    except RuntimeError:
        torch.cuda.empty_cache()
        return False

# -------------------------------------------------
# 7. SHORT TRAINING PROBE
# -------------------------------------------------

def train_probe(config):
    """
    Trains the model for a small number of steps and returns validation loss.
    Used only for hyperparameter comparison.
    """
    torch.manual_seed(42)

    batch_size = config["batch_size"]
    n_layer = config["n_layer"]
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    lr = config["lr"]

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head
    ).to(device)

    if not can_fit_on_gpu(model):
        del model
        return float("inf")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(probe_steps):
        xb, yb = get_batch("train", batch_size)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Validation loss
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = get_batch("val", batch_size)
            _, loss = model(xb, yb)
            losses.append(loss.item())

    val_loss = sum(losses) / len(losses)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss

# -------------------------------------------------
# 8. AUTOTUNING LOOP
# -------------------------------------------------

def autotune():
    """
    Runs grid search over hyperparameters and returns the best configuration.
    """
    best_loss = float("inf")
    best_config = None

    search_space = []

    for bs, nl, ne, nh, lr in itertools.product(
        BATCH_SIZES, N_LAYERS, N_EMBD, N_HEADS, LEARNING_RATES
    ):
        if ne % nh == 0:
            search_space.append({
                "batch_size": bs,
                "n_layer": nl,
                "n_embd": ne,
                "n_head": nh,
                "lr": lr
            })

    print(f"Total configs to test: {len(search_space)}")

    for i, config in enumerate(search_space):
        print(f"\n[{i+1}/{len(search_space)}] Testing {config}")
        val_loss = train_probe(config)
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config
            print("🔥 New best configuration found!")

    return best_config, best_loss

# -------------------------------------------------
# 9. ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    best_config, best_loss = autotune()

    print("\n✅ AUTOTUNING COMPLETE")
    print("Best config:", best_config)
    print("Best validation loss:", best_loss)
