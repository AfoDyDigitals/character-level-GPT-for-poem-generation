 """
train.py
========
Training loop for the character-level GPT model.

Features:
    - Character-level tokenisation (builds vocab from corpus)
    - Train / val split loading
    - AdamW optimiser with linear learning rate warm-up
    - Loss + bits-per-character logging
    - Periodic poetry sample generation
    - Checkpoint saving
    - Flash vs Classic attention speed/memory comparison

Usage (in Colab):
    !python training/train.py
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn

# ── Make sure Python can find model.py regardless of working directory ──
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import CharacterGPT, GPTConfig


# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════

# ── Paths ────────────────────────────────────────────────────
TRAIN_FILE   = "data/train.txt"
VAL_FILE     = "data/val.txt"
CHECKPOINT_DIR = "checkpoints"

# ── Training hyperparameters ─────────────────────────────────
# These match the lab's suggested starting point.
# Change these for Stage III (hyperparameter exploration).
CONTEXT_LENGTH = 256
N_LAYERS       = 6
N_HEADS        = 8
D_MODEL        = 256
MLP_RATIO      = 4
DROPOUT        = 0.1

BATCH_SIZE     = 32
LEARNING_RATE  = 3e-4
WARMUP_STEPS   = 500       # linear LR warm-up over first N steps
TOTAL_STEPS    = 20000     # increase to 50k for better results

# ── Logging & saving ─────────────────────────────────────────
LOG_INTERVAL      = 100    # print loss every N steps
EVAL_INTERVAL     = 500    # evaluate on val set every N steps
EVAL_STEPS        = 50     # how many val batches to average over
SAMPLE_INTERVAL   = 2000   # generate a poem sample every N steps
CHECKPOINT_INTERVAL = 5000 # save a checkpoint every N steps

SAMPLE_LENGTH  = 300       # characters to generate per sample
SAMPLE_PROMPT  = "\n"      # start generation from a newline (blank line)
TEMPERATURE    = 0.8       # generation temperature (0.8 = slightly focused)
TOP_K          = 40        # restrict sampling to top-40 characters

# ── Device ───────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str):
    """
    Build a character-level vocabulary from the corpus.

    Returns:
        vocab     : sorted list of unique characters
        char2idx  : dict mapping character → integer index
        idx2char  : dict mapping integer index → character
    """
    vocab    = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    return vocab, char2idx, idx2char


def encode(text: str, char2idx: dict) -> torch.Tensor:
    """Convert a string to a 1D tensor of integer indices."""
    return torch.tensor([char2idx[c] for c in text if c in char2idx],
                        dtype=torch.long)


def decode(indices: torch.Tensor, idx2char: dict) -> str:
    """Convert a 1D tensor of integer indices back to a string."""
    return "".join(idx2char[i.item()] for i in indices)


def get_batch(data: torch.Tensor, batch_size: int, context_length: int, device: str):
    """
    Sample a random batch of (input, target) pairs from the data tensor.

    For character-level language modelling:
        input[i]  = data[start : start+context_length]
        target[i] = data[start+1 : start+context_length+1]
    i.e. targets are inputs shifted by one character.
    """
    # Pick random starting positions
    starts = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[s   : s + context_length    ] for s in starts])
    y = torch.stack([data[s+1 : s + context_length + 1] for s in starts])
    return x.to(device), y.to(device)


# ══════════════════════════════════════════════════════════════
# 3. EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, eval_steps, device):
    """
    Estimate validation loss by averaging over eval_steps random batches.
    Returns average loss and bits-per-character (BPC).

    Bits-per-character = loss / log(2)
    Measures how many bits the model needs on average to predict each character.
    A perfect model → 0 BPC. A random model → log2(vocab_size) BPC.
    """
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = get_batch(val_data, batch_size, context_length, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()

    avg_loss = sum(losses) / len(losses)
    bpc = avg_loss / math.log(2)   # convert nats → bits
    return avg_loss, bpc


# ══════════════════════════════════════════════════════════════
# 4. SAMPLE GENERATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_sample(model, prompt: str, char2idx: dict, idx2char: dict,
                    max_new_tokens: int, temperature: float, top_k: int,
                    device: str) -> str:
    """Generate a poetry sample from the model given a string prompt."""
    model.eval()

    # Encode the prompt; fall back to newline if chars are unknown
    prompt_clean = "".join(c for c in prompt if c in char2idx)
    if not prompt_clean:
        prompt_clean = "\n"

    idx = encode(prompt_clean, char2idx).unsqueeze(0).to(device)   # (1, T)
    output_idx = model.generate(idx, max_new_tokens, temperature, top_k)
    generated = decode(output_idx[0], idx2char)

    model.train()
    return generated


# ══════════════════════════════════════════════════════════════
# 5. LEARNING RATE SCHEDULE
# ══════════════════════════════════════════════════════════════

def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """
    Linear warm-up learning rate schedule.
    - Steps 0..warmup_steps: LR rises linearly from 0 to base_lr
    - After warmup: LR stays constant at base_lr

    Why warm-up? In the first steps, weights are random and gradients
    are noisy. A small LR prevents the model from making huge, unstable
    updates early on.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


# ══════════════════════════════════════════════════════════════
# 6. FLASH VS CLASSIC ATTENTION COMPARISON
# ══════════════════════════════════════════════════════════════

def compare_attention_implementations(model, train_data, batch_size,
                                      context_length, device, n_steps=20):
    """
    Compare Flash Attention vs Classic masked attention.
    Measures wall-clock time and peak GPU memory for both implementations.
    Prints a summary table for your report.
    """
    print("\n" + "═" * 50)
    print("  Attention Implementation Comparison")
    print("═" * 50)

    results = {}

    for use_flash, label in [(True, "Flash Attention"), (False, "Classic Attention")]:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()

        for _ in range(n_steps):
            x, y = get_batch(train_data, batch_size, context_length, device)
            _, loss = model(x, y, use_flash=use_flash)
            loss.backward()
            model.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - t0
        steps_per_sec = n_steps / elapsed

        if device == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_mem_mb = float("nan")

        results[label] = {"steps/s": steps_per_sec, "peak_mem_MB": peak_mem_mb}
        print(f"  {label:<20} | {steps_per_sec:.1f} steps/s "
              f"| peak mem: {peak_mem_mb:.0f} MB")

    if device != "cuda":
        print("  (Memory stats only available on GPU)")

    print("═" * 50 + "\n")
    return results


# ══════════════════════════════════════════════════════════════
# 7. MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 55)
    print("  Character-Level GPT — Training")
    print("=" * 55)
    print(f"  Device: {DEVICE}")

    # ── Load data ────────────────────────────────────────────
    print("\n→ Loading corpus...")
    train_text = load_corpus(TRAIN_FILE)
    val_text   = load_corpus(VAL_FILE)

    # Build vocabulary from training data only
    vocab, char2idx, idx2char = build_vocab(train_text)
    vocab_size = len(vocab)
    print(f"  Vocab size : {vocab_size} unique characters")
    print(f"  Train size : {len(train_text):,} characters")
    print(f"  Val size   : {len(val_text):,} characters")

    # Save vocabulary for later use (analysis, generation)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR, "vocab.json"), "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f)
    print("  Vocab saved to checkpoints/vocab.json")

    # Encode corpora as integer tensors
    train_data = encode(train_text, char2idx)
    val_data   = encode(val_text,   char2idx)

    # ── Build model ──────────────────────────────────────────
    print("\n→ Building model...")
    config = GPTConfig(
        vocab_size     = vocab_size,
        context_length = CONTEXT_LENGTH,
        n_layers       = N_LAYERS,
        n_heads        = N_HEADS,
        d_model        = D_MODEL,
        mlp_ratio      = MLP_RATIO,
        dropout        = DROPOUT,
    )
    model = CharacterGPT(config).to(DEVICE)
    print(f"  {model}")

    # ── Attention comparison (do this once before training) ──
    compare_attention_implementations(
        model, train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE
    )

    # ── Optimiser ────────────────────────────────────────────
    # AdamW with weight decay only on weight matrices (not biases or LayerNorms)
    decay_params    = [p for n, p in model.named_parameters()
                       if p.dim() >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.dim() < 2 and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=LEARNING_RATE)

    # ── Training history (for plotting later) ────────────────
    history = {
        "step": [], "train_loss": [], "val_loss": [], "val_bpc": []
    }

    # ── Training loop ────────────────────────────────────────
    print("→ Starting training...\n")
    model.train()
    t_start = time.time()

    for step in range(1, TOTAL_STEPS + 1):

        # Update learning rate (warm-up schedule)
        lr = get_lr(step, WARMUP_STEPS, LEARNING_RATE)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Forward + backward pass
        x, y = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ── Logging ──────────────────────────────────────────
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed
            print(f"  step {step:>6}/{TOTAL_STEPS} | "
                  f"loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | "
                  f"{steps_per_sec:.1f} steps/s")

            history["step"].append(step)
            history["train_loss"].append(loss.item())

        # ── Validation ───────────────────────────────────────
        if step % EVAL_INTERVAL == 0:
            val_loss, val_bpc = evaluate(
                model, val_data, BATCH_SIZE, CONTEXT_LENGTH, EVAL_STEPS, DEVICE
            )
            print(f"\n  ── Val @ step {step} ──")
            print(f"     val loss : {val_loss:.4f}")
            print(f"     val BPC  : {val_bpc:.4f} bits/char\n")

            history["val_loss"].append(val_loss)
            history["val_bpc"].append(val_bpc)

            # Save history for plotting
            with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
                json.dump(history, f)

        # ── Sample generation ─────────────────────────────────
        if step % SAMPLE_INTERVAL == 0:
            print(f"\n  ── Generated sample @ step {step} ──")
            sample = generate_sample(
                model, SAMPLE_PROMPT, char2idx, idx2char,
                SAMPLE_LENGTH, TEMPERATURE, TOP_K, DEVICE
            )
            print(sample)
            print("  " + "─" * 40 + "\n")

        # ── Checkpointing ────────────────────────────────────
        if step % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_step{step}.pt")
            torch.save({
                "step":        step,
                "config":      config.__dict__,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "history":     history,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved: {ckpt_path}\n")

    # ── Final checkpoint ─────────────────────────────────────
    torch.save({
        "step":        TOTAL_STEPS,
        "config":      config.__dict__,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "history":     history,
    }, os.path.join(CHECKPOINT_DIR, "ckpt_final.pt"))

    total_time = time.time() - t_start
    print(f"\n✓ Training complete in {total_time/60:.1f} minutes")
    print(f"  Final checkpoint saved to {CHECKPOINT_DIR}/ckpt_final.pt")


# ══════════════════════════════════════════════════════════════
# 8. PLOTTING (run after training)
# ══════════════════════════════════════════════════════════════

def plot_training_curves(history_path: str = "checkpoints/history.json"):
    """
    Load saved training history and plot loss + BPC curves.
    Run this in a Colab cell after training:
        from training.train import plot_training_curves
        plot_training_curves()
    """
    import matplotlib.pyplot as plt

    with open(history_path) as f:
        h = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(h["step"], h["train_loss"], label="Train loss", alpha=0.8)
    val_steps = [h["step"][i] for i in range(0, len(h["step"]), len(h["step"]) // max(1, len(h["val_loss"])))]
    val_steps = val_steps[:len(h["val_loss"])]
    ax1.plot(val_steps, h["val_loss"], label="Val loss", linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # BPC curve
    ax2.plot(val_steps, h["val_bpc"], color="orange", linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Bits per character")
    ax2.set_title("Validation Bits-per-Character")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("checkpoints/training_curves.png", dpi=150)
    plt.show()
    print("✓ Plot saved to checkpoints/training_curves.png")


if __name__ == "__main__":
    train()