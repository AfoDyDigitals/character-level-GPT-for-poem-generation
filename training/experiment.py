import os, sys, argparse, json
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import CharacterGPT, GPTConfig
from training.train import (load_corpus, build_vocab, encode, get_batch,
                             evaluate, generate_sample, get_lr,
                             compare_attention_implementations,
                             TRAIN_FILE, VAL_FILE,
                             BATCH_SIZE, LEARNING_RATE, WARMUP_STEPS,
                             LOG_INTERVAL, EVAL_INTERVAL, EVAL_STEPS,
                             SAMPLE_INTERVAL, CHECKPOINT_INTERVAL,
                             SAMPLE_LENGTH, SAMPLE_PROMPT, TEMPERATURE, TOP_K)

EXPERIMENTS = {
    "layers9": {
        "label":          "More layers (9L, 256ctx, 20k steps)",
        "n_layers":       9,
        "context_length": 256,
        "total_steps":    20000,
        "checkpoint_dir": "/content/drive/MyDrive/lab3_checkpoints/layers9",
    },
    "steps50k": {
        "label":          "More steps (6L, 256ctx, 50k steps)",
        "n_layers":       6,
        "context_length": 256,
        "total_steps":    50000,
        "checkpoint_dir": "/content/drive/MyDrive/lab3_checkpoints/steps50k",
    },
}

def run_experiment(exp_name):
    cfg = EXPERIMENTS[exp_name]
    print(f"\n{'═'*55}\n  EXPERIMENT: {cfg['label']}\n{'═'*55}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = cfg["checkpoint_dir"]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load data
    train_text = load_corpus(TRAIN_FILE)
    val_text   = load_corpus(VAL_FILE)
    vocab, char2idx, idx2char = build_vocab(train_text)
    with open(os.path.join(CHECKPOINT_DIR, "vocab.json"), "w") as f:
        json.dump({"char2idx": char2idx,
                   "idx2char": {str(k): v for k, v in idx2char.items()}}, f)
    train_data = encode(train_text, char2idx)
    val_data   = encode(val_text,   char2idx)

    # Build model with experiment-specific config
    config = GPTConfig(
        vocab_size     = len(vocab),
        context_length = cfg["context_length"],
        n_layers       = cfg["n_layers"],    # ← correctly passed now
        n_heads        = 8,
        d_model        = 256,
        dropout        = 0.1,
    )
    model = CharacterGPT(config).to(DEVICE)
    print(f"  {model}")

    # Optimiser
    decay_params    = [p for n, p in model.named_parameters()
                       if p.dim() >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.dim() < 2 and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=LEARNING_RATE)

    history = {"step": [], "train_loss": [], "val_loss": [], "val_bpc": []}
    import time, math
    model.train()
    t_start = time.time()

    for step in range(1, cfg["total_steps"] + 1):
        lr = get_lr(step, WARMUP_STEPS, LEARNING_RATE)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, BATCH_SIZE, cfg["context_length"], DEVICE)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            print(f"  step {step:>6}/{cfg['total_steps']} | "
                  f"loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"{step/elapsed:.1f} steps/s")
            history["step"].append(step)
            history["train_loss"].append(loss.item())

        if step % EVAL_INTERVAL == 0:
            val_loss, val_bpc = evaluate(model, val_data, BATCH_SIZE,
                                         cfg["context_length"], EVAL_STEPS, DEVICE)
            print(f"\n  ── Val @ step {step} ──")
            print(f"     val loss : {val_loss:.4f}")
            print(f"     val BPC  : {val_bpc:.4f}\n")
            history["val_loss"].append(val_loss)
            history["val_bpc"].append(val_bpc)
            with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
                json.dump(history, f)

        if step % SAMPLE_INTERVAL == 0:
            print(f"\n  ── Sample @ step {step} ──")
            sample = generate_sample(model, SAMPLE_PROMPT, char2idx, idx2char,
                                     SAMPLE_LENGTH, TEMPERATURE, TOP_K, DEVICE)
            print(sample)

        if step % CHECKPOINT_INTERVAL == 0:
            torch.save({
                "step": step, "config": config.__dict__,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "history": history,
            }, os.path.join(CHECKPOINT_DIR, f"ckpt_step{step}.pt"))
            print(f"  ✓ Checkpoint saved to Drive\n")

    torch.save({
        "step": cfg["total_steps"], "config": config.__dict__,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "history": history,
    }, os.path.join(CHECKPOINT_DIR, "ckpt_final.pt"))

    total_time = time.time() - t_start
    print(f"\n✓ Done in {total_time/60:.1f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    run_experiment(args.exp)
