
import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from training.train import *

# ── Experiment configurations ────────────────────────────────
EXPERIMENTS = {
    "baseline": {
        "label":          "Baseline (6L, 256ctx, 20k steps)",
        "n_layers":       6,
        "context_length": 256,
        "total_steps":    20000,
        "checkpoint_dir": "checkpoints/baseline",
    },
    "steps50k": {
        "label":          "More steps (6L, 256ctx, 50k steps)",
        "n_layers":       6,
        "context_length": 256,
        "total_steps":    50000,
        "checkpoint_dir": "checkpoints/steps50k",
    },
    "layers9": {
        "label":          "More layers (9L, 256ctx, 20k steps)",
        "n_layers":       9,
        "context_length": 256,
        "total_steps":    20000,
        "checkpoint_dir": "checkpoints/layers9",
    },
    "context512": {
        "label":          "Larger context (6L, 512ctx, 20k steps)",
        "n_layers":       6,
        "context_length": 512,
        "total_steps":    20000,
        "checkpoint_dir": "checkpoints/context512",
    },
}


def run_experiment(exp_name: str):
    """Run a single named experiment."""
    if exp_name not in EXPERIMENTS:
        print(f"Unknown experiment '{exp_name}'. Choose from: {list(EXPERIMENTS)}")
        sys.exit(1)

    cfg = EXPERIMENTS[exp_name]

    print("\n" + "═" * 55)
    print(f"  EXPERIMENT: {cfg['label']}")
    print("═" * 55)

    # Override globals from train.py
    global N_LAYERS, CONTEXT_LENGTH, TOTAL_STEPS, CHECKPOINT_DIR
    N_LAYERS       = cfg["n_layers"]
    CONTEXT_LENGTH = cfg["context_length"]
    TOTAL_STEPS    = cfg["total_steps"]
    CHECKPOINT_DIR = cfg["checkpoint_dir"]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Run training
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Which experiment to run"
    )
    args = parser.parse_args()
    run_experiment(args.exp)
