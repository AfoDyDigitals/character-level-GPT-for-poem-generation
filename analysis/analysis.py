
"""
analysis.py
===========
Stage IV: Analyse the structural properties of generated poems
and compare them against the training corpus.

Measures:
    1. Line length distribution
    2. Stanza structure (lines per stanza)
    3. Rhyme detection (line-ending rhymes)
    4. Vocabulary overlap with training corpus
    5. Generates poems with different temperature settings

Usage (in Colab):
    !python analysis/analysis.py

Outputs (saved to analysis/results/):
    - generated_poems.txt       : raw generated poems
    - structure_comparison.png  : line length + stanza plots
    - rhyme_analysis.png        : rhyme frequency plot
    - summary_stats.json        : all numbers for the report
"""

import os
import sys
import json
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import CharacterGPT, GPTConfig
from training.train import generate_sample

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

REPO_ROOT       = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoints/baseline/ckpt_final.pt")
VOCAB_PATH      = os.path.join(REPO_ROOT, "checkpoints/baseline/vocab.json")
CORPUS_PATH     = os.path.join(REPO_ROOT, "data/train.txt")
OUTPUT_DIR      = os.path.join(REPO_ROOT, "analysis/results")

N_POEMS         = 20       # number of poems to generate for analysis
POEM_LENGTH     = 500      # characters per generated poem
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Different temperatures to explore
TEMPERATURES = [0.6, 0.8, 1.0, 1.2]


# ══════════════════════════════════════════════════════════════
# 1. LOAD MODEL
# ══════════════════════════════════════════════════════════════

def load_model(checkpoint_path, vocab_path, device):
    """Load a trained model and vocabulary from disk."""
    print(f"→ Loading model from {checkpoint_path}")

    with open(vocab_path) as f:
        vocab_data = json.load(f)
    char2idx = vocab_data["char2idx"]
    idx2char = {int(k): v for k, v in vocab_data["idx2char"].items()}

    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model  = CharacterGPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  ✓ Loaded: {model.count_parameters():,} parameters")
    return model, char2idx, idx2char


# ══════════════════════════════════════════════════════════════
# 2. STRUCTURAL ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def split_into_stanzas(text: str) -> list[list[str]]:
    """
    Split a poem text into stanzas (separated by blank lines).
    Each stanza is a list of non-empty lines.
    """
    stanzas = []
    for block in re.split(r"\n\n+", text.strip()):
        lines = [l for l in block.split("\n") if l.strip()]
        if lines:
            stanzas.append(lines)
    return stanzas


def get_line_lengths(text: str) -> list[int]:
    """Return the character length of each non-empty line."""
    return [len(l) for l in text.split("\n") if l.strip()]


def get_stanza_sizes(text: str) -> list[int]:
    """Return the number of lines in each stanza."""
    return [len(s) for s in split_into_stanzas(text)]


def get_last_word(line: str) -> str:
    """Extract the last alphabetic word from a line (for rhyme detection)."""
    words = re.findall(r"[a-zA-Z]+", line)
    return words[-1].lower() if words else ""


def get_ending_sound(word: str) -> str:
    """
    Simple rhyme approximation: last 3 characters of a word.
    Not a perfect phonetic rhyme detector, but effective for
    detecting statistical patterns in endings.
    """
    return word[-3:] if len(word) >= 3 else word


def detect_rhymes(text: str) -> dict:
    """
    Detect rhyming patterns in a poem.

    Checks the four most common schemes:
        AABB — consecutive pairs rhyme
        ABAB — alternating lines rhyme
        ABBA — envelope rhyme

    Returns a dict with rhyme scheme counts and rhyme rate.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return {"rhyme_rate": 0.0, "aabb": 0, "abab": 0, "abba": 0}

    endings = [get_ending_sound(get_last_word(l)) for l in lines]

    # Count rhyming pairs for each scheme
    aabb = sum(1 for i in range(0, len(endings)-1, 2)
               if endings[i] == endings[i+1] and endings[i])
    abab = sum(1 for i in range(len(endings)-2)
               if endings[i] == endings[i+2] and endings[i])
    abba = sum(1 for i in range(len(endings)-3)
               if endings[i] == endings[i+3] and endings[i])

    # Overall rhyme rate: what fraction of consecutive line-end pairs rhyme
    pairs = [(endings[i], endings[i+1]) for i in range(len(endings)-1)
             if endings[i] and endings[i+1]]
    rhyme_rate = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else 0.0

    return {
        "rhyme_rate": rhyme_rate,
        "aabb": aabb,
        "abab": abab,
        "abba": abba,
    }


def vocabulary_overlap(generated_text: str, corpus_text: str) -> dict:
    """
    Measure what fraction of words in the generated text
    also appear in the training corpus.
    A high overlap means the model learned real words.
    """
    gen_words    = set(re.findall(r"[a-zA-Z]+", generated_text.lower()))
    corpus_words = set(re.findall(r"[a-zA-Z]+", corpus_text.lower()))

    overlap  = gen_words & corpus_words
    fraction = len(overlap) / len(gen_words) if gen_words else 0.0

    return {
        "unique_words_generated": len(gen_words),
        "unique_words_in_corpus": len(corpus_words),
        "overlap_count":          len(overlap),
        "overlap_fraction":       fraction,
    }


# ══════════════════════════════════════════════════════════════
# 3. GENERATE POEMS
# ══════════════════════════════════════════════════════════════

def generate_poem_collection(model, char2idx, idx2char, n_poems,
                              poem_length, temperature, device) -> list[str]:
    """Generate a collection of poems at a given temperature."""
    poems = []
    prompts = ["\n", "I ", "The ", "And ", "O "]   # varied prompts

    for i in range(n_poems):
        prompt = prompts[i % len(prompts)]
        poem = generate_sample(
            model, prompt, char2idx, idx2char,
            max_new_tokens=poem_length,
            temperature=temperature,
            top_k=40,
            device=device,
        )
        poems.append(poem)

    return poems


# ══════════════════════════════════════════════════════════════
# 4. PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_structure_comparison(corpus_text, generated_poems, output_dir):
    """
    Plot side-by-side comparison of corpus vs generated poems:
        - Line length distributions
        - Stanza size distributions
    """
    gen_text = "\n\n".join(generated_poems)

    corpus_line_lens   = get_line_lengths(corpus_text)
    generated_line_lens = get_line_lengths(gen_text)

    corpus_stanza_sizes   = get_stanza_sizes(corpus_text)
    generated_stanza_sizes = get_stanza_sizes(gen_text)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Corpus vs Generated: Structural Comparison", fontsize=14)

    # Line length — corpus
    axes[0, 0].hist(corpus_line_lens, bins=40, color="steelblue", alpha=0.8)
    axes[0, 0].set_title("Corpus: Line Length Distribution")
    axes[0, 0].set_xlabel("Characters per line")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(sum(corpus_line_lens)/len(corpus_line_lens),
                       color="red", linestyle="--", label=f"Mean: {sum(corpus_line_lens)/len(corpus_line_lens):.1f}")
    axes[0, 0].legend()

    # Line length — generated
    axes[0, 1].hist(generated_line_lens, bins=40, color="darkorange", alpha=0.8)
    axes[0, 1].set_title("Generated: Line Length Distribution")
    axes[0, 1].set_xlabel("Characters per line")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(sum(generated_line_lens)/len(generated_line_lens),
                       color="red", linestyle="--", label=f"Mean: {sum(generated_line_lens)/len(generated_line_lens):.1f}")
    axes[0, 1].legend()

    # Stanza size — corpus
    stanza_counter_c = Counter(min(s, 12) for s in corpus_stanza_sizes)
    axes[1, 0].bar(stanza_counter_c.keys(), stanza_counter_c.values(), color="steelblue", alpha=0.8)
    axes[1, 0].set_title("Corpus: Stanza Size Distribution")
    axes[1, 0].set_xlabel("Lines per stanza")
    axes[1, 0].set_ylabel("Frequency")

    # Stanza size — generated
    stanza_counter_g = Counter(min(s, 12) for s in generated_stanza_sizes)
    axes[1, 1].bar(stanza_counter_g.keys(), stanza_counter_g.values(), color="darkorange", alpha=0.8)
    axes[1, 1].set_title("Generated: Stanza Size Distribution")
    axes[1, 1].set_xlabel("Lines per stanza")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    path = os.path.join(output_dir, "structure_comparison.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  ✓ Saved: {path}")

    return {
        "corpus_mean_line_len":    sum(corpus_line_lens) / len(corpus_line_lens),
        "generated_mean_line_len": sum(generated_line_lens) / len(generated_line_lens),
        "corpus_mean_stanza_size":    sum(corpus_stanza_sizes) / len(corpus_stanza_sizes),
        "generated_mean_stanza_size": sum(generated_stanza_sizes) / len(generated_stanza_sizes),
    }


def plot_rhyme_analysis(poems_by_temp, output_dir):
    """
    Plot rhyme rates across different generation temperatures.
    Shows how temperature affects the model's tendency to rhyme.
    """
    temps       = sorted(poems_by_temp.keys())
    rhyme_rates = []

    for temp in temps:
        poems = poems_by_temp[temp]
        rates = [detect_rhymes(p)["rhyme_rate"] for p in poems]
        rhyme_rates.append(sum(rates) / len(rates))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temps, rhyme_rates, marker="o", linewidth=2, color="purple")
    ax.set_title("Rhyme Rate vs Generation Temperature")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Rhyme rate (fraction of consecutive line pairs)")
    ax.set_ylim(0, max(rhyme_rates) * 1.5 + 0.01)
    ax.grid(alpha=0.3)

    for t, r in zip(temps, rhyme_rates):
        ax.annotate(f"{r:.3f}", (t, r), textcoords="offset points",
                    xytext=(0, 10), ha="center")

    plt.tight_layout()
    path = os.path.join(output_dir, "rhyme_analysis.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  ✓ Saved: {path}")

    return dict(zip(temps, rhyme_rates))


def plot_training_curves_comparison(exp_dirs: dict, output_dir: str):
    """
    Plot training curves for all experiments side by side.
    exp_dirs: {label: path_to_history.json}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, history_path in exp_dirs.items():
        if not os.path.exists(history_path):
            print(f"  ✗ Missing: {history_path}")
            continue
        with open(history_path) as f:
            h = json.load(f)

        steps   = h["step"]
        val_bpc = h["val_bpc"]
        # Align val steps with logged steps
        n       = len(steps)
        val_x   = [steps[int(i * n / len(val_bpc))] for i in range(len(val_bpc))]

        ax1.plot(steps, h["train_loss"], label=label, alpha=0.8)
        ax2.plot(val_x, val_bpc,         label=label, linewidth=2)

    ax1.set(title="Train Loss", xlabel="Step", ylabel="Loss")
    ax2.set(title="Val Bits-per-Character", xlabel="Step", ylabel="BPC")
    ax1.legend(); ax2.legend()
    ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "experiment_comparison.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  ✓ Saved: {path}")


# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  Stage IV: Poetry Analysis")
    print("=" * 55)

    # ── Load model ───────────────────────────────────────────
    model, char2idx, idx2char = load_model(CHECKPOINT_PATH, VOCAB_PATH, DEVICE)

    # ── Load corpus for comparison ───────────────────────────
    print("\n→ Loading corpus sample for comparison...")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus_text = f.read()
    # Use a sample to keep comparisons fast
    corpus_sample = corpus_text[:200_000]
    print(f"  Using {len(corpus_sample):,} chars from corpus")

    # ── Generate poems at different temperatures ─────────────
    print("\n→ Generating poems...")
    poems_by_temp = {}
    for temp in TEMPERATURES:
        print(f"  Temperature {temp}...")
        poems = generate_poem_collection(
            model, char2idx, idx2char,
            n_poems=N_POEMS, poem_length=POEM_LENGTH,
            temperature=temp, device=DEVICE,
        )
        poems_by_temp[temp] = poems

    # Save all generated poems to file
    poems_path = os.path.join(OUTPUT_DIR, "generated_poems.txt")
    with open(poems_path, "w", encoding="utf-8") as f:
        for temp, poems in poems_by_temp.items():
            f.write(f"\n{'='*55}\n  TEMPERATURE: {temp}\n{'='*55}\n\n")
            for i, poem in enumerate(poems):
                f.write(f"--- Poem {i+1} ---\n{poem}\n\n")
    print(f"  ✓ Saved: {poems_path}")

    # ── Structural analysis ───────────────────────────────────
    print("\n→ Analysing structure...")
    # Use temperature=0.8 poems as the main comparison
    main_poems = poems_by_temp[0.8]
    struct_stats = plot_structure_comparison(corpus_sample, main_poems, OUTPUT_DIR)

    # ── Rhyme analysis ────────────────────────────────────────
    print("\n→ Analysing rhymes...")
    rhyme_by_temp = plot_rhyme_analysis(poems_by_temp, OUTPUT_DIR)

    # Corpus rhyme rate for reference
    corpus_rhyme = detect_rhymes(corpus_sample)
    print(f"  Corpus rhyme rate:    {corpus_rhyme['rhyme_rate']:.4f}")
    for temp, rate in rhyme_by_temp.items():
        print(f"  Generated (T={temp}): {rate:.4f}")

    # ── Vocabulary overlap ────────────────────────────────────
    print("\n→ Analysing vocabulary...")
    all_generated = "\n\n".join(p for poems in poems_by_temp.values() for p in poems)
    vocab_stats = vocabulary_overlap(all_generated, corpus_sample)
    print(f"  Unique words generated:    {vocab_stats['unique_words_generated']}")
    print(f"  Words found in corpus:     {vocab_stats['overlap_count']}")
    print(f"  Vocabulary overlap:        {vocab_stats['overlap_fraction']:.1%}")

    # ── Training curves comparison ────────────────────────────
    print("\n→ Plotting experiment comparison...")
    exp_dirs = {
        "Baseline (6L, 20k)":    os.path.join(REPO_ROOT, "checkpoints/baseline/history.json"),
        "More layers (9L, 10k)": os.path.join(REPO_ROOT, "checkpoints/layers9/history.json"),
        "More steps (6L, 30k)":  os.path.join(REPO_ROOT, "checkpoints/steps30k/history.json"),
    }
    plot_training_curves_comparison(exp_dirs, OUTPUT_DIR)

    # ── Save summary stats ────────────────────────────────────
    summary = {
        "structural": struct_stats,
        "rhyme":      {"corpus": corpus_rhyme["rhyme_rate"], "by_temperature": rhyme_by_temp},
        "vocabulary": vocab_stats,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    # ── Print report-ready summary ────────────────────────────
    print("\n" + "═" * 55)
    print("  SUMMARY (copy into your report)")
    print("═" * 55)
    print(f"  Corpus mean line length:    {struct_stats['corpus_mean_line_len']:.1f} chars")
    print(f"  Generated mean line length: {struct_stats['generated_mean_line_len']:.1f} chars")
    print(f"  Corpus mean stanza size:    {struct_stats['corpus_mean_stanza_size']:.1f} lines")
    print(f"  Generated mean stanza size: {struct_stats['generated_mean_stanza_size']:.1f} lines")
    print(f"  Corpus rhyme rate:          {corpus_rhyme['rhyme_rate']:.4f}")
    print(f"  Generated rhyme rate (T=0.8): {rhyme_by_temp.get(0.8, 0):.4f}")
    print(f"  Vocabulary overlap:         {vocab_stats['overlap_fraction']:.1%}")
    print("═" * 55)

    # ── Print two sample poems for the report ─────────────────
    print("\n  SAMPLE POEMS FOR REPORT")
    print("─" * 55)
    for i in range(2):
        print(f"\n[Generated poem {i+1}, T=0.8]")
        print(poems_by_temp[0.8][i])


if __name__ == "__main__":
    main()

