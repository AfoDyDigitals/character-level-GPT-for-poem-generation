import requests
import re
import random
import os

POETS = [
    ("Walt Whitman",          "https://www.gutenberg.org/cache/epub/1322/pg1322.txt"),   # Leaves of Grass
    ("Emily Dickinson",       "https://www.gutenberg.org/cache/epub/12242/pg12242.txt"), # Complete Poems
    ("John Keats",            "https://www.gutenberg.org/cache/epub/23684/pg23684.txt"), # Poetical Works
    ("Percy Bysshe Shelley",  "https://www.gutenberg.org/cache/epub/4800/pg4800.txt"),   # Complete Poetical Works
    ("William Blake",         "https://www.gutenberg.org/cache/epub/574/pg574.txt"),     # Poems
    ("Alfred Lord Tennyson",  "https://www.gutenberg.org/cache/epub/8601/pg8601.txt"),   # Poetical Works
    ("Edgar Allan Poe",       "https://www.gutenberg.org/cache/epub/2147/pg2147.txt"),   # Complete Poetical Works
    ("Henry W. Longfellow",   "https://www.gutenberg.org/cache/epub/1365/pg1365.txt"),   # Poetical Works
    ("Christina Rossetti",    "https://www.gutenberg.org/cache/epub/473/pg473.txt"),     # Goblin Market & Other Poems
    ("William Wordsworth",    "https://www.gutenberg.org/cache/epub/9622/pg9622.txt"),   # Lyrical Ballads
    ("Lord Byron",            "https://www.gutenberg.org/cache/epub/9810/pg9810.txt"),   # Poetical Works
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def download_text(url: str) -> str | None:
    """Download raw text from a URL. Returns None on failure."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return None


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove the Gutenberg header and footer.
    The actual content lives between the START and END markers.
    """
    # Try to find the standard start marker
    start_patterns = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*",
    ]
    end_patterns = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*",
    ]

    start_idx = 0
    for pat in start_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            start_idx = m.end()
            break

    end_idx = len(text)
    for pat in end_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            end_idx = m.start()
            break

    return text[start_idx:end_idx]


def clean_text(text: str) -> str:
    """
    Clean the raw poem text while preserving poetic structure.
    - Keep line breaks (essential for verse)
    - Keep stanza boundaries (double newlines)
    - Remove Windows-style carriage returns
    - Remove lines that look like metadata (all-caps titles, page numbers)
    - Strip excessive blank lines (keep max 2 in a row)
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Remove lines that are just page numbers (e.g. "  42  " or "[42]")
        if re.fullmatch(r"\s*\[?\d+\]?\s*", line):
            continue
        # Remove lines that look like chapter headings in ALL CAPS
        # (but keep short all-caps words that might be poem titles — allow up to 6 words)
        stripped = line.strip()
        if stripped and stripped == stripped.upper() and len(stripped.split()) > 6:
            continue
        cleaned.append(line.rstrip())   # remove trailing spaces but keep the line

    text = "\n".join(cleaned)

    # Collapse runs of more than 2 blank lines into exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def split_corpus(text: str, val_fraction: float = 0.1, seed: int = 42):
    """
    Split the corpus into train and validation sets at the stanza level.
    Splitting at stanza boundaries preserves poetic structure rather than
    cutting in the middle of a poem.
    """
    # Split into "chunks" separated by blank lines (stanzas / poem blocks)
    chunks = re.split(r"\n\n+", text)
    chunks = [c.strip() for c in chunks if c.strip()]

    random.seed(seed)
    random.shuffle(chunks)

    split_point = int(len(chunks) * (1 - val_fraction))
    train_chunks = chunks[:split_point]
    val_chunks   = chunks[split_point:]

    train_text = "\n\n".join(train_chunks)
    val_text   = "\n\n".join(val_chunks)

    return train_text, val_text


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    all_texts = []
    total_chars = 0

    print("=" * 55)
    print("  Gutenberg Poetry Corpus Builder")
    print("=" * 55)

    for name, url in POETS:
        print(f"\n→ Downloading: {name}")
        raw = download_text(url)
        if raw is None:
            continue

        content = strip_gutenberg_boilerplate(raw)
        content = clean_text(content)

        char_count = len(content)
        total_chars += char_count
        all_texts.append(content)

        print(f"  ✓ {char_count:,} characters kept")

    print("\n" + "=" * 55)
    print(f"  Total corpus size: {total_chars:,} characters "
          f"({total_chars / 1_000_000:.2f} MB)")
    print("=" * 55)

    # Combine everything
    full_corpus = "\n\n".join(all_texts)

    # Save the full corpus
    with open("poetry_corpus.txt", "w", encoding="utf-8") as f:
        f.write(full_corpus)
    print("\n✓ Saved: poetry_corpus.txt")

    # Train / val split
    train_text, val_text = split_corpus(full_corpus)

    with open("train.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open("val.txt", "w", encoding="utf-8") as f:
        f.write(val_text)

    print(f"✓ Saved: train.txt  ({len(train_text):,} chars)")
    print(f"✓ Saved: val.txt    ({len(val_text):,} chars)")

    # Quick sanity check: show a small excerpt
    print("\n--- Sample excerpt from corpus ---")
    excerpt_lines = full_corpus.split("\n")[:20]
    print("\n".join(excerpt_lines))
    print("...\n")

    # Vocabulary check
    vocab = sorted(set(full_corpus))
    print(f"Vocabulary size: {len(vocab)} unique characters")
    printable_vocab = "".join(c if c.isprintable() else "?" for c in vocab)
    print(f"Characters: {printable_vocab}")


if __name__ == "__main__":
    main()