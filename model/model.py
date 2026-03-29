"""
model.py
========
Character-level GPT model (decoder-only Transformer).

Architecture overview:
    Input characters
        → Token Embedding + Positional Embedding
        → N x Transformer Blocks (Attention + FeedForward)
        → Layer Norm
        → Linear head → logits over vocabulary

Each Transformer Block contains:
    - Multi-Head Causal Self-Attention  (with residual connection)
    - Feed-Forward Network              (with residual connection)
    - Layer Normalisation before each sub-layer (Pre-LN style)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

class GPTConfig:
    """
    Holds all hyperparameters for the model.
    Change these values to run your hyperparameter experiments (Stage III).
    """
    def __init__(
        self,
        vocab_size: int,        # number of unique characters in your corpus
        context_length: int = 256,  # how many characters the model sees at once
        n_layers: int = 6,      # number of stacked Transformer blocks
        n_heads: int = 8,       # number of attention heads
        d_model: int = 256,     # size of token embeddings (hidden dimension)
        mlp_ratio: int = 4,     # FeedForward hidden dim = mlp_ratio * d_model
        dropout: float = 0.1,   # dropout probability (regularisation)
    ):
        self.vocab_size     = vocab_size
        self.context_length = context_length
        self.n_layers       = n_layers
        self.n_heads        = n_heads
        self.d_model        = d_model
        self.mlp_ratio      = mlp_ratio
        self.dropout        = dropout

      
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

    def __repr__(self):
        params = (
            f"vocab={self.vocab_size}, ctx={self.context_length}, "
            f"layers={self.n_layers}, heads={self.n_heads}, "
            f"d_model={self.d_model}, dropout={self.dropout}"
        )
        return f"GPTConfig({params})"


# ══════════════════════════════════════════════════════════════
# POSITIONAL EMBEDDINGS
# ══════════════════════════════════════════════════════════════

class LearnedPositionalEmbedding(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.context_length, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len)
        # positions: [0, 1, 2, ..., seq_len-1]
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)    # (seq_len,)
        return self.embedding(positions)                        # (seq_len, d_model)


# ══════════════════════════════════════════════════════════════
# MULTI-HEAD CAUSAL SELF-ATTENTION
# ══════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head  = config.d_model // config.n_heads   # dimension per head

        # Single linear layer that projects input into Q, K, V for all heads at once
        # Output size is 3 * d_model because we need Q, K, and V
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Output projection: maps concatenated heads back to d_model
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.dropout = config.dropout

        # Classic causal mask (only used when flash=False)
        # We register it as a buffer so it moves to GPU automatically
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x: torch.Tensor, use_flash: bool = True) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        Returns: same shape (batch_size, seq_len, d_model)
        """
        B, T, C = x.shape   # batch, time (sequence length), channels (d_model)

        # ── Step 1: compute Q, K, V ──────────────────────────────────────
        qkv = self.qkv_proj(x)              # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=2)   # each (B, T, d_model)

        # Reshape for multi-head: (B, n_heads, T, d_head)
        def reshape_for_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape_for_heads(q), reshape_for_heads(k), reshape_for_heads(v)

        # ── Step 2: compute attention ─────────────────────────────────────
        if use_flash:
            # ★ IMPLEMENTATION 1: Flash Attention
            # PyTorch automatically uses the FlashAttention CUDA kernel
            # when running on a GPU. is_causal=True handles the mask internally.
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # ★ IMPLEMENTATION 2: Classic masked attention
            # scale: divide by sqrt(d_head) to keep gradients stable
            scale = 1.0 / math.sqrt(self.d_head)
            scores = (q @ k.transpose(-2, -1)) * scale    # (B, n_heads, T, T)

            # Apply causal mask: set future positions to -inf so softmax → 0
            scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

            # Softmax + dropout
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)

            attn_out = weights @ v                         # (B, n_heads, T, d_head)

        # ── Step 3: merge heads and project ──────────────────────────────
        # Transpose back to (B, T, n_heads, d_head) then flatten heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection + dropout
        return self.resid_dropout(self.out_proj(attn_out))


# ══════════════════════════════════════════════════════════════
# FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = config.mlp_ratio * config.d_model
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(), #activation function
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ffn  = FeedForward(config)

    def forward(self, x: torch.Tensor, use_flash: bool = True) -> torch.Tensor:
        # Attention sub-layer with residual
        x = x + self.attn(self.ln1(x), use_flash=use_flash)
        # FFN sub-layer with residual
        x = x + self.ffn(self.ln2(x))
        return x


# ══════════════════════════════════════════════════════════════
# FULL GPT MODEL
# ══════════════════════════════════════════════════════════════

class CharacterGPT(nn.Module):
    """
    Character-level GPT model.

    Full forward pass:
        token indices (B, T)
            → token embedding (B, T, d_model)
            +  positional embedding (T, d_model)
            → dropout
            → N transformer blocks
            → final layer norm
            → linear projection to vocab_size
            → logits (B, T, vocab_size)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding    = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = LearnedPositionalEmbedding(config)
        self.embed_dropout      = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.final_ln = nn.LayerNorm(config.d_model)

        # Projects d_model → vocab_size to get raw logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and lm_head
        self.lm_head.weight = self.token_embedding.weight

        # Initialise weights (important for stable training)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard GPT-2 weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,          # (B, T) integer token indices
        targets: torch.Tensor = None,  # (B, T) shifted targets for loss
        use_flash: bool = True,
    ):
        """
        Args:
            idx:      (B, T) batch of character index sequences
            targets:  (B, T) same sequences shifted by 1 (for next-char prediction)
            use_flash: whether to use Flash Attention (True) or classic (False)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (None if targets not provided)
        """
        B, T = idx.shape
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context_length {self.config.context_length}"
        )

        # Embeddings
        tok_emb = self.token_embedding(idx)        # (B, T, d_model)
        pos_emb = self.position_embedding(idx)     # (T,  d_model) — broadcast over B
        x = self.embed_dropout(tok_emb + pos_emb)  # (B, T, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, use_flash=use_flash)

        # Final normalisation + project to vocabulary
        x = self.final_ln(x)                       # (B, T, d_model)
        logits = self.lm_head(x)                   # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab_size) and (B*T,) for cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,      # (1, T) starting context
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
       
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to the last context_length tokens
            idx_cond = idx[:, -self.config.context_length:]

            # Get predictions
            logits, _ = self(idx_cond)

            # Take logits at the last position (the next character to predict)
            logits = logits[:, -1, :]              # (1, vocab_size)

            # Apply temperature scaling
            logits = logits / temperature

            # Optionally restrict to top-k candidates
            if top_k is not None:
                # Zero out everything except the top-k logits
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)      # (1, vocab_size)
            next_idx = torch.multinomial(probs, num_samples=1)   # (1, 1)

            # Append to sequence
            idx = torch.cat([idx, next_idx], dim=1)   # (1, T+1)

        return idx

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        n_params = self.count_parameters()
        return (
            f"CharacterGPT | {self.config}\n"
            f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)"
        )

