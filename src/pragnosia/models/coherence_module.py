"""
Coherence Module: Lightweight Transformer for Sequential Binding

This module provides global coherence to locally-learned expert representations.
It's the key innovation that enables Pragnosia to generate coherent text while
maintaining local learning in experts.

Key Design Principles:
1. Lightweight: Only 2-4 layers, small attention heads
2. GPU-resident: But small enough to fit alongside 2 active experts
3. Global learning: Uses standard backprop + cross-entropy
4. Complementary: Works WITH local learning, not against it

Architecture Philosophy:
- Experts (local) learn WHAT: features, patterns, representations
- Coherence (global) learns HOW: sequences, grammar, structure

This is like the brain:
- Cortical columns learn features locally
- Integrative layers bind features into coherent percepts
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Lightweight multi-head attention for coherence."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Projections
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * hidden_size)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # (batch, heads, seq_len, seq_len)

        # Apply causal mask (can only attend to past)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask if provided
        if attention_mask is not None:
            # (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.hidden_size)

        # Final projection
        output = self.out_proj(output)

        return output


class TransformerBlock(nn.Module):
    """Single transformer block for coherence."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        ff_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or hidden_size * 4

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)

        Returns:
            (batch, seq_len, hidden_size)
        """
        # Self-attention with residual
        attn_out = self.attention(x, attention_mask)
        x = self.attn_norm(x + self.attn_dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)

        return x


class CoherenceModule(nn.Module):
    """
    Lightweight transformer module for sequential coherence.

    This module takes expert outputs (learned via local rules) and binds them
    into coherent sequences using self-attention and global objectives.

    Key properties:
    - Lightweight: 2-4 layers only (~50-100MB)
    - GPU-resident: Stays on GPU alongside router
    - Global learning: Uses standard backprop
    - Complementary: Enhances local learning, doesn't replace it

    Design rationale:
    Local learning (experts) discovers diverse features but lacks sequential
    coherence. This module provides the "glue" that binds features into
    grammatical, coherent sequences.

    Think of it as:
    - Experts = cortical columns (local feature detection)
    - Coherence = integrative layers (global binding)
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: Size of expert outputs (must match model hidden_size)
            num_layers: Number of transformer layers (2-4 recommended)
            num_heads: Number of attention heads per layer
            ff_dim: Feed-forward dimension (default: 4 * hidden_size)
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection (from expert outputs to coherence space)
        # This allows the coherence module to operate in its own space
        self.input_proj = nn.Linear(hidden_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        expert_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Bind expert outputs into coherent sequences.

        Args:
            expert_outputs: (batch, seq_len, hidden_size)
                Combined output from experts (learned locally)
            attention_mask: (batch, seq_len)
                1 for real tokens, 0 for padding

        Returns:
            coherent_outputs: (batch, seq_len, hidden_size)
                Contextualized representations with sequential coherence
        """
        # Project expert outputs
        x = self.input_proj(expert_outputs)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output projection and normalization
        x = self.output_proj(x)
        x = self.output_norm(x)

        return x

    def get_memory_size_mb(self) -> float:
        """Calculate memory footprint in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        # 4 bytes per float32 parameter
        return (total_params * 4) / (1024 ** 2)


# Example usage and testing
if __name__ == "__main__":
    # Test coherence module
    batch_size = 2
    seq_len = 128
    hidden_size = 512

    # Create module
    coherence = CoherenceModule(
        hidden_size=hidden_size,
        num_layers=2,
        num_heads=4,
    )

    print(f"Coherence Module: {coherence.get_memory_size_mb():.2f} MB")
    print(f"Parameters: {sum(p.numel() for p in coherence.parameters()):,}")

    # Test forward pass
    expert_outputs = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    coherent_outputs = coherence(expert_outputs, attention_mask)

    print(f"\nInput shape: {expert_outputs.shape}")
    print(f"Output shape: {coherent_outputs.shape}")
    print("✓ Coherence module working correctly")
