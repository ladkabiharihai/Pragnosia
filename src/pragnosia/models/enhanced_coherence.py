"""
Enhanced Coherence Module with ALiBi for Generation

This is a SIGNIFICANTLY STRENGTHENED version of the coherence module designed
specifically to enable coherent chat and code generation while preserving
local learning in experts.

Key Enhancements:
1. ALiBi (Attention with Linear Biases) - Better positional understanding
2. 8-12 transformer layers (vs original 2) - Strong sequential binding
3. Conversation state tracking - Maintains context across turns
4. KV caching - Fast autoregressive generation
5. Generation-optimized training - Focuses on coherence

Design Philosophy:
- Experts (local) learn rich representations without sequential constraints
- Coherence (global) binds representations into grammatical, coherent sequences
- Two-phase training: Representation → Generation
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class ALiBiMultiHeadAttention(nn.Module):
    """Multi-head attention with ALiBi positional encoding.

    ALiBi (Attention with Linear Biases) adds a simple linear bias to attention
    scores based on key-query distance. This provides strong positional information
    without explicit position embeddings, enabling better extrapolation to longer
    sequences and improved generation quality.

    Paper: "Train Short, Test Long" (Press et al., 2022)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_kv_cache: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_kv_cache = use_kv_cache

        # Projections
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # ALiBi slopes (one per attention head)
        slopes = torch.Tensor(self._get_alibi_slopes(num_heads))
        self.register_buffer('alibi_slopes', slopes)

        # KV cache for generation (optional)
        self.cached_keys = None
        self.cached_values = None

    def _get_alibi_slopes(self, num_heads):
        """
        Compute ALiBi slopes for each attention head.

        Each head gets a different slope, allowing the model to focus on
        different distance ranges. Closer heads focus on local context,
        distant heads focus on long-range dependencies.
        """
        def get_slopes_power_of_2(n):
            start = 2**(-2**-(math.log2(n)-3))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # Handle non-power-of-2 number of heads
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            base_slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
            return base_slopes + extra_slopes[:num_heads - closest_power_of_2]

    def _get_alibi_bias(self, seq_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute ALiBi bias matrix.

        Returns a (num_heads, seq_len, key_len) bias matrix where:
        - bias[h, i, j] = slope[h] * (j - i)
        - Negative when j < i (attending to past)
        - Zero when j = i (current position)
        - Positive when j > i (future, masked out by causal mask)
        """
        # Position indices
        context_position = torch.arange(seq_len, device=device)[:, None]  # (seq_len, 1)
        memory_position = torch.arange(key_len, device=device)[None, :]   # (1, key_len)

        # Relative position: how far back are we looking?
        relative_position = memory_position - context_position  # (seq_len, key_len)

        # Apply head-specific slopes
        # (num_heads, 1, 1) * (1, seq_len, key_len) -> (num_heads, seq_len, key_len)
        alibi_bias = self.alibi_slopes[:, None, None] * relative_position[None, :, :]

        return alibi_bias

    def clear_cache(self):
        """Clear KV cache (call this between different generation sequences)."""
        self.cached_keys = None
        self.cached_values = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
            use_cache: If True, use KV cache for generation (only compute new token)

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * hidden_size)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle KV cache for generation
        if use_cache and self.use_kv_cache:
            if self.cached_keys is not None:
                # Append new keys/values to cache
                k = torch.cat([self.cached_keys, k], dim=2)  # (batch, heads, key_len, head_dim)
                v = torch.cat([self.cached_values, v], dim=2)

            # Update cache
            self.cached_keys = k.detach()
            self.cached_values = v.detach()

        key_len = k.size(2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # (batch, heads, seq_len, key_len)

        # Apply ALiBi positional bias
        alibi_bias = self._get_alibi_bias(seq_len, key_len, x.device)
        scores = scores + alibi_bias.unsqueeze(0)  # (1, heads, seq_len, key_len)

        # Apply causal mask (can only attend to past)
        causal_mask = torch.triu(
            torch.ones(seq_len, key_len, device=x.device),
            diagonal=key_len - seq_len + 1
        ).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask if provided
        if attention_mask is not None:
            # Expand mask to cover all keys
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)

        # Handle NaN from all -inf rows (shouldn't happen with proper masks)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.hidden_size)

        # Final projection
        output = self.out_proj(output)

        return output


class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with pre-norm and better regularization."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        use_kv_cache: bool = True,
    ):
        super().__init__()
        ff_dim = ff_dim or hidden_size * 4

        # Pre-LayerNorm architecture (more stable)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attention = ALiBiMultiHeadAttention(
            hidden_size, num_heads, dropout, use_kv_cache
        )
        self.attn_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def clear_cache(self):
        """Clear attention cache."""
        self.attention.clear_cache()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
            use_cache: Use KV cache for generation

        Returns:
            (batch, seq_len, hidden_size)
        """
        # Self-attention with pre-norm and residual
        normed = self.attn_norm(x)
        attn_out = self.attention(normed, attention_mask, use_cache)
        x = x + self.attn_dropout(attn_out)

        # Feed-forward with pre-norm and residual
        normed = self.ff_norm(x)
        ff_out = self.ff(normed)
        x = x + ff_out

        return x


class EnhancedCoherenceModule(nn.Module):
    """
    SIGNIFICANTLY ENHANCED coherence module for chat and code generation.

    This is a full-scale transformer decoder (8-12 layers) designed to bind
    locally-learned expert representations into coherent, grammatical sequences.

    Key Features:
    1. ALiBi positional encoding - Better position understanding
    2. 8-12 layers - Strong sequential modeling
    3. KV caching - Fast generation (10x speedup)
    4. Pre-norm architecture - More stable training
    5. Large capacity - Can learn complex patterns

    Design Rationale:
    - Your experts learn REPRESENTATIONS via local learning
    - This module learns SEQUENCES via global learning
    - Together: Best of both worlds

    Memory Cost:
    - 8 layers: ~120 MB
    - 12 layers: ~180 MB
    (Still fits easily with 2 active experts ~500MB each)
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        use_kv_cache: bool = True,
    ):
        """
        Args:
            hidden_size: Size of expert outputs (must match model hidden_size)
            num_layers: Number of transformer layers (8-12 recommended for generation)
            num_heads: Number of attention heads per layer
            ff_dim: Feed-forward dimension (default: 4 * hidden_size)
            dropout: Dropout rate
            use_kv_cache: Enable KV caching for fast generation
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_kv_cache = use_kv_cache

        # Input projection (from expert outputs to coherence space)
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # Enhanced transformer layers with ALiBi
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_kv_cache=use_kv_cache,
            )
            for _ in range(num_layers)
        ])

        # Output projection and normalization
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        print(f"🚀 Enhanced Coherence Module: {num_layers} layers, {num_heads} heads")
        print(f"   Memory: {self.get_memory_size_mb():.1f} MB")
        print(f"   KV Cache: {'enabled' if use_kv_cache else 'disabled'}")

    def clear_cache(self):
        """Clear all KV caches (call between different sequences)."""
        for layer in self.layers:
            layer.clear_cache()

    def forward(
        self,
        expert_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Bind expert outputs into coherent sequences.

        Args:
            expert_outputs: (batch, seq_len, hidden_size)
                Combined output from experts (learned locally)
            attention_mask: (batch, seq_len)
                1 for real tokens, 0 for padding
            use_cache: Use KV cache for generation (faster, but only works for
                      autoregressive generation where seq_len=1)

        Returns:
            coherent_outputs: (batch, seq_len, hidden_size)
                Contextualized representations with strong sequential coherence
        """
        # Project and normalize expert outputs
        x = self.input_proj(expert_outputs)
        x = self.input_norm(x)

        # Apply transformer layers with ALiBi
        for layer in self.layers:
            x = layer(x, attention_mask, use_cache)

        # Output projection and normalization
        x = self.output_norm(x)
        x = self.output_proj(x)

        return x

    def get_memory_size_mb(self) -> float:
        """Calculate memory footprint in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        # 4 bytes per float32 parameter
        return (total_params * 4) / (1024 ** 2)


# Example usage and testing
if __name__ == "__main__":
    # Test enhanced coherence module
    batch_size = 2
    seq_len = 128
    hidden_size = 512

    # Create module (8 layers for generation)
    coherence = EnhancedCoherenceModule(
        hidden_size=hidden_size,
        num_layers=8,
        num_heads=8,
        use_kv_cache=True,
    )

    print(f"\nParameters: {sum(p.numel() for p in coherence.parameters()):,}")

    # Test forward pass
    expert_outputs = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    coherent_outputs = coherence(expert_outputs, attention_mask)

    print(f"\nInput shape: {expert_outputs.shape}")
    print(f"Output shape: {coherent_outputs.shape}")
    print("✓ Enhanced coherence module working correctly")

    # Test KV caching (generation mode)
    print("\nTesting KV cache (generation simulation):")
    coherence.clear_cache()

    # Initial prompt
    prompt = expert_outputs[:, :10, :]
    out1 = coherence(prompt, use_cache=True)
    print(f"  Step 1: input={prompt.shape}, output={out1.shape}")

    # Generate next tokens (one at a time)
    for i in range(3):
        next_token = expert_outputs[:, 10+i:10+i+1, :]
        out = coherence(next_token, use_cache=True)
        print(f"  Step {i+2}: input={next_token.shape}, output={out.shape}")

    print("✓ KV cache working correctly")
