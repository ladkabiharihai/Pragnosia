"""
Attention mechanisms for Pragnosia.

Implements Grouped Query Attention (GQA) with optional Flash Attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from pragnosia.modules.embeddings import RotaryEmbedding, apply_rotary_pos_emb


class PragnosiaAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) and RoPE.

    GQA uses fewer key-value heads than query heads, reducing memory
    while maintaining quality.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout

        if (self.head_dim * num_attention_heads) != hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads "
                f"({hidden_size} vs {num_attention_heads})"
            )

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        # Check for Flash Attention availability
        self.use_flash_attention = use_flash_attention and self._check_flash_attention()

    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            from torch.nn.functional import scaled_dot_product_attention
            return hasattr(F, "scaled_dot_product_attention")
        except ImportError:
            return False

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads to match query heads for GQA.

        Args:
            hidden_states: [batch, num_kv_heads, seq_len, head_dim]
            n_rep: Number of times to repeat

        Returns:
            Tensor of shape [batch, num_heads, seq_len, head_dim]
        """
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, 1, seq_len, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_value: Cached key/value states for generation
            use_cache: Whether to return key/value cache
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary embeddings
        # rotary_emb expects [batch, seq, heads, dim], but we have [batch, heads, seq, dim]
        cos, sin = self.rotary_emb(query_states.transpose(1, 2), position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Compute attention
        if self.use_flash_attention and not output_attentions:
            # Use PyTorch's scaled_dot_product_attention (Flash Attention)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )
            attn_weights = None
        else:
            # Manual attention computation
            attn_weights = torch.matmul(
                query_states, key_states.transpose(-2, -1)
            ) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_weights = F.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value
