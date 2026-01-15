"""
Transformer blocks and model for Pragnosia.

Combines attention, MoE, and normalization into complete transformer layers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from pragnosia.modules.attention import PragnosiaAttention
from pragnosia.modules.experts import CognitiveCortex
from pragnosia.modules.normalization import RMSNorm
from pragnosia.modules.mlp import PragnosiaMLP


class PragnosiaBlock(nn.Module):
    """
    Single Pragnosia Transformer block.

    Architecture:
    - Pre-norm with RMSNorm
    - Multi-head attention (GQA + RoPE)
    - Residual connection
    - Pre-norm with RMSNorm
    - MoE feed-forward (Cognitive Cortex)
    - Residual connection
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_types: Optional[List[str]] = None,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        hidden_act: str = "silu",
        use_moe: bool = True,  # If False, use standard MLP
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.use_moe = use_moe

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Attention
        self.self_attn = PragnosiaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            use_flash_attention=use_flash_attention,
        )

        # Post-attention normalization
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Feed-forward: MoE or standard MLP
        if use_moe:
            self.feed_forward = CognitiveCortex(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                expert_types=expert_types,
                hidden_act=hidden_act,
                dropout=hidden_dropout,
            )
        else:
            self.feed_forward = PragnosiaMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout=hidden_dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        energy_budget: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple], Optional[Dict]]:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: KV cache for generation
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            energy_budget: Energy constraint for MoE routing

        Returns:
            Tuple of (hidden_states, attention_weights, past_key_value, moe_aux_info)
        """
        residual = hidden_states

        # Pre-norm + Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # Pre-norm + Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        moe_aux_info = None
        if self.use_moe:
            hidden_states, moe_aux_info = self.feed_forward(hidden_states, energy_budget)
        else:
            hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, past_key_value, moe_aux_info


class PragnosiaTransformer(nn.Module):
    """
    Complete Pragnosia Transformer stack.

    Contains:
    - Token embeddings
    - Multiple PragnosiaBlocks (with MoE)
    - Final layer normalization
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_types: Optional[List[str]] = None,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        hidden_act: str = "silu",
        pad_token_id: int = 0,
        use_gradient_checkpointing: bool = False,
        moe_layer_frequency: int = 1,  # Apply MoE every N layers (1 = all layers)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        # Transformer layers
        self.layers = nn.ModuleList()
        for layer_idx in range(num_hidden_layers):
            # Alternate between MoE and standard layers based on frequency
            use_moe = (layer_idx % moe_layer_frequency == 0)

            self.layers.append(
                PragnosiaBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    num_experts=num_experts if use_moe else 1,
                    num_experts_per_token=num_experts_per_token if use_moe else 1,
                    expert_types=expert_types if use_moe else None,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                    attention_dropout=attention_dropout,
                    hidden_dropout=hidden_dropout,
                    rms_norm_eps=rms_norm_eps,
                    use_flash_attention=use_flash_attention,
                    hidden_act=hidden_act,
                    use_moe=use_moe,
                    layer_idx=layer_idx,
                )
            )

        # Final normalization
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """Prepare causal attention mask."""
        batch_size, seq_len = input_shape

        # Create causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        # Expand for past key values
        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros(seq_len, past_key_values_length, device=device, dtype=torch.bool),
                mask
            ], dim=-1)

        # Convert to attention mask format (0 for attend, -inf for mask)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq+past]
        mask = mask.expand(batch_size, 1, -1, -1)
        mask = mask.to(dtype) * torch.finfo(dtype).min

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq] with 1 for valid, 0 for pad
            expanded_mask = attention_mask[:, None, None, :]
            expanded_mask = (1.0 - expanded_mask.to(dtype)) * torch.finfo(dtype).min
            mask = mask + expanded_mask

        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        energy_budget: float = 1.0,
    ) -> Dict[str, Union[torch.Tensor, List, Dict]]:
        """
        Forward pass through transformer.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_values: KV cache for generation
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            energy_budget: Energy constraint for MoE routing

        Returns:
            Dict with last_hidden_state, past_key_values, hidden_states,
            attentions, and moe_aux_losses
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Determine past sequence length
        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Prepare attention mask
        causal_mask = self._prepare_attention_mask(
            attention_mask,
            (batch_size, seq_len),
            hidden_states.dtype,
            device,
            past_key_values_length,
        )

        # Initialize outputs
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_moe_aux_losses = []
        new_past_key_values = [] if use_cache else None

        # Process through layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            past_kv = past_key_values[idx] if past_key_values is not None else None

            if self.use_gradient_checkpointing and self.training:
                hidden_states, attn_weights, past_kv, moe_aux = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_kv,
                    use_cache,
                    output_attentions,
                    energy_budget,
                    use_reentrant=False,
                )
            else:
                hidden_states, attn_weights, past_kv, moe_aux = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    energy_budget=energy_budget,
                )

            if use_cache:
                new_past_key_values.append(past_kv)

            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

            if moe_aux is not None:
                all_moe_aux_losses.append(moe_aux)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Aggregate MoE losses
        total_aux_loss = torch.tensor(0.0, device=device)
        total_z_loss = torch.tensor(0.0, device=device)
        if all_moe_aux_losses:
            for aux in all_moe_aux_losses:
                total_aux_loss = total_aux_loss + aux["aux_loss"]
                total_z_loss = total_z_loss + aux["z_loss"]

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": new_past_key_values,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "moe_aux_loss": total_aux_loss,
            "moe_z_loss": total_z_loss,
            "moe_aux_info": all_moe_aux_losses,
        }

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value
