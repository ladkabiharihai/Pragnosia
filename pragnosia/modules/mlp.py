"""
Feed-forward network (MLP) for Pragnosia.

Implements SwiGLU activation as used in LLaMA and other modern LLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PragnosiaMLP(nn.Module):
    """
    Feed-forward network with SwiGLU activation.

    SwiGLU: x * SiLU(gate(x)) provides better gradient flow
    and is used in LLaMA, Mistral, etc.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate and up projections for SwiGLU
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.hidden_dropout = hidden_dropout

        # Activation function
        self.act_fn = self._get_activation(hidden_act)

    def _get_activation(self, act: str):
        """Get activation function by name."""
        activations = {
            "silu": F.silu,
            "gelu": F.gelu,
            "relu": F.relu,
            "tanh": torch.tanh,
        }
        if act not in activations:
            raise ValueError(f"Unknown activation: {act}. Choose from {list(activations.keys())}")
        return activations[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # SwiGLU: down(act(gate(x)) * up(x))
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        if self.hidden_dropout > 0 and self.training:
            hidden = F.dropout(hidden, p=self.hidden_dropout, training=True)

        output = self.down_proj(hidden)
        return output
