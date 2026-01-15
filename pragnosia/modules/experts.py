"""
Expert modules for Pragnosia's Cognitive Cortex.

Each expert specializes in a specific cognitive function:
- Language: Syntax and fluency
- Reasoning: Math and logic
- Memory: Long-context recall
- Planning: Step-by-step reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from enum import Enum

from pragnosia.modules.mlp import PragnosiaMLP
from pragnosia.modules.thalamus import ThalamusRouter, RouterOutput


class ExpertType(Enum):
    """Types of cognitive experts."""
    LANGUAGE = "language"
    REASONING = "reasoning"
    MEMORY = "memory"
    PLANNING = "planning"
    GENERAL = "general"


class Expert(nn.Module):
    """
    Single expert module (specialized MLP).

    Each expert is a feed-forward network that can be specialized
    for different cognitive tasks through training.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        expert_type: str = "general",
        hidden_act: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_type = expert_type

        # Core MLP
        self.mlp = PragnosiaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout=dropout,
        )

        # Expert-specific statistics
        self.register_buffer("activation_count", torch.tensor(0.0))
        self.register_buffer("total_norm", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        output = self.mlp(x)

        # Track activation statistics
        if self.training:
            self.activation_count += x.shape[0] * x.shape[1]
            self.total_norm += output.norm().detach()

        return output

    def get_average_activation_norm(self) -> float:
        """Get average output norm (for pruning decisions)."""
        if self.activation_count == 0:
            return 0.0
        return (self.total_norm / self.activation_count).item()

    def reset_stats(self):
        """Reset activation statistics."""
        self.activation_count.zero_()
        self.total_norm.zero_()


class CognitiveCortex(nn.Module):
    """
    Mixture-of-Experts layer implementing the Cognitive Cortex.

    Routes tokens to specialized experts based on Thalamus Router decisions.
    Supports dynamic expert growth and pruning (plasticity).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_types: Optional[List[str]] = None,
        expert_capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        enable_energy_gating: bool = True,
        hidden_act: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        # Initialize expert types
        if expert_types is None:
            expert_types = ["general"] * num_experts
        elif len(expert_types) < num_experts:
            # Pad with general experts
            expert_types = expert_types + ["general"] * (num_experts - len(expert_types))

        # Create experts
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                expert_type=expert_types[i],
                hidden_act=hidden_act,
                dropout=dropout,
            )
            for i in range(num_experts)
        ])

        # Thalamus Router
        self.router = ThalamusRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            expert_capacity_factor=expert_capacity_factor,
            aux_loss_coef=aux_loss_coef,
            z_loss_coef=z_loss_coef,
            enable_energy_gating=enable_energy_gating,
        )

        # Shared expert (always active, provides baseline capability)
        self.shared_expert = Expert(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size // 2,  # Smaller shared expert
            expert_type="shared",
            hidden_act=hidden_act,
            dropout=dropout,
        )
        self.use_shared_expert = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        energy_budget: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            energy_budget: Energy constraint (0.0 to 1.0)

        Returns:
            Tuple of (output tensor, aux_info dict)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get routing decisions from Thalamus
        router_output = self.router(hidden_states, energy_budget)

        # Process through experts
        # Flatten for easier processing
        flat_hidden = hidden_states.view(-1, hidden_dim)  # [batch*seq, hidden]

        # Initialize output
        expert_output = torch.zeros_like(flat_hidden)

        # Get expert assignments
        expert_indices = router_output.expert_indices.view(-1, self.num_experts_per_token)
        combine_weights = router_output.combine_weights.view(-1, self.num_experts)

        # Process each expert
        for expert_id, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_id).any(dim=-1)

            if expert_mask.sum() == 0:
                continue

            # Get tokens for this expert
            expert_input = flat_hidden[expert_mask]

            # Process through expert
            expert_out = expert(expert_input)

            # Get weights for this expert
            weights = combine_weights[expert_mask, expert_id].unsqueeze(-1)

            # Add weighted output
            expert_output[expert_mask] += expert_out * weights

        # Add shared expert output (always computed)
        if self.use_shared_expert:
            shared_output = self.shared_expert(flat_hidden)
            # Shared expert contributes with reduced weight
            expert_output = expert_output + 0.5 * shared_output

        # Reshape output
        output = expert_output.view(batch_size, seq_len, hidden_dim)

        # Collect auxiliary information
        aux_info = {
            "aux_loss": router_output.aux_loss,
            "z_loss": router_output.z_loss,
            "router_logits": router_output.router_logits,
            "routing_entropy": router_output.routing_entropy,
            "expert_indices": router_output.expert_indices,
        }

        return output, aux_info

    def get_expert_utilization(self) -> torch.Tensor:
        """Get utilization ratio for each expert."""
        return self.router.get_expert_utilization()

    def reset_stats(self):
        """Reset all statistics."""
        self.router.reset_activation_stats()
        for expert in self.experts:
            expert.reset_stats()
        self.shared_expert.reset_stats()


class SparseMoEBlock(nn.Module):
    """
    Complete sparse MoE block that can replace standard FFN.

    Combines attention output with MoE processing.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_types: Optional[List[str]] = None,
        hidden_act: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.cognitive_cortex = CognitiveCortex(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            expert_types=expert_types,
            hidden_act=hidden_act,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        energy_budget: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through sparse MoE block."""
        return self.cognitive_cortex(hidden_states, energy_budget)
