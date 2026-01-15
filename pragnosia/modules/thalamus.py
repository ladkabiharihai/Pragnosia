"""
Thalamus Router - Core MoE routing mechanism for Pragnosia.

The Thalamus acts as the brain's relay station, routing information
to appropriate cortical regions. Similarly, this router directs tokens
to specialized experts based on content and energy budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class RouterOutput:
    """Output from the Thalamus Router."""
    dispatch_mask: torch.Tensor  # [batch, seq_len, num_experts]
    combine_weights: torch.Tensor  # [batch, seq_len, num_experts]
    router_logits: torch.Tensor  # [batch, seq_len, num_experts]
    aux_loss: torch.Tensor  # Load balancing loss
    z_loss: torch.Tensor  # Router z-loss for stability
    expert_indices: torch.Tensor  # [batch, seq_len, top_k]
    routing_entropy: torch.Tensor  # For plasticity monitoring


class ThalamusRouter(nn.Module):
    """
    Token-to-Expert Router with energy-aware gating.

    Features:
    - Top-K sparse routing (only K experts active per token)
    - Energy budget constraints
    - Load balancing via auxiliary loss
    - Router z-loss for training stability
    - Routing entropy tracking for plasticity decisions
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        enable_energy_gating: bool = True,
        min_energy_threshold: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity_factor = expert_capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.enable_energy_gating = enable_energy_gating
        self.min_energy_threshold = min_energy_threshold

        # Router projection: hidden_size -> num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Energy gating network (predicts compute requirement)
        if enable_energy_gating:
            self.energy_gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid(),
            )

        # Expert activation tracking for plasticity
        self.register_buffer(
            "expert_activation_counts",
            torch.zeros(num_experts),
            persistent=True
        )
        self.register_buffer("total_tokens_routed", torch.tensor(0.0), persistent=True)

    def _compute_routing_weights(
        self,
        hidden_states: torch.Tensor,
        energy_budget: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights with optional energy constraints.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            energy_budget: 0.0 to 1.0, controls sparsity

        Returns:
            Tuple of (router_logits, routing_weights, expert_mask)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch, seq_len, num_experts]

        # Apply energy budget to adjust number of active experts
        effective_top_k = self.num_experts_per_token
        if self.enable_energy_gating and energy_budget < 1.0:
            # Scale down active experts based on energy budget
            effective_top_k = max(
                1,
                int(self.num_experts_per_token * energy_budget)
            )

        # Top-K selection
        routing_weights, expert_indices = torch.topk(
            router_logits, effective_top_k, dim=-1
        )  # [batch, seq_len, top_k]

        # Normalize weights (softmax over selected experts)
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Create dispatch mask
        expert_mask = torch.zeros_like(router_logits)
        expert_mask.scatter_(-1, expert_indices, 1.0)

        return router_logits, routing_weights, expert_mask, expert_indices

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.

        Encourages even distribution of tokens across experts.
        """
        # Average routing probability per expert
        router_probs = F.softmax(router_logits, dim=-1)
        avg_probs = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Fraction of tokens routed to each expert
        tokens_per_expert = expert_mask.sum(dim=[0, 1])  # [num_experts]
        total_tokens = expert_mask.sum()
        expert_fraction = tokens_per_expert / (total_tokens + 1e-6)

        # Auxiliary loss: minimize product of avg_probs and expert_fraction
        # This encourages uniform distribution
        aux_loss = (avg_probs * expert_fraction).sum() * self.num_experts

        return aux_loss * self.aux_loss_coef

    def _compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute router z-loss for training stability.

        Penalizes large logits to prevent router collapse.
        """
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        return z_loss * self.z_loss_coef

    def _compute_routing_entropy(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute routing entropy for plasticity monitoring.

        High entropy indicates uncertain routing -> may need more experts.
        Low entropy indicates confident routing -> may have redundant experts.
        """
        probs = F.softmax(router_logits, dim=-1)
        log_probs = F.log_softmax(router_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

    def _update_activation_stats(
        self,
        expert_indices: torch.Tensor,
        training: bool = True,
    ):
        """Update expert activation statistics for plasticity."""
        if training:
            # Count activations per expert
            for expert_id in range(self.num_experts):
                count = (expert_indices == expert_id).sum().float()
                self.expert_activation_counts[expert_id] += count

            self.total_tokens_routed += expert_indices.numel()

    def get_expert_utilization(self) -> torch.Tensor:
        """Get utilization ratio for each expert."""
        if self.total_tokens_routed == 0:
            return torch.ones(self.num_experts) / self.num_experts
        return self.expert_activation_counts / self.total_tokens_routed

    def reset_activation_stats(self):
        """Reset activation statistics."""
        self.expert_activation_counts.zero_()
        self.total_tokens_routed.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        energy_budget: float = 1.0,
    ) -> RouterOutput:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            energy_budget: Energy constraint (0.0 to 1.0)

        Returns:
            RouterOutput with routing information
        """
        # Compute routing
        router_logits, routing_weights, expert_mask, expert_indices = \
            self._compute_routing_weights(hidden_states, energy_budget)

        # Compute losses
        aux_loss = self._compute_aux_loss(router_logits, expert_mask)
        z_loss = self._compute_z_loss(router_logits)

        # Compute entropy for plasticity
        routing_entropy = self._compute_routing_entropy(router_logits)

        # Update activation statistics
        self._update_activation_stats(expert_indices, self.training)

        # Create combine weights tensor
        combine_weights = torch.zeros_like(router_logits)
        combine_weights.scatter_(-1, expert_indices, routing_weights)

        return RouterOutput(
            dispatch_mask=expert_mask,
            combine_weights=combine_weights,
            router_logits=router_logits,
            aux_loss=aux_loss,
            z_loss=z_loss,
            expert_indices=expert_indices,
            routing_entropy=routing_entropy,
        )
