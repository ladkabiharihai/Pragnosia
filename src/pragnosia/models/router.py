"""Hebbian router implementation for expert gating."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class HebbianRouter(nn.Module):
    """
    Hebbian expert router with provable stability properties.

    Uses local, reward-free updates based on correlation between input features
    and expert error reduction. Implements winner-take-most with lateral inhibition.

    Routing update: Δr_e = η_r · corr(φ(x), ΔL_e)
    Activation: a_e = ReLU(r_e - θ - Σ_{e'≠e} w_{ee'} · a_{e'})
    """

    def __init__(
        self,
        input_size: int,
        num_experts: int,
        num_active_experts: int = 2,
        learning_rate: float = 0.01,
        lateral_inhibition: float = 0.1,
        threshold: float = 0.1,
        router_size: int = 256,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.learning_rate = learning_rate
        self.lateral_inhibition = lateral_inhibition
        self.threshold = threshold

        # Feature projection for input (φ(x))
        self.feature_projection = nn.Sequential(
            nn.Linear(input_size, router_size),
            nn.LayerNorm(router_size),
            nn.GELU(),
            nn.Linear(router_size, router_size),
        )

        # Routing scores (r_e for each expert) - learnable parameters, not buffers
        # Using nn.Parameter instead of register_buffer allows gradient-based updates
        # while keeping memory O(n) instead of O(n²)
        initial_scores = torch.randn(num_experts) * 0.1 + 1.0
        initial_scores = initial_scores / initial_scores.sum()  # Normalize
        self.routing_scores = nn.Parameter(initial_scores, requires_grad=False)

        # For constant-VRAM: Use scalar inhibition instead of O(n²) matrix
        # Lateral inhibition is computed on-the-fly during forward pass
        # This keeps memory O(1) instead of O(n²)

        # Statistics for stability analysis - use fixed-size rolling buffers
        # These don't grow with num_experts
        self.register_buffer("routing_entropy_history", torch.zeros(1000))
        self.register_buffer("expert_activation_counts", torch.zeros(num_experts))
        self.entropy_idx = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_routing_weights: bool = False,
        hard_routing: bool = False,
    ) -> Tuple[torch.Tensor, List[int], Optional[torch.Tensor]]:
        """
        Route inputs to top-k experts using Hebbian gating.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            return_routing_weights: Whether to return routing weights
            hard_routing: If True, uses hard (discrete) routing with no soft weights.
                          This enforces TRUE sparsity - exactly k experts activated.
                          Critical for constant-VRAM training.

        Returns:
            features: Projected features for expert selection
            selected_experts: List of expert indices to activate
            routing_weights: Optional routing weights for each expert (None if hard_routing)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Project input to feature space
        # Pool over sequence dimension for routing decision
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)
        features = self.feature_projection(pooled)  # (batch, router_size)

        # Compute activations with lateral inhibition
        # a_e = ReLU(r_e - θ - Σ_{e'≠e} w_{ee'} · a_{e'})
        # Simplified: use routing scores directly with competition
        routing_logits = self.routing_scores.unsqueeze(0)  # (1, num_experts)

        # Add input-dependent modulation
        expert_affinity = torch.matmul(
            features,
            self.feature_projection[-1].weight.T[:, :self.num_experts]
        )  # (batch, num_experts)

        # Combine base routing scores with input affinity
        routing_logits = routing_logits + 0.1 * expert_affinity

        # Apply lateral inhibition through iterative competition
        activations = self._competitive_activation(routing_logits)

        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            activations,
            self.num_active_experts,
            dim=-1
        )

        # Normalize routing weights (only if soft routing)
        if not hard_routing:
            routing_weights = F.softmax(routing_weights, dim=-1)
        else:
            # Hard routing: equal weights (1/k) for selected experts
            routing_weights = torch.ones_like(routing_weights) / self.num_active_experts

        # Update statistics
        with torch.no_grad():
            for expert_idx in selected_experts[0]:
                self.expert_activation_counts[expert_idx] += 1

            # Compute and store routing entropy
            entropy = self._compute_entropy(activations[0])
            self.routing_entropy_history[self.entropy_idx % 1000] = entropy
            self.entropy_idx += 1

        # Convert to list for first item in batch (assuming batch routing)
        selected_experts_list = selected_experts[0].tolist()

        # Return weights only if requested and not hard routing
        if return_routing_weights and not hard_routing:
            return features, selected_experts_list, routing_weights[0]
        else:
            return features, selected_experts_list, None

    def _competitive_activation(
        self,
        logits: torch.Tensor,
        num_iterations: int = 3,
    ) -> torch.Tensor:
        """
        Apply winner-take-most competition with lateral inhibition.

        For constant-VRAM: Uses scalar inhibition instead of O(n²) matrix.
        Each expert inhibits all others uniformly by: lateral_inhibition * activation
        This gives winner-take-most dynamics with O(1) memory instead of O(n²).

        Iteratively updates: a_e = ReLU(r_e - θ - λ * Σ_{e'≠e} a_{e'})
        """
        activations = torch.relu(logits - self.threshold)

        for _ in range(num_iterations):
            # Compute total activation from all experts
            total_activation = activations.sum(dim=-1, keepdim=True)

            # Each expert is inhibited by all others uniformly
            # Subtract own activation to get inhibition from others only
            inhibition = self.lateral_inhibition * (total_activation - activations)

            # Update activations with inhibition
            activations = torch.relu(logits - self.threshold - inhibition)

        return activations

    def hebbian_update(
        self,
        features: torch.Tensor,
        expert_errors: torch.Tensor,
    ):
        """
        Update routing scores using Hebbian learning rule.

        Δr_e = η_r · corr(φ(x), ΔL_e)

        Args:
            features: Projected input features (batch, router_size)
            expert_errors: Error reduction for each expert (num_experts,)
        """
        if not self.training:
            return

        with torch.no_grad():
            # Compute correlation between features and error reduction
            # Higher correlation = expert is better for this input type
            feature_magnitude = features.norm(dim=-1, keepdim=True).mean()

            for expert_idx in range(self.num_experts):
                if expert_errors[expert_idx] != 0:
                    # Correlation-based update
                    error_signal = expert_errors[expert_idx]

                    # Update routing score (higher for better experts)
                    delta = self.learning_rate * error_signal / (feature_magnitude + 1e-8)
                    self.routing_scores[expert_idx] += delta.item()

            # Normalize routing scores to prevent drift, but maintain diversity
            # Use temperature to control concentration
            temperature = 0.5  # Lower = more diverse, higher = more concentrated
            normalized_scores = F.softmax(self.routing_scores / temperature, dim=0)
            self.routing_scores.data.copy_(normalized_scores)

    def _compute_entropy(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of expert activations."""
        probs = F.softmax(activations, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return entropy

    def get_routing_entropy(self) -> float:
        """Get average routing entropy over recent history."""
        return self.routing_entropy_history[
            max(0, self.entropy_idx - 100):self.entropy_idx
        ].mean().item()

    def get_expert_balance(self) -> float:
        """
        Get expert load balance (1.0 = perfect balance, 0.0 = all on one expert).

        Uses coefficient of variation of activation counts.
        """
        if self.expert_activation_counts.sum() == 0:
            return 1.0

        counts = self.expert_activation_counts.float()
        mean = counts.mean()
        std = counts.std()

        if mean == 0:
            return 1.0

        cv = std / mean  # Coefficient of variation
        balance = 1.0 / (1.0 + cv)  # Normalize to [0, 1]
        return balance.item()

    def check_stability(self) -> dict:
        """
        Check routing stability according to Proposition 1.

        Returns metrics indicating convergence:
        - routing_entropy: Should stay above min threshold
        - expert_balance: Should stay above min threshold
        - routing_variance: Should decrease over time
        """
        entropy = self.get_routing_entropy()
        balance = self.get_expert_balance()

        # Compute variance of recent routing scores
        recent_history = self.routing_entropy_history[
            max(0, self.entropy_idx - 100):self.entropy_idx
        ]
        variance = recent_history.var().item() if len(recent_history) > 1 else 0.0

        return {
            "routing_entropy": entropy,
            "expert_balance": balance,
            "routing_variance": variance,
            "is_stable": entropy > 1.0 and balance > 0.5,
        }

    def get_memory_size_mb(self) -> float:
        """Estimate memory footprint in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
