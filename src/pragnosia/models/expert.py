"""Expert module implementation for Pragnosia."""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ExpertModule(nn.Module):
    """
    Single expert module in the Mixture of Experts architecture.

    Each expert is ~500MB and can be offloaded to CPU when not active.
    Only k=2 experts are kept on GPU at a time for memory efficiency.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
        expert_id: int = 0,
    ):
        super().__init__()
        self.expert_id = expert_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Two-layer FFN with GELU activation
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Expert-specific statistics for Hebbian learning
        self.register_buffer("routing_score", torch.zeros(1))
        self.register_buffer("activation_count", torch.zeros(1))
        self.register_buffer("average_error", torch.zeros(1))
        self.register_buffer("error_history", torch.zeros(100))
        self.error_idx = 0

        # Pruning mask for neuroplasticity
        self.register_buffer("pruning_mask", torch.ones(intermediate_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the expert.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            output: Expert output of shape (batch, seq_len, hidden_size)
            stats: Dictionary of expert statistics
        """
        batch_size = x.shape[0]

        # Update activation count
        self.activation_count += batch_size

        # Forward pass
        residual = x
        x = self.layer_norm(x)
        hidden = self.fc1(x)
        hidden = self.activation(hidden)

        # Apply pruning mask during neuroplasticity
        if self.training and hasattr(self, "pruning_mask"):
            hidden = hidden * self.pruning_mask.unsqueeze(0).unsqueeze(0)

        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        output = self.dropout(output)
        output = output + residual

        # Compute statistics
        stats = {
            "expert_id": self.expert_id,
            "activation_count": self.activation_count.item(),
            "routing_score": self.routing_score.item(),
            "average_error": self.average_error.item(),
            "active_params": self.pruning_mask.float().mean().item(),
        }

        return output, stats

    def update_error(self, error: torch.Tensor):
        """Update error statistics for Hebbian learning."""
        error_val = error.detach().mean().item()
        self.error_history[self.error_idx % 100] = error_val
        self.error_idx += 1
        self.average_error = self.error_history.mean()

    def update_routing_score(self, delta: float):
        """Update routing score using Hebbian update rule."""
        self.routing_score += delta

    def grow_neurons(self, growth_rate: float):
        """Grow new neurons during exploration phase."""
        if not self.training:
            return

        num_to_grow = int(growth_rate * self.intermediate_size)
        if num_to_grow == 0:
            return

        # Find pruned neurons to reactivate
        pruned_indices = (self.pruning_mask == 0).nonzero(as_tuple=True)[0]
        if len(pruned_indices) > 0:
            num_to_grow = min(num_to_grow, len(pruned_indices))
            indices = pruned_indices[torch.randperm(len(pruned_indices))[:num_to_grow]]
            self.pruning_mask[indices] = 1.0

    def prune_neurons(self, pruning_rate: float):
        """Prune low-importance neurons during stabilization phase."""
        if not self.training:
            return

        num_to_prune = int(pruning_rate * self.intermediate_size)
        if num_to_prune == 0:
            return

        # Compute neuron importance based on weight magnitude
        with torch.no_grad():
            importance = (
                self.fc1.weight.abs().mean(dim=0) +
                self.fc2.weight.abs().mean(dim=1)
            )
            importance = importance * self.pruning_mask

            # Find least important active neurons
            active_indices = (self.pruning_mask > 0).nonzero(as_tuple=True)[0]
            if len(active_indices) > 0:
                active_importance = importance[active_indices]
                _, sorted_indices = torch.sort(active_importance)
                num_to_prune = min(num_to_prune, len(sorted_indices))
                prune_indices = active_indices[sorted_indices[:num_to_prune]]
                self.pruning_mask[prune_indices] = 0.0

    def to_cpu(self):
        """Move expert to CPU for memory efficiency."""
        return self.cpu()

    def to_gpu(self, device: torch.device):
        """Load expert to GPU when activated."""
        return self.to(device)

    def get_memory_size_mb(self) -> float:
        """Estimate memory footprint in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
