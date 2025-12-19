"""Homeostatic regulation for operational stability."""
import torch
import torch.nn as nn
from typing import Dict


class HomeostasisRegulator(nn.Module):
    """
    Homeostatic regulation to maintain operational stability.

    L_homeostatic = λ_u·H(uncertainty) + λ_s·H(surprise) + λ_c·churn(experts)

    This is not reward learning but metabolic constraint ensuring stable operation.
    """

    def __init__(
        self,
        lambda_uncertainty: float = 0.1,
        lambda_surprise: float = 0.1,
        lambda_churn: float = 0.05,
        target_entropy: float = 2.0,
    ):
        super().__init__()
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_surprise = lambda_surprise
        self.lambda_churn = lambda_churn
        self.target_entropy = target_entropy

        # Track previous expert activations for churn computation
        self.register_buffer("prev_expert_activations", None)

    def forward(
        self,
        prediction_entropy: torch.Tensor,
        surprise: torch.Tensor,
        expert_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute homeostatic penalty.

        Args:
            prediction_entropy: Entropy of model predictions
            surprise: Surprise signal (negative log probability)
            expert_activations: Binary tensor of active experts (num_experts,)

        Returns:
            homeostatic_loss: Penalty for deviations from homeostasis
        """
        # Uncertainty penalty (entropy deviation from target)
        uncertainty_penalty = self._uncertainty_penalty(prediction_entropy)

        # Surprise penalty (excessive surprise is metabolically costly)
        surprise_penalty = self._surprise_penalty(surprise)

        # Expert churn penalty (frequent switching is costly)
        churn_penalty = self._churn_penalty(expert_activations)

        # Combine penalties
        total_penalty = (
            self.lambda_uncertainty * uncertainty_penalty +
            self.lambda_surprise * surprise_penalty +
            self.lambda_churn * churn_penalty
        )

        return total_penalty

    def _uncertainty_penalty(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Penalize excessive uncertainty (entropy too high or too low).

        Optimal entropy is near target_entropy.
        """
        deviation = (entropy - self.target_entropy).abs()
        penalty = deviation.mean()
        return penalty

    def _surprise_penalty(self, surprise: torch.Tensor) -> torch.Tensor:
        """
        Penalize excessive surprise.

        High surprise indicates model is unprepared for inputs (costly).
        """
        # Use squared surprise to penalize extreme values more
        penalty = (surprise ** 2).mean()
        return penalty

    def _churn_penalty(self, expert_activations: torch.Tensor) -> torch.Tensor:
        """
        Penalize frequent changes in expert activations.

        Expert switching has overhead (loading/unloading from memory).
        """
        if self.prev_expert_activations is None:
            # First call, no churn yet
            self.prev_expert_activations = expert_activations.detach()
            return torch.tensor(0.0, device=expert_activations.device)

        # Compute number of experts that changed state
        changes = (expert_activations != self.prev_expert_activations).float()
        churn = changes.sum() / len(expert_activations)

        # Update history
        self.prev_expert_activations = expert_activations.detach()

        return churn

    def get_statistics(
        self,
        prediction_entropy: torch.Tensor,
        surprise: torch.Tensor,
    ) -> Dict[str, float]:
        """Get homeostatic statistics."""
        return {
            "prediction_entropy": prediction_entropy.mean().item(),
            "surprise": surprise.mean().item(),
            "target_entropy": self.target_entropy,
            "entropy_deviation": (prediction_entropy.mean() - self.target_entropy).abs().item(),
        }
