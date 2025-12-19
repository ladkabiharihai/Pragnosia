"""Neuromodulatory gating for adaptive intrinsic learning."""
import torch
import torch.nn as nn
from typing import Dict, Optional


class Neuromodulator(nn.Module):
    """
    Brain-inspired neuromodulation for adaptive intrinsic learning.

    Mimics dopaminergic and noradrenergic systems that modulate learning
    based on novelty, surprise, and stability.

    Key insight: When model becomes too stable → increase curiosity
                 When model is chaotic → decrease exploration

    Formula: modulation = sigmoid(α * novelty - β * stability + γ * prediction_error)

    This creates a homeostatic feedback loop that prevents:
    - Over-stabilization (intrinsic learning collapse)
    - Over-exploration (training instability)
    """

    def __init__(
        self,
        history_length: int = 100,
        novelty_weight: float = 2.0,
        stability_weight: float = 1.0,
        error_weight: float = 1.5,
        baseline_modulation: float = 1.0,
        tau: float = 0.9,  # EMA smoothing factor
    ):
        super().__init__()

        self.history_length = history_length
        self.novelty_weight = novelty_weight
        self.stability_weight = stability_weight
        self.error_weight = error_weight
        self.baseline_modulation = baseline_modulation
        self.tau = tau

        # History buffers for computing statistics
        self.register_buffer("loss_history", torch.zeros(history_length))
        self.register_buffer("intrinsic_history", torch.zeros(history_length))
        self.register_buffer("surprise_history", torch.zeros(history_length))
        self.step_idx = 0

        # Running statistics (EMA)
        self.register_buffer("ema_loss", torch.tensor(0.0))
        self.register_buffer("ema_intrinsic", torch.tensor(0.0))
        self.register_buffer("ema_surprise", torch.tensor(0.0))
        self.register_buffer("ema_variance", torch.tensor(1.0))

        # Current modulation value
        self.register_buffer("current_modulation", torch.tensor(baseline_modulation))

    def forward(
        self,
        intrinsic_loss: torch.Tensor,
        prediction_loss: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute neuromodulation signal for intrinsic learning.

        Args:
            intrinsic_loss: Current intrinsic learning signal
            prediction_loss: Current prediction error
            surprise: Optional surprise signal (prediction error on targets)

        Returns:
            modulation: Scalar multiplier for intrinsic loss (typically 0.5 to 2.0)
        """
        # Detach to avoid gradient flow (modulation is meta-learning)
        intrinsic_val = intrinsic_loss.detach().mean().item()
        loss_val = prediction_loss.detach().mean().item()
        surprise_val = surprise.detach().mean().item() if surprise is not None else loss_val

        # Update history
        self.loss_history[self.step_idx % self.history_length] = loss_val
        self.intrinsic_history[self.step_idx % self.history_length] = intrinsic_val
        self.surprise_history[self.step_idx % self.history_length] = surprise_val
        self.step_idx += 1

        # Update EMA statistics
        if self.step_idx == 1:
            # Initialize
            self.ema_loss = torch.tensor(loss_val)
            self.ema_intrinsic = torch.tensor(intrinsic_val)
            self.ema_surprise = torch.tensor(surprise_val)
        else:
            self.ema_loss = self.tau * self.ema_loss + (1 - self.tau) * loss_val
            self.ema_intrinsic = self.tau * self.ema_intrinsic + (1 - self.tau) * intrinsic_val
            self.ema_surprise = self.tau * self.ema_surprise + (1 - self.tau) * surprise_val

        # Only start modulating after warmup
        if self.step_idx < 10:
            self.current_modulation = torch.tensor(self.baseline_modulation)
            return self.current_modulation

        # Compute novelty (how much current state differs from recent history)
        recent_window = min(20, self.step_idx)
        recent_losses = self._get_recent_values(self.loss_history, recent_window)
        recent_intrinsic = self._get_recent_values(self.intrinsic_history, recent_window)

        # Novelty = deviation from recent mean
        loss_deviation = abs(loss_val - recent_losses.mean().item())
        intrinsic_deviation = abs(intrinsic_val - recent_intrinsic.mean().item())
        novelty = loss_deviation + intrinsic_deviation

        # Compute stability (inverse of recent variance)
        # High variance = low stability (chaotic learning)
        # Low variance = high stability (risk of over-stabilization)
        loss_variance = recent_losses.var().item()
        intrinsic_variance = recent_intrinsic.var().item()
        stability = 1.0 / (1.0 + loss_variance + intrinsic_variance)

        # Update variance EMA
        self.ema_variance = self.tau * self.ema_variance + (1 - self.tau) * (loss_variance + intrinsic_variance)

        # Compute prediction error trend (are we still improving?)
        if self.step_idx >= 20:
            # Compare first half vs second half of recent window
            half = recent_window // 2
            early_loss = recent_losses[:half].mean()
            late_loss = recent_losses[half:].mean()
            improvement = (early_loss - late_loss).item()  # Positive = improving
        else:
            improvement = 0.0

        # Neuromodulation formula:
        # High novelty → increase intrinsic (explore new patterns)
        # High stability → increase intrinsic (prevent collapse)
        # No improvement → increase intrinsic (stuck in local minimum)

        modulation_signal = (
            self.novelty_weight * novelty +
            self.stability_weight * stability +
            self.error_weight * max(0.0, -improvement)  # Negative improvement = not learning
        )

        # Apply sigmoid to bound modulation in [0.5, 2.0]
        # This ensures intrinsic learning never completely shuts off
        # but also doesn't explode
        normalized_signal = modulation_signal - 1.0  # Center around 0
        modulation = 0.5 + 1.5 * torch.sigmoid(torch.tensor(normalized_signal))

        # Smooth modulation changes (avoid sudden jumps)
        smoothing = 0.8
        self.current_modulation = (
            smoothing * self.current_modulation +
            (1 - smoothing) * modulation
        )

        return self.current_modulation

    def _get_recent_values(self, history: torch.Tensor, window: int) -> torch.Tensor:
        """Extract recent values from circular buffer."""
        if self.step_idx < window:
            return history[:self.step_idx]

        start_idx = (self.step_idx - window) % self.history_length
        end_idx = self.step_idx % self.history_length

        if end_idx > start_idx:
            return history[start_idx:end_idx]
        else:
            # Handle wrap-around
            return torch.cat([history[start_idx:], history[:end_idx]])

    def get_statistics(self) -> Dict[str, float]:
        """Get neuromodulator statistics for logging."""
        if self.step_idx < 10:
            return {
                "modulation": self.baseline_modulation,
                "ema_loss": 0.0,
                "ema_intrinsic": 0.0,
                "ema_variance": 0.0,
                "novelty": 0.0,
                "stability": 0.0,
            }

        # Compute current statistics
        recent_window = min(20, self.step_idx)
        recent_losses = self._get_recent_values(self.loss_history, recent_window)
        recent_intrinsic = self._get_recent_values(self.intrinsic_history, recent_window)

        current_loss = self.loss_history[(self.step_idx - 1) % self.history_length].item()
        current_intrinsic = self.intrinsic_history[(self.step_idx - 1) % self.history_length].item()

        novelty = abs(current_loss - recent_losses.mean().item()) + \
                  abs(current_intrinsic - recent_intrinsic.mean().item())

        loss_variance = recent_losses.var().item()
        intrinsic_variance = recent_intrinsic.var().item()
        stability = 1.0 / (1.0 + loss_variance + intrinsic_variance)

        return {
            "modulation": self.current_modulation.item(),
            "ema_loss": self.ema_loss.item(),
            "ema_intrinsic": self.ema_intrinsic.item(),
            "ema_variance": self.ema_variance.item(),
            "novelty": novelty,
            "stability": stability,
        }

    def reset(self):
        """Reset neuromodulator state (e.g., for new task)."""
        self.loss_history.zero_()
        self.intrinsic_history.zero_()
        self.surprise_history.zero_()
        self.step_idx = 0
        self.ema_loss.zero_()
        self.ema_intrinsic.zero_()
        self.ema_surprise.zero_()
        self.ema_variance.fill_(1.0)
        self.current_modulation.fill_(self.baseline_modulation)
