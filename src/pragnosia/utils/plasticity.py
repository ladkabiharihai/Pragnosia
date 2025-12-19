"""Controlled neuroplasticity with developmental phases."""
import torch
import torch.nn as nn
from enum import Enum
from typing import Dict, Optional


class PlasticityPhase(Enum):
    """Developmental phases for controlled neuroplasticity."""
    EXPLORATION = "exploration"  # Growth enabled, pruning disabled
    STABILIZATION = "stabilization"  # Growth disabled, pruning enabled
    EXPLOITATION = "exploitation"  # Both disabled


class PlasticityScheduler:
    """
    Schedule neuroplasticity operations across developmental phases.

    Phases:
    - Exploration (0-30%): Growth enabled, pruning disabled
    - Stabilization (30-70%): Growth disabled, pruning enabled
    - Exploitation (70-100%): Both disabled
    """

    def __init__(
        self,
        total_steps: int,
        exploration_end: float = 0.3,
        stabilization_end: float = 0.7,
        max_growth_rate: float = 0.01,
        max_pruning_rate: float = 0.01,
        min_active_params: float = 0.3,
        max_active_params: float = 0.9,
        min_expert_entropy: float = 2.0,
    ):
        self.total_steps = total_steps
        self.exploration_end = int(exploration_end * total_steps)
        self.stabilization_end = int(stabilization_end * total_steps)
        self.max_growth_rate = max_growth_rate
        self.max_pruning_rate = max_pruning_rate
        self.min_active_params = min_active_params
        self.max_active_params = max_active_params
        self.min_expert_entropy = min_expert_entropy

        self.current_step = 0
        self.safety_violations = 0

    def step(self) -> PlasticityPhase:
        """Advance scheduler and return current phase."""
        self.current_step += 1
        return self.get_phase()

    def get_phase(self) -> PlasticityPhase:
        """Get current developmental phase."""
        if self.current_step < self.exploration_end:
            return PlasticityPhase.EXPLORATION
        elif self.current_step < self.stabilization_end:
            return PlasticityPhase.STABILIZATION
        else:
            return PlasticityPhase.EXPLOITATION

    def can_grow(self) -> bool:
        """Check if growth is permitted in current phase."""
        return self.get_phase() == PlasticityPhase.EXPLORATION

    def can_prune(self) -> bool:
        """Check if pruning is permitted in current phase."""
        return self.get_phase() == PlasticityPhase.STABILIZATION

    def get_growth_rate(self, step: Optional[int] = None) -> float:
        """
        Get current growth rate (linearly decays during exploration).

        Returns rate in [0, max_growth_rate].
        """
        if not self.can_grow():
            return 0.0

        step = step or self.current_step
        progress = step / self.exploration_end
        # Decay from max to 0
        rate = self.max_growth_rate * (1.0 - progress)
        return rate

    def get_pruning_rate(self, step: Optional[int] = None) -> float:
        """
        Get current pruning rate (linearly increases during stabilization).

        Returns rate in [0, max_pruning_rate].
        """
        if not self.can_prune():
            return 0.0

        step = step or self.current_step
        stabilization_progress = (
            (step - self.exploration_end) /
            (self.stabilization_end - self.exploration_end)
        )
        # Ramp from 0 to max
        rate = self.max_pruning_rate * stabilization_progress
        return rate

    def check_safety_bounds(
        self,
        active_param_ratio: float,
        expert_entropy: float,
    ) -> Dict[str, bool]:
        """
        Check structural safety bounds.

        Returns dict indicating which bounds are violated.
        """
        violations = {
            "active_params_too_low": active_param_ratio < self.min_active_params,
            "active_params_too_high": active_param_ratio > self.max_active_params,
            "expert_entropy_too_low": expert_entropy < self.min_expert_entropy,
        }

        if any(violations.values()):
            self.safety_violations += 1

        return violations

    def get_statistics(self) -> Dict[str, any]:
        """Get scheduler statistics."""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": self.current_step / self.total_steps,
            "phase": self.get_phase().value,
            "can_grow": self.can_grow(),
            "can_prune": self.can_prune(),
            "growth_rate": self.get_growth_rate(),
            "pruning_rate": self.get_pruning_rate(),
            "safety_violations": self.safety_violations,
        }
