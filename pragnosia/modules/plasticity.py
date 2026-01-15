"""
Plasticity Engine for Pragnosia.

Implements dynamic neural architecture adaptation inspired by brain neuroplasticity:
- Expert growth: Add new experts when routing entropy is high
- Expert pruning: Remove underutilized experts
- Capacity rebalancing: Redistribute capacity across experts
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlasticityMetrics:
    """Metrics for plasticity decisions."""
    routing_entropy: float
    expert_utilization: torch.Tensor
    loss_trend: List[float]
    expert_gradient_norms: Optional[torch.Tensor] = None


@dataclass
class PlasticityAction:
    """Action taken by plasticity engine."""
    action_type: str  # "grow", "prune", "rebalance", "none"
    expert_id: Optional[int] = None
    reason: str = ""


class PlasticityEngine:
    """
    Manages dynamic expert growth and pruning.

    Growth triggers:
    - High routing entropy (uncertain which expert to use)
    - Persistent loss spikes (current capacity insufficient)
    - Expert saturation (all experts heavily utilized)

    Pruning triggers:
    - Low activation frequency (expert rarely used)
    - Redundant gradients (expert duplicates another)
    - Overfitting indicators
    """

    def __init__(
        self,
        growth_threshold: float = 0.8,  # Entropy threshold for growth
        prune_threshold: float = 0.05,  # Utilization threshold for pruning
        max_experts: int = 16,
        min_experts: int = 4,
        warmup_steps: int = 1000,  # Steps before plasticity kicks in
        check_frequency: int = 100,  # How often to check for plasticity
        loss_window: int = 50,  # Window for loss trend analysis
    ):
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.warmup_steps = warmup_steps
        self.check_frequency = check_frequency
        self.loss_window = loss_window

        # Tracking
        self.step_count = 0
        self.loss_history: List[float] = []
        self.entropy_history: List[float] = []
        self.actions_taken: List[PlasticityAction] = []

    def should_check(self) -> bool:
        """Check if we should evaluate plasticity this step."""
        if self.step_count < self.warmup_steps:
            return False
        return self.step_count % self.check_frequency == 0

    def record_step(
        self,
        loss: float,
        routing_entropy: float,
    ):
        """Record metrics for this training step."""
        self.step_count += 1
        self.loss_history.append(loss)
        self.entropy_history.append(routing_entropy)

        # Keep only recent history
        if len(self.loss_history) > self.loss_window * 2:
            self.loss_history = self.loss_history[-self.loss_window:]
            self.entropy_history = self.entropy_history[-self.loss_window:]

    def analyze(
        self,
        model: nn.Module,
        routing_entropy: float,
        expert_utilization: torch.Tensor,
    ) -> PlasticityAction:
        """
        Analyze model state and decide on plasticity action.

        Args:
            model: The Pragnosia model
            routing_entropy: Current routing entropy
            expert_utilization: Per-expert utilization ratios

        Returns:
            PlasticityAction describing what to do
        """
        num_experts = len(expert_utilization)

        # Check for growth conditions
        if num_experts < self.max_experts:
            # High entropy indicates router is uncertain
            if routing_entropy > self.growth_threshold:
                return PlasticityAction(
                    action_type="grow",
                    reason=f"High routing entropy ({routing_entropy:.3f} > {self.growth_threshold})"
                )

            # Loss spike detection
            if len(self.loss_history) >= self.loss_window:
                recent_loss = sum(self.loss_history[-10:]) / 10
                earlier_loss = sum(self.loss_history[-self.loss_window:-10]) / (self.loss_window - 10)
                if recent_loss > earlier_loss * 1.2:  # 20% increase
                    return PlasticityAction(
                        action_type="grow",
                        reason=f"Loss spike detected ({recent_loss:.4f} vs {earlier_loss:.4f})"
                    )

            # Expert saturation
            if (expert_utilization > 0.9).all():
                return PlasticityAction(
                    action_type="grow",
                    reason="All experts saturated (utilization > 90%)"
                )

        # Check for pruning conditions
        if num_experts > self.min_experts:
            # Find underutilized experts
            underutilized = (expert_utilization < self.prune_threshold).nonzero(as_tuple=True)[0]
            if len(underutilized) > 0:
                expert_to_prune = underutilized[0].item()
                return PlasticityAction(
                    action_type="prune",
                    expert_id=expert_to_prune,
                    reason=f"Expert {expert_to_prune} underutilized "
                           f"({expert_utilization[expert_to_prune]:.3f} < {self.prune_threshold})"
                )

        return PlasticityAction(action_type="none", reason="No plasticity needed")

    def grow_expert(
        self,
        cognitive_cortex: nn.Module,
        expert_type: str = "general",
    ) -> int:
        """
        Add a new expert to the cognitive cortex.

        Args:
            cognitive_cortex: The CognitiveCortex module
            expert_type: Type of expert to add

        Returns:
            ID of the new expert
        """
        from pragnosia.modules.experts import Expert

        # Create new expert
        new_expert = Expert(
            hidden_size=cognitive_cortex.hidden_size,
            intermediate_size=cognitive_cortex.intermediate_size,
            expert_type=expert_type,
        )

        # Move to same device as existing experts
        device = next(cognitive_cortex.parameters()).device
        new_expert = new_expert.to(device)

        # Add to expert list
        new_expert_id = len(cognitive_cortex.experts)
        cognitive_cortex.experts.append(new_expert)
        cognitive_cortex.num_experts += 1

        # Update router
        old_gate = cognitive_cortex.router.gate
        new_gate = nn.Linear(
            cognitive_cortex.hidden_size,
            cognitive_cortex.num_experts,
            bias=False,
        ).to(device)

        # Copy old weights
        with torch.no_grad():
            new_gate.weight[:new_expert_id] = old_gate.weight
            # Initialize new expert gate with small random values
            nn.init.normal_(new_gate.weight[new_expert_id:], std=0.01)

        cognitive_cortex.router.gate = new_gate
        cognitive_cortex.router.num_experts = cognitive_cortex.num_experts

        # Update activation tracking buffer
        old_counts = cognitive_cortex.router.expert_activation_counts
        new_counts = torch.zeros(cognitive_cortex.num_experts, device=device)
        new_counts[:new_expert_id] = old_counts
        cognitive_cortex.router.expert_activation_counts = new_counts

        logger.info(f"Grew expert {new_expert_id} (type: {expert_type})")
        return new_expert_id

    def prune_expert(
        self,
        cognitive_cortex: nn.Module,
        expert_id: int,
    ):
        """
        Remove an expert from the cognitive cortex.

        Args:
            cognitive_cortex: The CognitiveCortex module
            expert_id: ID of expert to remove
        """
        if len(cognitive_cortex.experts) <= self.min_experts:
            logger.warning(f"Cannot prune: already at minimum experts ({self.min_experts})")
            return

        device = next(cognitive_cortex.parameters()).device

        # Remove expert from list
        del cognitive_cortex.experts[expert_id]
        cognitive_cortex.num_experts -= 1

        # Update router
        old_gate = cognitive_cortex.router.gate
        new_gate = nn.Linear(
            cognitive_cortex.hidden_size,
            cognitive_cortex.num_experts,
            bias=False,
        ).to(device)

        # Copy weights, skipping the pruned expert
        with torch.no_grad():
            mask = torch.ones(old_gate.weight.shape[0], dtype=torch.bool)
            mask[expert_id] = False
            new_gate.weight.copy_(old_gate.weight[mask])

        cognitive_cortex.router.gate = new_gate
        cognitive_cortex.router.num_experts = cognitive_cortex.num_experts

        # Update activation tracking
        old_counts = cognitive_cortex.router.expert_activation_counts
        mask = torch.ones(len(old_counts), dtype=torch.bool, device=device)
        mask[expert_id] = False
        cognitive_cortex.router.expert_activation_counts = old_counts[mask]

        logger.info(f"Pruned expert {expert_id}")

    def step(
        self,
        model: nn.Module,
        loss: float,
        routing_entropy: float,
        expert_utilization: torch.Tensor,
    ) -> PlasticityAction:
        """
        Perform one plasticity step.

        Args:
            model: The Pragnosia model
            loss: Current training loss
            routing_entropy: Current routing entropy
            expert_utilization: Per-expert utilization

        Returns:
            Action taken (if any)
        """
        self.record_step(loss, routing_entropy)

        if not self.should_check():
            return PlasticityAction(action_type="none", reason="Not checking this step")

        action = self.analyze(model, routing_entropy, expert_utilization)

        # Execute action
        if action.action_type == "grow":
            # Find all cognitive cortex modules and grow them
            for name, module in model.named_modules():
                if hasattr(module, 'experts') and hasattr(module, 'router'):
                    self.grow_expert(module)

        elif action.action_type == "prune" and action.expert_id is not None:
            for name, module in model.named_modules():
                if hasattr(module, 'experts') and hasattr(module, 'router'):
                    self.prune_expert(module, action.expert_id)

        self.actions_taken.append(action)
        return action

    def get_summary(self) -> Dict:
        """Get summary of plasticity actions taken."""
        return {
            "total_steps": self.step_count,
            "actions_taken": len([a for a in self.actions_taken if a.action_type != "none"]),
            "growths": len([a for a in self.actions_taken if a.action_type == "grow"]),
            "prunes": len([a for a in self.actions_taken if a.action_type == "prune"]),
            "recent_entropy": self.entropy_history[-10:] if self.entropy_history else [],
            "recent_loss": self.loss_history[-10:] if self.loss_history else [],
        }
