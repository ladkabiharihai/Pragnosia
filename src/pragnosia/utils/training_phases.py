"""
Training phase scheduler for local learning.

Implements staged learning progression:
Phase A: Representation Formation (intrinsic-driven)
Phase B: Task Alignment (balance intrinsic + task)
Phase C: Stabilization (freeze mature experts)

This is CRITICAL for local learning systems.
Standard backprop can learn everything at once.
Local learning requires explicit staging.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""

    name: str
    duration_steps: int

    # Intrinsic vs task weighting
    intrinsic_weight: float  # How much to weight intrinsic learning
    task_weight: float       # How much to weight task loss

    # Intrinsic component weights
    surprise_weight: float
    temporal_weight: float
    disagreement_weight: float
    compression_weight: float
    entropy_weight: float  # NEW: Sequence-level entropy (critical for local learning)

    # Expert dynamics
    expert_learning_rate: float
    allow_expert_freezing: bool
    maturity_threshold: float  # Stability threshold for freezing

    # Routing behavior
    routing_temperature: float  # Higher = more exploration

    description: str


class TrainingPhaseScheduler:
    """
    Manages training phases for local learning.

    Key principles:
    1. Early: Learn representations (intrinsic dominates)
    2. Middle: Align with task (balance intrinsic + task)
    3. Late: Stabilize (freeze mature experts)
    """

    def __init__(self, total_training_steps: int):
        self.total_training_steps = total_training_steps
        self.current_step = 0

        # Define phases
        self.phases = self._create_phases(total_training_steps)

        # Track phase transitions
        self.current_phase_idx = 0
        self.phase_start_step = 0

    def _create_phases(self, total_steps: int) -> list:
        """Create phase configurations."""

        # Calculate phase durations
        # Phase A: 30% of training
        # Phase B: 50% of training
        # Phase C: 20% of training

        phase_a_steps = int(total_steps * 0.30)
        phase_b_steps = int(total_steps * 0.50)
        phase_c_steps = total_steps - phase_a_steps - phase_b_steps

        phases = [
            # PHASE A: REPRESENTATION FORMATION
            PhaseConfig(
                name="representation",
                duration_steps=phase_a_steps,

                # Intrinsic dominates
                intrinsic_weight=0.9,
                task_weight=0.1,

                # Encourage structure learning
                surprise_weight=0.2,
                temporal_weight=0.5,    # High temporal = learn stable representations
                disagreement_weight=0.2,  # Encourage diversity
                compression_weight=0.1,
                entropy_weight=0.3,  # Moderate entropy enforcement during representation learning

                # Fast expert learning
                expert_learning_rate=0.01,
                allow_expert_freezing=False,  # Don't freeze yet
                maturity_threshold=0.0,

                # High exploration
                routing_temperature=1.5,

                description="Learning stable representations. High intrinsic, high temporal."
            ),

            # PHASE B: TASK ALIGNMENT
            PhaseConfig(
                name="alignment",
                duration_steps=phase_b_steps,

                # Task starts to dominate (70/30 split)
                # FIXED: Stronger task signal to ensure proper transition
                intrinsic_weight=0.3,  # Reduced from 0.5
                task_weight=0.7,       # Increased from 0.5

                # Shift to task-relevant signals
                surprise_weight=0.3,     # Increase surprise (task errors)
                temporal_weight=0.2,     # Decrease temporal (already stable)
                disagreement_weight=0.2,
                compression_weight=0.3,  # Increase compression (improve predictions)
                entropy_weight=0.4,      # Stronger entropy enforcement to prevent repetition

                # Moderate learning rate
                expert_learning_rate=0.005,
                allow_expert_freezing=True,   # Start freezing mature experts
                maturity_threshold=0.80,      # FIXED: Lowered from 0.95 to allow freezing

                # Balanced exploration
                routing_temperature=1.0,

                description="Task alignment. Task dominates (70%), intrinsic support (30%)."
            ),

            # PHASE C: STABILIZATION
            PhaseConfig(
                name="stabilization",
                duration_steps=phase_c_steps,

                # Task COMPLETELY dominates (intrinsic learning disabled)
                intrinsic_weight=0.0,  # FIXED: Intrinsic fully yields to task
                task_weight=1.0,

                # Intrinsic weights don't matter (weight=0), but keep for compatibility
                surprise_weight=0.0,
                temporal_weight=0.0,
                disagreement_weight=0.0,
                compression_weight=0.0,
                entropy_weight=0.0,  # Disabled with other intrinsic components

                # Slow learning (fine-tuning)
                expert_learning_rate=0.001,
                allow_expert_freezing=True,
                maturity_threshold=0.75,  # FIXED: Lowered from 0.90 to allow freezing

                # Less exploration
                routing_temperature=0.7,

                description="Pure task learning. Intrinsic disabled, focus on supervised signal only."
            ),
        ]

        return phases

    def step(self):
        """Advance one training step."""
        self.current_step += 1

        # Check if we need to transition to next phase
        steps_in_current_phase = self.current_step - self.phase_start_step
        current_phase = self.phases[self.current_phase_idx]

        if steps_in_current_phase >= current_phase.duration_steps:
            # Transition to next phase
            if self.current_phase_idx < len(self.phases) - 1:
                self.current_phase_idx += 1
                self.phase_start_step = self.current_step
                print(f"\n{'='*80}")
                print(f"PHASE TRANSITION: Entering {self.get_current_phase().name.upper()} phase")
                print(f"{'='*80}")
                print(f"{self.get_current_phase().description}")
                print(f"Steps: {self.current_step}/{self.total_training_steps}")
                print(f"{'='*80}\n")

    def get_current_phase(self) -> PhaseConfig:
        """Get current phase configuration."""
        return self.phases[self.current_phase_idx]

    def get_phase_progress(self) -> float:
        """Get progress within current phase (0.0 to 1.0)."""
        steps_in_phase = self.current_step - self.phase_start_step
        phase_duration = self.phases[self.current_phase_idx].duration_steps
        return min(steps_in_phase / max(phase_duration, 1), 1.0)

    def get_overall_progress(self) -> float:
        """Get overall training progress (0.0 to 1.0)."""
        return min(self.current_step / max(self.total_training_steps, 1), 1.0)

    def get_intrinsic_weights(self) -> Dict[str, float]:
        """Get current intrinsic component weights."""
        phase = self.get_current_phase()
        return {
            "alpha": phase.surprise_weight,
            "beta": phase.temporal_weight,
            "gamma": phase.disagreement_weight,
            "delta": phase.compression_weight,
            "epsilon": phase.entropy_weight,
        }

    def get_loss_weights(self, recent_task_loss: Optional[float] = None) -> Dict[str, float]:
        """
        Get current loss combination weights with state-dependent adaptation.

        FIXED Priority 7: Make intrinsic weighting state-dependent.
        - High task loss → increase intrinsic (need exploration)
        - Low task loss → decrease intrinsic (task signal working)
        """
        phase = self.get_current_phase()
        base_intrinsic = phase.intrinsic_weight
        base_task = phase.task_weight

        # If no recent loss provided, return base weights
        if recent_task_loss is None:
            return {
                "intrinsic": base_intrinsic,
                "task": base_task,
            }

        # State-dependent adaptation based on task performance
        # High loss (>5.0) → boost intrinsic exploration
        # Low loss (<2.0) → reduce intrinsic, trust task signal
        if recent_task_loss > 5.0:
            # Performance struggling → increase intrinsic by up to 0.2
            boost = min(0.2, (recent_task_loss - 5.0) / 20.0)
            adapted_intrinsic = min(1.0, base_intrinsic + boost)
        elif recent_task_loss < 2.0 and base_intrinsic > 0.1:
            # Performance good → reduce intrinsic by up to 0.15
            reduction = min(0.15, (2.0 - recent_task_loss) / 10.0)
            adapted_intrinsic = max(0.0, base_intrinsic - reduction)
        else:
            # Medium loss → use base weights
            adapted_intrinsic = base_intrinsic

        # Normalize to ensure weights sum appropriately
        adapted_task = 1.0 - adapted_intrinsic if adapted_intrinsic < 1.0 else 0.0

        return {
            "intrinsic": adapted_intrinsic,
            "task": adapted_task,
        }

    def should_freeze_expert(self, expert_stability: float) -> bool:
        """Check if an expert should be frozen based on stability."""
        phase = self.get_current_phase()

        if not phase.allow_expert_freezing:
            return False

        return expert_stability >= phase.maturity_threshold

    def get_stats(self) -> Dict:
        """Get current phase statistics."""
        phase = self.get_current_phase()

        return {
            "phase_name": phase.name,
            "phase_idx": self.current_phase_idx,
            "phase_progress": self.get_phase_progress(),
            "overall_progress": self.get_overall_progress(),
            "current_step": self.current_step,
            "total_steps": self.total_training_steps,
            "intrinsic_weight": phase.intrinsic_weight,
            "task_weight": phase.task_weight,
            "expert_lr": phase.expert_learning_rate,
            "can_freeze": phase.allow_expert_freezing,
        }

    def print_phase_info(self):
        """Print current phase information."""
        stats = self.get_stats()
        phase = self.get_current_phase()

        print(f"\n{'─'*80}")
        print(f"Phase: {stats['phase_name'].upper()} "
              f"({stats['phase_idx'] + 1}/{len(self.phases)}) "
              f"[{stats['phase_progress']*100:.1f}% complete]")
        print(f"Overall: {stats['overall_progress']*100:.1f}% "
              f"({stats['current_step']}/{stats['total_steps']} steps)")
        print(f"Weights: Intrinsic={stats['intrinsic_weight']:.2f}, "
              f"Task={stats['task_weight']:.2f}")
        print(f"Expert LR: {stats['expert_lr']:.4f} | "
              f"Freezing: {'ON' if stats['can_freeze'] else 'OFF'}")
        print(f"{'─'*80}\n")
