"""Reference-based calibration using frozen teacher checkpoints.

Prevents representational drift while allowing task alignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ReferenceCalibrator:
    """
    Reference-based calibration for local learning.

    Key idea: Save a "frozen teacher" checkpoint at end of Phase A (representation).
    During Phases B and C, measure divergence from teacher representations.
    Use divergence as a calibration signal to prevent catastrophic drift.

    This is critical for local learning because:
    - Without global gradients, experts can drift arbitrarily
    - Task alignment might destroy good representations
    - Need anchor point to maintain stability
    """

    def __init__(
        self,
        calibration_weight: float = 0.1,
        divergence_threshold: float = 2.0,
        use_kl_divergence: bool = True,
    ):
        """
        Initialize reference calibrator.

        Args:
            calibration_weight: Weight for calibration loss (0.0 = disabled)
            divergence_threshold: Alert if divergence exceeds this
            use_kl_divergence: Use KL divergence (True) vs MSE (False)
        """
        self.calibration_weight = calibration_weight
        self.divergence_threshold = divergence_threshold
        self.use_kl_divergence = use_kl_divergence

        # Storage for frozen teacher
        self.teacher_model = None
        self.teacher_checkpoint_step = None
        self.calibration_active = False

        # Track divergence statistics
        self.divergence_history = []
        self.max_divergence = 0.0

    def save_teacher_checkpoint(self, model: nn.Module, step: int):
        """
        Save frozen teacher checkpoint.

        Typically called at end of Phase A (representation formation).
        Creates a copy of the model using state_dict to avoid pickling issues.

        Args:
            model: Model to checkpoint
            step: Training step when checkpoint created
        """
        print(f"\n{'='*80}")
        print(f"SAVING FROZEN TEACHER CHECKPOINT")
        print(f"{'='*80}")
        print(f"Step: {step}")
        print(f"Purpose: Anchor representations to prevent drift during alignment")
        print(f"{'='*80}\n")

        # Save state dict to avoid pickling threading.Lock and other non-serializable objects
        state_dict = model.state_dict()

        # Create new model instance with same config
        # Import here to avoid circular dependency
        from ..models.pragnosia_model import PragnosiaModel

        # Create teacher model with same configuration
        self.teacher_model = PragnosiaModel(model.config)

        # Load saved state
        self.teacher_model.load_state_dict(state_dict)
        self.teacher_model.eval()

        # Freeze all parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.teacher_checkpoint_step = step
        self.calibration_active = True

    def compute_calibration_loss(
        self,
        student_hidden: torch.Tensor,
        student_logits: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute calibration loss relative to frozen teacher.

        Args:
            student_hidden: Current model hidden states (batch, seq, hidden)
            student_logits: Current model logits (batch, seq, vocab) - can be None for hidden-only
            input_ids: Input token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)

        Returns:
            Dictionary containing:
            - calibration_loss: Loss penalizing drift from teacher
            - divergence: Divergence metric
            - alert: Whether divergence exceeds threshold
        """
        if not self.calibration_active or self.teacher_model is None:
            # No calibration yet
            return {
                "calibration_loss": torch.tensor(0.0, device=student_hidden.device),
                "divergence": torch.tensor(0.0, device=student_hidden.device),
                "alert": False,
            }

        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_hidden = teacher_outputs["hidden_states"]
            teacher_logits = teacher_outputs["logits"]

        # Compute divergence
        if self.use_kl_divergence and student_logits is not None:
            # KL divergence between output distributions
            student_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            # KL(teacher || student)
            kl_div = F.kl_div(
                student_probs,
                teacher_probs,
                reduction="batchmean",
                log_target=False,
            )
            divergence = kl_div
            calibration_loss = kl_div
        else:
            # MSE on hidden representations (default if no logits)
            mse = F.mse_loss(student_hidden, teacher_hidden, reduction="mean")
            divergence = mse
            calibration_loss = mse

        # Track divergence
        div_value = divergence.item()
        self.divergence_history.append(div_value)
        if len(self.divergence_history) > 1000:
            self.divergence_history = self.divergence_history[-1000:]

        self.max_divergence = max(self.max_divergence, div_value)

        # Check for excessive drift
        alert = div_value > self.divergence_threshold

        if alert:
            print(f"\n{'!'*80}")
            print(f"CALIBRATION ALERT: Excessive drift from teacher")
            print(f"Divergence: {div_value:.4f} (threshold: {self.divergence_threshold})")
            print(f"Consider reducing learning rate or increasing calibration weight")
            print(f"{'!'*80}\n")

        # Weight calibration loss
        weighted_loss = self.calibration_weight * calibration_loss

        return {
            "calibration_loss": weighted_loss,
            "divergence": divergence.detach(),
            "alert": alert,
        }

    def get_statistics(self) -> Dict:
        """Get calibration statistics."""
        if len(self.divergence_history) == 0:
            return {
                "active": self.calibration_active,
                "teacher_step": self.teacher_checkpoint_step,
                "mean_divergence": 0.0,
                "max_divergence": 0.0,
                "recent_divergence": 0.0,
            }

        import numpy as np
        divergences = np.array(self.divergence_history)

        return {
            "active": self.calibration_active,
            "teacher_step": self.teacher_checkpoint_step,
            "mean_divergence": float(divergences.mean()),
            "max_divergence": self.max_divergence,
            "recent_divergence": float(divergences[-10:].mean()) if len(divergences) >= 10 else 0.0,
            "num_alerts": sum(1 for d in divergences if d > self.divergence_threshold),
        }

    def adjust_weight(self, new_weight: float):
        """Dynamically adjust calibration weight."""
        old_weight = self.calibration_weight
        self.calibration_weight = new_weight
        print(f"Calibration weight adjusted: {old_weight:.4f} → {new_weight:.4f}")

    def should_save_teacher(self, phase_name: str, phase_progress: float) -> bool:
        """
        Determine if teacher checkpoint should be saved.

        Typically save at end of Phase A (representation formation).

        Args:
            phase_name: Current phase name
            phase_progress: Progress within phase (0.0 to 1.0)

        Returns:
            True if should save teacher checkpoint
        """
        # Save at end of representation phase (when transitioning to alignment)
        if phase_name == "representation" and phase_progress > 0.95:
            if not self.calibration_active:
                return True
        return False

    def to(self, device: torch.device):
        """Move teacher model to device."""
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(device)
