"""Intrinsic learning objectives for representation learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..utils.neuromodulation import Neuromodulator


class IntrinsicObjective(nn.Module):
    """
    Multi-dimensional intrinsic learning objective.

    Combines five components:
    1. Surprise-weighted prediction error
    2. Temporal consistency loss
    3. Cross-expert disagreement
    4. Compression progress
    5. Sequence-level entropy (CRITICAL for local learning models)

    L_intrinsic = α·L_surprise + β·L_temporal + γ·L_disagreement + δ·L_compression + ε·L_entropy

    The entropy term prevents repetitive generation by penalizing low token diversity.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        alpha: float = 0.3,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.2,
        epsilon: float = 0.3,  # NEW: Sequence entropy weight (critical for local learning)
        learn_weights: bool = True,
        use_neuromodulation: bool = True,
        entropy_window: int = 20,  # Rolling window for entropy computation
        entropy_min_threshold: float = 2.0,  # Minimum entropy target (bits)
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_neuromodulation = use_neuromodulation
        self.entropy_window = entropy_window
        self.entropy_min_threshold = entropy_min_threshold

        # Loss component weights
        if learn_weights:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
            self.gamma = nn.Parameter(torch.tensor(gamma))
            self.delta = nn.Parameter(torch.tensor(delta))
            self.epsilon = nn.Parameter(torch.tensor(epsilon))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta", torch.tensor(beta))
            self.register_buffer("gamma", torch.tensor(gamma))
            self.register_buffer("delta", torch.tensor(delta))
            self.register_buffer("epsilon", torch.tensor(epsilon))

        # Predictor network for temporal prediction
        predictor_hidden = max(hidden_size // 2, 128)
        self.temporal_predictor = nn.Sequential(
            nn.Linear(hidden_size, predictor_hidden),
            nn.LayerNorm(predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, hidden_size),
        )

        # History for compression progress
        self.register_buffer("loss_history", torch.zeros(100))
        self.loss_idx = 0

        # Neuromodulator for adaptive intrinsic learning
        if use_neuromodulation:
            self.neuromodulator = Neuromodulator(
                history_length=100,
                novelty_weight=2.0,
                stability_weight=1.0,
                error_weight=1.5,
                baseline_modulation=1.0,
            )
        else:
            self.neuromodulator = None

    def forward(
        self,
        current_hidden: torch.Tensor,
        previous_hidden: Optional[torch.Tensor] = None,
        expert_outputs: Optional[list] = None,
        prediction_logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        same_context: Optional[torch.Tensor] = None,
        task_loss: Optional[torch.Tensor] = None,
        task_improvement: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute task-aware intrinsic learning objective.

        Args:
            current_hidden: Current hidden states (batch, seq_len, hidden_size)
            previous_hidden: Previous hidden states for temporal consistency
            expert_outputs: List of expert outputs for disagreement
            prediction_logits: Model predictions for surprise
            targets: Target tokens for surprise
            same_context: Boolean mask indicating same context
            task_loss: Current task loss (for task-aware weighting)
            task_improvement: Recent task improvement (positive = getting better)

        Returns:
            total_loss: Combined intrinsic loss
            component_losses: Dictionary of individual loss components
        """
        component_losses = {}

        # Compute task-aware modulation
        # Higher task loss = more intrinsic exploration needed
        # Positive improvement = reduce intrinsic, focus on task
        task_awareness_weight = 1.0
        if task_loss is not None and task_improvement is not None:
            # If improving (task_improvement > 0), reduce intrinsic weight
            # If stuck (task_improvement ≈ 0), increase intrinsic weight
            # Scale: improvement of 1.0 → weight 0.5, improvement of 0.0 → weight 1.0
            improvement_factor = torch.clamp(1.0 - task_improvement * 0.5, 0.5, 1.5)

            # If task loss is high, increase intrinsic (need exploration)
            # Normalize by assuming typical loss range [1, 10]
            task_factor = torch.clamp(task_loss / 5.0, 0.5, 2.0)

            task_awareness_weight = improvement_factor * task_factor
            component_losses["task_awareness_weight"] = task_awareness_weight.detach()

        # 1. Surprise-weighted prediction error
        if prediction_logits is not None and targets is not None:
            surprise_loss = self.surprise_weighted_loss(
                prediction_logits,
                targets,
                current_hidden,
                previous_hidden,
            )
            component_losses["surprise"] = surprise_loss
        else:
            surprise_loss = torch.tensor(0.0, device=current_hidden.device)
            component_losses["surprise"] = surprise_loss

        # 2. Temporal consistency loss
        if previous_hidden is not None:
            temporal_loss = self.temporal_consistency_loss(
                current_hidden,
                previous_hidden,
                same_context,
            )
            component_losses["temporal"] = temporal_loss
        else:
            temporal_loss = torch.tensor(0.0, device=current_hidden.device)
            component_losses["temporal"] = temporal_loss

        # 3. Cross-expert disagreement
        if expert_outputs is not None and len(expert_outputs) > 1:
            disagreement_loss = self.expert_disagreement_loss(expert_outputs)
            component_losses["disagreement"] = disagreement_loss
        else:
            disagreement_loss = torch.tensor(0.0, device=current_hidden.device)
            component_losses["disagreement"] = disagreement_loss

        # 4. Compression progress
        if prediction_logits is not None and targets is not None:
            compression_loss = self.compression_progress_loss(
                prediction_logits,
                targets,
            )
            component_losses["compression"] = compression_loss
        else:
            compression_loss = torch.tensor(0.0, device=current_hidden.device)
            component_losses["compression"] = compression_loss

        # 5. Sequence-level entropy (DISABLED - interfering with learning)
        # The discrete argmax-based entropy computation doesn't work during early training
        # Will re-enable with a different formulation later
        entropy_loss = torch.tensor(0.0, device=current_hidden.device)
        component_losses["entropy"] = entropy_loss

        # Clamp individual loss components to prevent explosion
        surprise_loss = torch.clamp(surprise_loss, 0.0, 100.0)
        temporal_loss = torch.clamp(temporal_loss, 0.0, 100.0)
        disagreement_loss = torch.clamp(disagreement_loss, 0.0, 100.0)
        compression_loss = torch.clamp(compression_loss, 0.0, 10.0)
        entropy_loss = torch.clamp(entropy_loss, 0.0, 20.0)

        # Clamp weights to prevent collapse (min 0.05, max 1.0)
        alpha_clamped = torch.clamp(self.alpha, 0.05, 1.0)
        beta_clamped = torch.clamp(self.beta, 0.05, 1.0)
        gamma_clamped = torch.clamp(self.gamma, 0.05, 1.0)
        delta_clamped = torch.clamp(self.delta, 0.05, 1.0)
        epsilon_clamped = torch.clamp(self.epsilon, 0.05, 1.0)

        # Combine with learned/fixed weights
        total_loss = (
            alpha_clamped * surprise_loss +
            beta_clamped * temporal_loss +
            gamma_clamped * disagreement_loss +
            delta_clamped * compression_loss +
            epsilon_clamped * entropy_loss  # NEW: Sequence diversity term
        )

        # Add intrinsic signal floor (brain never has zero curiosity)
        total_loss = torch.clamp(total_loss, 0.01, 100.0)

        # Apply task-aware modulation
        # This couples intrinsic learning to task performance
        # When task is improving, reduce intrinsic exploration
        # When task is stuck/high loss, increase intrinsic exploration
        total_loss = total_loss * task_awareness_weight

        # Apply neuromodulation (adaptive intrinsic learning strength)
        if self.neuromodulator is not None and prediction_logits is not None and targets is not None:
            # Compute prediction loss for neuromodulator
            pred_loss = F.cross_entropy(
                prediction_logits.view(-1, prediction_logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

            # Compute surprise for neuromodulator (handle -100 ignore index)
            log_probs = F.log_softmax(prediction_logits, dim=-1)
            targets_safe = targets.clone()
            targets_safe[targets == -100] = 0  # Replace -100 temporarily
            surprise = -log_probs.gather(dim=-1, index=targets_safe.unsqueeze(-1)).squeeze(-1)
            surprise[targets == -100] = 0.0  # Mask out ignored positions

            # Get modulation signal
            modulation = self.neuromodulator(
                intrinsic_loss=total_loss,
                prediction_loss=pred_loss,
                surprise=surprise,
            )

            # Apply modulation
            modulated_loss = total_loss * modulation

            # Store modulation statistics
            neuro_stats = self.neuromodulator.get_statistics()
            component_losses.update({
                f"neuro_{k}": v for k, v in neuro_stats.items()
            })

            total_loss = modulated_loss

        # Store weights for monitoring
        component_losses["alpha"] = self.alpha.detach()
        component_losses["beta"] = self.beta.detach()
        component_losses["gamma"] = self.gamma.detach()
        component_losses["delta"] = self.delta.detach()
        component_losses["epsilon"] = self.epsilon.detach()

        return total_loss, component_losses

    def surprise_weighted_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        current_hidden: torch.Tensor,
        previous_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Surprise-weighted prediction error: L_surprise = Σ_t -log p(x_t) · ||h_t - f(h_{t-1})||²

        Focus learning on unexpected inputs.
        """
        # Compute prediction error (surprise)
        log_probs = F.log_softmax(logits, dim=-1)
        surprise = F.nll_loss(
            log_probs.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(targets.shape)  # (batch, seq_len)

        # Compute hidden state prediction error
        if previous_hidden is not None:
            # Handle batch size mismatch between current and previous hidden
            if previous_hidden.shape[0] != current_hidden.shape[0]:
                # Use only the minimum batch size to avoid broadcasting errors
                min_batch = min(previous_hidden.shape[0], current_hidden.shape[0])
                previous_hidden = previous_hidden[:min_batch]
                current_hidden_subset = current_hidden[:min_batch]

                # Predict current hidden from previous
                predicted_hidden = self.temporal_predictor(previous_hidden)
                prediction_error_subset = F.mse_loss(
                    current_hidden_subset,
                    predicted_hidden,
                    reduction="none",
                ).mean(dim=-1)  # (min_batch, seq_len)

                # Pad to match current batch size
                prediction_error = torch.zeros(current_hidden.shape[:2], device=current_hidden.device)
                prediction_error[:min_batch] = prediction_error_subset
            else:
                # Predict current hidden from previous
                predicted_hidden = self.temporal_predictor(previous_hidden)
                prediction_error = F.mse_loss(
                    current_hidden,
                    predicted_hidden,
                    reduction="none",
                ).mean(dim=-1)  # (batch, seq_len)
        else:
            prediction_error = torch.zeros_like(surprise)

        # Weight prediction error by surprise
        weighted_loss = (surprise * prediction_error).mean()

        return weighted_loss

    def temporal_consistency_loss(
        self,
        current_hidden: torch.Tensor,
        previous_hidden: torch.Tensor,
        same_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Temporal consistency: L_temporal = ||h_t - sg(h_{t-1})||² · I[same_context]

        Encourage stable representations within contexts.
        """
        # Stop gradient on previous hidden (don't propagate through time too far)
        previous_hidden_sg = previous_hidden.detach()

        # Handle batch size mismatch
        if previous_hidden_sg.shape[0] != current_hidden.shape[0]:
            min_batch = min(previous_hidden_sg.shape[0], current_hidden.shape[0])
            previous_hidden_sg = previous_hidden_sg[:min_batch]
            current_hidden_subset = current_hidden[:min_batch]

            # Compute consistency loss for matching portion
            consistency_loss_subset = F.mse_loss(
                current_hidden_subset,
                previous_hidden_sg,
                reduction="none",
            ).mean(dim=-1)  # (min_batch, seq_len)

            # Pad to match current batch size with zeros
            consistency_loss = torch.zeros(current_hidden.shape[:2], device=current_hidden.device)
            consistency_loss[:min_batch] = consistency_loss_subset
        else:
            # Compute consistency loss
            consistency_loss = F.mse_loss(
                current_hidden,
                previous_hidden_sg,
                reduction="none",
            ).mean(dim=-1)  # (batch, seq_len)

        # Apply context mask if provided
        if same_context is not None:
            consistency_loss = consistency_loss * same_context

        return consistency_loss.mean()

    def expert_disagreement_loss(
        self,
        expert_outputs: list,
    ) -> torch.Tensor:
        """
        Cross-expert disagreement: L_disagreement = Var_e[f_e(x)] · I[high_uncertainty]

        Use internal cognitive conflict as learning signal.
        """
        # Stack expert outputs (num_experts, batch, seq_len, hidden_size)
        expert_stack = torch.stack(expert_outputs, dim=0)

        # Compute variance across experts
        disagreement = expert_stack.var(dim=0).mean()

        # Weight by uncertainty (higher disagreement = more uncertain)
        # We treat disagreement itself as the uncertainty signal
        return disagreement

    def compression_progress_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compression progress: L_compression = ΔL_pred / Δt

        Reward improvement in prediction ability.
        """
        # Compute current prediction loss
        current_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        # Store in history
        self.loss_history[self.loss_idx % 100] = current_loss.detach()
        self.loss_idx += 1

        # Compute compression progress (negative derivative of loss)
        if self.loss_idx >= 10:
            recent_window = min(10, self.loss_idx)
            start_idx = max(0, (self.loss_idx - recent_window) % 100)
            end_idx = self.loss_idx % 100

            if end_idx > start_idx:
                recent_losses = self.loss_history[start_idx:end_idx]
            else:
                # Handle wrap-around
                recent_losses = torch.cat([
                    self.loss_history[start_idx:],
                    self.loss_history[:end_idx]
                ])

            if len(recent_losses) > 1:
                # Simple derivative: (old_loss - new_loss) / time
                # Positive = improvement = reward
                improvement = recent_losses[0] - recent_losses[-1]
                compression_reward = improvement / len(recent_losses)

                # Clamp to prevent extreme values
                compression_reward = torch.clamp(compression_reward, -10.0, 10.0)
            else:
                compression_reward = torch.tensor(0.0, device=logits.device)
        else:
            compression_reward = torch.tensor(0.0, device=logits.device)

        # Return as a loss (positive value, not reward)
        # We want to minimize, so return magnitude of negative progress
        return torch.abs(compression_reward)

    def sequence_entropy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequence-level entropy loss: penalize low token diversity over rolling windows.

        CRITICAL FOR LOCAL LEARNING MODELS which lack global sequence control.

        This prevents the model from:
        - Repeating the same token (e.g., "ict ict ict...")
        - Getting stuck in limit cycles
        - Collapsing to high-frequency tokens

        Args:
            logits: Model predictions (batch, seq_len, vocab_size)
            targets: Target tokens (batch, seq_len)

        Returns:
            Entropy penalty (higher when diversity is low)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Skip if sequence too short
        if seq_len < self.entropy_window:
            return torch.tensor(0.0, device=logits.device)

        # Get predicted token distribution at each position
        probs = F.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

        # Sample from distribution to get predicted tokens
        # (In training, we use greedy for efficiency)
        pred_tokens = logits.argmax(dim=-1)  # (batch, seq_len)

        total_penalty = 0.0
        num_windows = 0

        # Slide a window across the sequence
        for start_idx in range(0, seq_len - self.entropy_window + 1, self.entropy_window // 2):
            end_idx = start_idx + self.entropy_window
            window_tokens = pred_tokens[:, start_idx:end_idx]  # (batch, window)

            # Compute token distribution entropy within this window
            for b in range(batch_size):
                tokens_in_window = window_tokens[b].tolist()

                # Count token frequencies
                token_counts = {}
                for token in tokens_in_window:
                    token_counts[token] = token_counts.get(token, 0) + 1

                # Compute empirical distribution
                total_count = len(tokens_in_window)
                token_probs_list = [count / total_count for count in token_counts.values()]

                # Compute entropy (in bits)
                if len(token_probs_list) > 0:
                    token_probs_tensor = torch.tensor(token_probs_list, device=logits.device)
                    window_entropy = -(token_probs_tensor * torch.log2(token_probs_tensor + 1e-10)).sum()

                    # Penalty = max(0, threshold - actual_entropy)
                    # Higher penalty when entropy is low (repetitive)
                    entropy_deficit = torch.clamp(
                        self.entropy_min_threshold - window_entropy,
                        min=0.0,
                        max=10.0
                    )

                    total_penalty += entropy_deficit
                    num_windows += 1

        # Average across all windows
        if num_windows > 0:
            avg_penalty = total_penalty / num_windows
        else:
            avg_penalty = torch.tensor(0.0, device=logits.device)

        return avg_penalty

    def get_component_weights(self) -> Dict[str, float]:
        """Get current values of loss component weights."""
        return {
            "alpha": self.alpha.item(),
            "beta": self.beta.item(),
            "gamma": self.gamma.item(),
            "delta": self.delta.item(),
            "epsilon": self.epsilon.item(),
        }
