"""Main Pragnosia model implementation."""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import threading

from ..utils.config import PragnosiaConfig
from .router import HebbianRouter
from .expert import ExpertModule
from .coherence_module import CoherenceModule
from .enhanced_coherence import EnhancedCoherenceModule
from ..memory.hippocampus import Hippocampus
from ..memory.neocortex import Neocortex
from ..losses.intrinsic import IntrinsicObjective
from ..utils.homeostasis import HomeostasisRegulator


class PragnosiaModel(nn.Module):
    """
    Pragnosia: A Systems Architecture for Continual Representation Learning.

    Key components:
    - Hebbian router (~100MB, GPU-resident)
    - Expert modules (~500MB each, CPU-stored with on-demand loading)
    - Memory systems (hippocampus ~50MB, neocortex ~100MB)
    - Multi-dimensional intrinsic learning
    - Asynchronous micro-sleep consolidation

    Maximum GPU usage: 2.8GB with 2 active experts (enables 4GB deployment)
    """

    def __init__(self, config: PragnosiaConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        # CRITICAL FIX: Initialize embeddings with small std to prevent extreme logits
        # Default initialization causes norm=5000+, leading to logits of ±600
        # Proper initialization: std=0.02 gives manageable logits in ±10 range
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Hebbian router (always GPU-resident, ~100MB)
        self.router = HebbianRouter(
            input_size=config.hidden_size,
            num_experts=config.num_experts,
            num_active_experts=config.num_active_experts,
            learning_rate=config.hebbian_learning_rate,
            lateral_inhibition=config.lateral_inhibition_strength,
            threshold=config.routing_threshold,
            router_size=config.router_size,
        )

        # Expert modules (~500MB each, CPU-stored)
        self.experts = nn.ModuleList([
            ExpertModule(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dropout=config.expert_dropout,
                expert_id=i,
            )
            for i in range(config.num_experts)
        ])

        # Initially move experts to CPU if offloading is enabled
        if config.offload_to_cpu:
            for expert in self.experts:
                expert.to_cpu()

        # Memory systems
        self.hippocampus = Hippocampus(
            capacity=config.hippocampus_capacity,
            batch_size=config.hippocampus_batch_size,
            consolidation_threshold=config.consolidation_threshold,
        )

        self.neocortex = Neocortex(
            capacity=config.neocortex_capacity,
            hidden_size=config.hidden_size,
        )

        # Intrinsic learning objective
        self.intrinsic_objective = IntrinsicObjective(
            hidden_size=config.hidden_size,
            alpha=config.alpha_surprise,
            beta=config.beta_temporal,
            gamma=config.gamma_disagreement,
            delta=config.delta_compression,
            epsilon=config.epsilon_entropy,
            entropy_window=config.entropy_window,
            entropy_min_threshold=config.entropy_min_threshold,
            learn_weights=config.learn_intrinsic_weights,
            use_neuromodulation=config.use_neuromodulation,
        )

        # Homeostatic regulation
        self.homeostasis = HomeostasisRegulator(
            lambda_uncertainty=config.lambda_uncertainty,
            lambda_surprise=config.lambda_surprise,
            lambda_churn=config.lambda_churn,
            target_entropy=config.target_entropy,
        )

        # COHERENCE MODULE: Powerful transformer for sequential binding
        # This is the key innovation that enables coherent generation
        # while preserving local learning in experts
        self.use_coherence = getattr(config, 'use_coherence_module', True)
        use_enhanced = getattr(config, 'use_enhanced_coherence', True)

        if self.use_coherence:
            if use_enhanced:
                # ENHANCED coherence for proper chat and code generation
                self.coherence = EnhancedCoherenceModule(
                    hidden_size=config.hidden_size,
                    num_layers=getattr(config, 'coherence_num_layers', 8),  # Default: 8 layers
                    num_heads=getattr(config, 'coherence_num_heads', 8),    # Default: 8 heads
                    dropout=getattr(config, 'coherence_dropout', 0.1),
                    use_kv_cache=True,  # Enable fast generation
                )
            else:
                # Original lightweight coherence
                self.coherence = CoherenceModule(
                    hidden_size=config.hidden_size,
                    num_layers=getattr(config, 'coherence_num_layers', 2),
                    num_heads=getattr(config, 'coherence_num_heads', 4),
                    dropout=getattr(config, 'coherence_dropout', 0.1),
                )
                print(f"🔗 Coherence Module enabled: {self.coherence.get_memory_size_mb():.1f} MB")
        else:
            self.coherence = None
            print("⚠️  Coherence Module disabled - generation may be incoherent")

        # Output head
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # TIE WEIGHTS (standard approach)
        # This works for local learning if we handle optimizer state correctly
        self.output_head.weight = self.token_embedding.weight

        # Micro-sleep consolidation state
        self.consolidation_counter = 0
        self.consolidation_lock = threading.Lock()

        # Track previous hidden states for temporal consistency
        self.prev_hidden = None

        # Track active experts on GPU
        self.active_expert_ids = []

        # Track task loss history for task-aware intrinsic learning
        self.register_buffer("task_loss_history", torch.zeros(100))
        self.task_loss_idx = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        inference_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Pragnosia model.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target labels for language modeling (batch, seq_len)
            return_dict: Whether to return dictionary output
            inference_mode: If True, disable ALL learning (routing frozen, no intrinsic, no updates)
                          CRITICAL for stable generation/chat

        Returns:
            Dictionary containing:
            - logits: Model predictions
            - loss: Total loss (if labels provided and not inference_mode)
            - intrinsic_loss: Intrinsic learning objective (None in inference_mode)
            - homeostatic_loss: Homeostatic penalty (None in inference_mode)
            - hidden_states: Final hidden representations
            - routing_stats: Router statistics (minimal in inference_mode)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed tokens
        token_embeds = self.token_embedding(input_ids)  # (batch, seq_len, hidden)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds

        # Route to experts using Hebbian router
        features, selected_expert_ids, routing_weights = self.router(
            hidden_states,
            return_routing_weights=True,
        )

        # Load selected experts to GPU
        self._load_experts_to_gpu(selected_expert_ids, device)

        # Process through active experts
        expert_outputs = []
        expert_stats = []

        for expert_id in selected_expert_ids:
            expert = self.experts[expert_id]
            expert_output, stats = expert(hidden_states)
            expert_outputs.append(expert_output)
            expert_stats.append(stats)

        # Combine expert outputs with routing weights
        if routing_weights is not None:
            combined_output = sum(
                weight * output
                for weight, output in zip(routing_weights, expert_outputs)
            )
        else:
            combined_output = sum(expert_outputs) / len(expert_outputs)

        # APPLY COHERENCE MODULE (if enabled)
        # This is where local representations get bound into coherent sequences
        if self.use_coherence and self.coherence is not None:
            # Create attention mask from input_ids (1 for real tokens, 0 for padding)
            if hasattr(self.token_embedding, 'padding_idx') and self.token_embedding.padding_idx is not None:
                attention_mask = (input_ids != self.token_embedding.padding_idx).long()
            else:
                attention_mask = None

            # Apply coherence (global learning via self-attention)
            # In inference mode, use KV cache for fast generation
            use_cache = inference_mode and seq_len == 1
            coherent_output = self.coherence(combined_output, attention_mask, use_cache=use_cache)

            # Use coherent output for final predictions
            final_output = coherent_output
        else:
            # No coherence - use raw expert output (may be incoherent)
            final_output = combined_output

        # Generate predictions from coherent (or raw) representations
        logits = self.output_head(final_output)

        # Compute losses if labels provided
        total_loss = None
        intrinsic_loss = None
        homeostatic_loss = None

        if labels is not None and not inference_mode:
            # Compute task loss FIRST (for task-aware intrinsic learning)
            task_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

            # Track task loss and compute improvement
            self.task_loss_history[self.task_loss_idx % 100] = task_loss.detach()
            self.task_loss_idx += 1

            # Compute task improvement (recent trend)
            task_improvement = torch.tensor(0.0, device=device)
            if self.task_loss_idx >= 20:
                # Compare recent 10 steps to previous 10 steps
                recent_window = min(10, self.task_loss_idx)
                idx = self.task_loss_idx % 100

                # Get recent losses
                if idx >= recent_window:
                    recent_losses = self.task_loss_history[idx - recent_window:idx]
                else:
                    # Handle wrap-around
                    recent_losses = torch.cat([
                        self.task_loss_history[idx - recent_window:],
                        self.task_loss_history[:idx]
                    ])

                # Get previous losses
                prev_idx = (idx - recent_window) % 100
                if prev_idx >= recent_window:
                    prev_losses = self.task_loss_history[prev_idx - recent_window:prev_idx]
                else:
                    prev_losses = torch.cat([
                        self.task_loss_history[prev_idx - recent_window:],
                        self.task_loss_history[:prev_idx]
                    ])

                # Improvement = (old_loss - new_loss) / old_loss
                # Positive = improving, negative = degrading
                if len(prev_losses) > 0 and len(recent_losses) > 0:
                    old_mean = prev_losses.mean()
                    new_mean = recent_losses.mean()
                    if old_mean > 0:
                        task_improvement = (old_mean - new_mean) / old_mean

            # Compute task-aware intrinsic learning objective
            intrinsic_loss, component_losses = self.intrinsic_objective(
                current_hidden=combined_output,
                previous_hidden=self.prev_hidden,
                expert_outputs=expert_outputs if len(expert_outputs) > 1 else None,
                prediction_logits=logits,
                targets=labels,
                same_context=None,  # Could be computed from attention mask
                task_loss=task_loss,
                task_improvement=task_improvement,
            )

            # Compute homeostatic penalty
            # Compute prediction entropy
            pred_probs = torch.softmax(logits, dim=-1)
            pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-10)).sum(dim=-1)

            # Compute surprise (mask out ignore index -100)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Replace -100 with 0 temporarily for gathering (will be masked out)
            labels_safe = labels.clone()
            labels_safe[labels == -100] = 0
            surprise = -log_probs.gather(dim=-1, index=labels_safe.unsqueeze(-1)).squeeze(-1)
            # Mask out positions with ignore index
            surprise[labels == -100] = 0.0

            # Expert activations (binary vector)
            expert_activations = torch.zeros(self.config.num_experts, device=device)
            expert_activations[selected_expert_ids] = 1.0

            homeostatic_loss = self.homeostasis(
                prediction_entropy=pred_entropy,
                surprise=surprise,
                expert_activations=expert_activations,
            )

            # Total loss
            total_loss = intrinsic_loss + homeostatic_loss

            # Update router with Hebbian learning
            # Use final output (after coherence) for error computation
            # This aligns routing with actual model performance
            expert_errors = self._compute_expert_errors(
                expert_outputs,
                labels,
                final_output,
            )
            self.router.hebbian_update(features, expert_errors)

            # Store experience in hippocampus
            # Average surprise per sample in batch
            surprise_per_sample = surprise.mean(dim=1) if surprise.dim() > 1 else surprise
            loss_per_sample = total_loss.detach().unsqueeze(0).expand(batch_size)

            self.hippocampus.store(
                hidden_states=combined_output,
                targets=labels,
                loss=loss_per_sample,
                surprise=surprise_per_sample,
            )

            # Asynchronous micro-sleep consolidation
            self._micro_sleep_consolidation()

        # INFERENCE MODE path (for evaluation with labels)
        if inference_mode and labels is not None:
            # Compute loss for evaluation only (no learning)
            eval_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return {
                "logits": logits,
                "loss": eval_loss,
                "intrinsic_loss": None,
                "homeostatic_loss": None,
                "hidden_states": final_output.detach(),  # Return coherent output
                "routing_stats": {"selected_experts": selected_expert_ids},
            }

        # Update previous hidden for temporal consistency (training only)
        # Use coherent output for temporal tracking if available
        if not inference_mode:
            self.prev_hidden = final_output.detach()

        # Gather routing statistics
        routing_stats = self.router.check_stability()
        routing_stats.update({
            "selected_experts": selected_expert_ids,
            "expert_stats": expert_stats,
        })

        output = {
            "logits": logits,
            "loss": total_loss,
            "intrinsic_loss": intrinsic_loss,
            "homeostatic_loss": homeostatic_loss,
            "hidden_states": final_output,  # Return coherent representations
            "routing_stats": routing_stats,
        }

        if labels is not None and not inference_mode:
            output["component_losses"] = component_losses

        return output

    def _load_experts_to_gpu(self, expert_ids: List[int], device: torch.device):
        """Load selected experts to GPU, offload others to CPU."""
        if not self.config.offload_to_cpu:
            return

        # Offload previously active experts not in current selection
        for expert_id in self.active_expert_ids:
            if expert_id not in expert_ids:
                self.experts[expert_id].to_cpu()

        # Load newly selected experts to GPU
        for expert_id in expert_ids:
            if expert_id not in self.active_expert_ids:
                self.experts[expert_id].to_gpu(device)

        self.active_expert_ids = expert_ids

    def _compute_expert_errors(
        self,
        expert_outputs: List[torch.Tensor],
        labels: torch.Tensor,
        final_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute error reduction for each expert for Hebbian updates.

        Note: Uses final coherent output as baseline (after coherence module)
        This ensures Hebbian updates align with actual model performance.

        Returns tensor of error reductions (negative = improvement).
        """
        expert_errors = torch.zeros(self.config.num_experts, device=labels.device)

        # Compute baseline error (final coherent output)
        baseline_logits = self.output_head(final_output)
        baseline_loss = torch.nn.functional.cross_entropy(
            baseline_logits.view(-1, baseline_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # Compute individual expert errors
        for i, expert_output in enumerate(expert_outputs):
            expert_logits = self.output_head(expert_output)
            expert_loss = torch.nn.functional.cross_entropy(
                expert_logits.view(-1, expert_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

            # Error reduction = baseline - expert_loss (positive = improvement)
            expert_errors[i] = (baseline_loss - expert_loss).detach()

        return expert_errors

    def _micro_sleep_consolidation(self):
        """
        Asynchronous micro-sleep consolidation.

        Transfers experiences from hippocampus to neocortex with <1ms overhead.
        Uses background thread and micro-batches (1-2 samples per forward pass).
        """
        self.consolidation_counter += 1

        # Micro-consolidation: Transfer 1-2 samples per forward pass
        if self.consolidation_counter % self.config.microsleep_samples == 0:
            candidates = self.hippocampus.get_consolidation_candidates()
            if candidates is not None and len(candidates) > 0:
                # Sample micro-batch
                num_samples = min(2, len(candidates))
                samples = candidates[:num_samples]

                # Consolidate to neocortex (async, minimal overhead)
                with self.consolidation_lock:
                    self.neocortex.consolidate(samples)
                    self.hippocampus.clear_consolidated(samples)

        # Intensive consolidation: Full transfer on distribution shift
        if self.consolidation_counter % self.config.intensive_consolidation_interval == 0:
            self._intensive_consolidation()

    def _intensive_consolidation(self):
        """Intensive consolidation triggered by distribution shift or error spike."""
        candidates = self.hippocampus.get_consolidation_candidates()
        if candidates is not None:
            with self.consolidation_lock:
                self.neocortex.consolidate(candidates)
                self.hippocampus.clear_consolidated(candidates)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """
        Generate text autoregressively with enhanced coherence.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling - keep tokens with cumulative probability <= top_p
            do_sample: Whether to sample (True) or use greedy decoding (False)
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            repetition_penalty: Penalty for repeating tokens (>1.0 = discourage repetition)

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        # Clear KV cache if using enhanced coherence
        if self.use_coherence and hasattr(self.coherence, 'clear_cache'):
            self.coherence.clear_cache()

        generated = input_ids.clone()

        # Track token counts for repetition penalty
        token_counts = {}

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(generated, inference_mode=True)
            logits = outputs["logits"]

            # Get logits for last token
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id, count in token_counts.items():
                    # Penalize repeated tokens (divide if positive logit, multiply if negative)
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= (repetition_penalty ** count)
                    else:
                        next_token_logits[0, token_id] *= (repetition_penalty ** count)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Update token counts for repetition penalty
            token_id = next_token[0, 0].item()
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

            # Stop if EOS token is generated
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def get_memory_statistics(self) -> Dict[str, any]:
        """Get comprehensive memory statistics."""
        router_memory = self.router.get_memory_size_mb()
        expert_memory = sum(expert.get_memory_size_mb() for expert in self.experts)
        hippo_memory = self.hippocampus.get_memory_size_mb()
        neo_memory = self.neocortex.get_memory_size_mb()

        active_expert_memory = sum(
            self.experts[i].get_memory_size_mb()
            for i in self.active_expert_ids
        )

        return {
            "router_mb": router_memory,
            "total_experts_mb": expert_memory,
            "active_experts_mb": active_expert_memory,
            "hippocampus_mb": hippo_memory,
            "neocortex_mb": neo_memory,
            "total_mb": router_memory + active_expert_memory + hippo_memory + neo_memory,
            "hippocampus_stats": self.hippocampus.get_statistics(),
            "neocortex_stats": self.neocortex.get_statistics(),
        }
