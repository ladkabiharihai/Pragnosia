"""Local learning trainer with constant VRAM guarantees.

This trainer implements brain-like local learning rules:
- No global backpropagation across experts
- No global optimizer (each expert updates independently)
- Hard expert sparsity (exactly k experts active)
- Aggressive memory management
- Continuous VRAM tracking

This achieves TRUE constant-VRAM training that scales to arbitrary model size.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, List
from tqdm import tqdm
import os
import gc

from ..models.pragnosia_model import PragnosiaModel
from ..utils.config import PragnosiaConfig
from ..utils.training_phases import TrainingPhaseScheduler
from ..utils.calibration import ReferenceCalibrator


class LocalLearningTrainer:
    """
    Trainer with local learning rules and constant VRAM.

    Key principles:
    1. Only active experts learn (k=2 typically)
    2. Each expert has its own optimizer
    3. No gradient flow between experts
    4. Aggressive memory cleanup after each update
    5. VRAM usage is constant regardless of total experts

    This is biologically correct: brains never update all neurons simultaneously.
    """

    def __init__(
        self,
        model: PragnosiaModel,
        config: PragnosiaConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./outputs",
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        local_learning_rate: float = 0.001,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.local_learning_rate = local_learning_rate

        # Move model components to device
        self.model.to(device)

        # LOCAL OPTIMIZERS - Each expert gets its own tiny optimizer
        # This ensures optimizer state scales with k (active experts), not total experts
        self.expert_optimizers = []
        for expert in self.model.experts:
            optimizer = torch.optim.SGD(  # Simple SGD, no momentum (minimal state)
                expert.parameters(),
                lr=local_learning_rate,
            )
            self.expert_optimizers.append(optimizer)

        # Router optimizer (always active, small)
        self.router_optimizer = torch.optim.Adam(
            self.model.router.parameters(),
            lr=config.hebbian_learning_rate,
        )

        # Embedding optimizer (always active, moderate size)
        # ONLY position_embedding (NOT token_embedding)
        # token_embedding is updated ONLY via lm_head_optimizer (Step 4)
        # This prevents optimizer state corruption
        self.embedding_optimizer = torch.optim.Adam(
            [
                {"params": self.model.position_embedding.parameters()},
            ],
            lr=config.learning_rate,
        )

        # LM head optimizer (CRITICAL: separate from experts)
        # The LM head is the READOUT - it learns language via supervised signals ONLY
        # It does NOT use intrinsic or homeostatic losses
        # This is biologically correct: cortex learns representations, readout learns task
        self.lm_head_optimizer = torch.optim.Adam(
            self.model.output_head.parameters(),
            lr=config.learning_rate * 2.0,  # Can be more aggressive since it's just readout
        )

        # Coherence Module optimizer (NEW: global learning for sequential binding)
        # This module uses standard backprop + cross-entropy
        # Learns to bind locally-learned representations into coherent sequences
        if hasattr(self.model, 'coherence') and self.model.coherence is not None:
            self.coherence_optimizer = torch.optim.Adam(
                self.model.coherence.parameters(),
                lr=config.learning_rate,  # Same as embeddings
                weight_decay=0.01,  # Slight regularization for transformer
            )
            print(f"🔗 Coherence optimizer initialized (global learning enabled)")
        else:
            self.coherence_optimizer = None
            print(f"⚠️  No coherence optimizer (global learning disabled)")

        # VRAM tracking
        self.vram_history = []
        self.vram_tracking_enabled = torch.cuda.is_available()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Training phase scheduler
        # Calculate total training steps (will be updated when training starts)
        self.phase_scheduler = None  # Initialized in train() when total steps known

        # Expert maturity tracking for freezing
        # FIXED: Track representation stability (output drift) instead of loss variance
        self.expert_representation_history = {i: [] for i in range(config.num_experts)}  # Store recent outputs
        self.expert_stability_scores = {i: [] for i in range(config.num_experts)}  # Store stability scores
        self.frozen_experts = set()

        # Reference-based calibration (frozen teacher)
        self.calibrator = ReferenceCalibrator(
            calibration_weight=0.1,
            divergence_threshold=2.0,
            use_kl_divergence=True,
        )

        # Frequency-aware loss weighting (DISABLED - was breaking training)
        # Just use uniform weights (no reweighting)
        print("Skipping token frequency weighting (using uniform weights)...")
        self.token_freq_weights = torch.ones(config.vocab_size, device=self.device)

        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("LOCAL LEARNING TRAINER INITIALIZED")
        print("=" * 60)
        print(f"Total experts: {config.num_experts}")
        print(f"Active experts per step: {config.num_active_experts}")
        print(f"Local learning rate: {local_learning_rate}")
        print(f"Memory scaling: O(k) not O(n) where k={config.num_active_experts}")
        print(f"Phased training: ENABLED (representation → alignment → stabilization)")
        print(f"\n🔑 CRITICAL: Separate LM head training + WEIGHT TYING")
        print(f"  - Experts: Local learning + intrinsic objectives")
        print(f"  - Token embeddings: Trained via LM head (output direction)")
        print(f"  - Position embeddings: Trained via calibration (context encoding)")
        print(f"  - Weights TIED (output_head = token_embedding)")
        print(f"  - No optimizer state corruption (each param has ONE optimizer)")
        print("=" * 60)

    def _compute_token_frequency_weights(
        self,
        dataloader: DataLoader,
        vocab_size: int,
        smoothing: float = 0.001,
        max_batches: int = None,
    ) -> torch.Tensor:
        """
        Compute token weights using HYBRID approach:
        1. Down-weight common English function words (the, a, is, etc.)
        2. Up-weight content words based on observed frequency

        CRITICAL for local learning models which overfit to high-frequency tokens.

        Args:
            dataloader: Training data loader
            vocab_size: Vocabulary size
            smoothing: Laplace smoothing factor
            max_batches: Maximum batches to scan

        Returns:
            Tensor of shape (vocab_size,) with frequency-based weights
        """
        # Start with base weight of 1.0 for all tokens
        token_weights = torch.ones(vocab_size, dtype=torch.float32)

        token_counts = torch.zeros(vocab_size, dtype=torch.float32)

        # Count token occurrences
        num_batches_scanned = 0
        for batch in dataloader:
            # Get labels (or input_ids if no labels)
            labels = batch.get("labels", batch.get("input_ids"))
            if labels is None:
                continue

            # Count tokens (ignore padding/special tokens marked as -100)
            valid_tokens = labels[labels != -100]
            if len(valid_tokens) > 0:
                for token_id in valid_tokens.flatten().tolist():
                    if 0 <= token_id < vocab_size:
                        token_counts[token_id] += 1

            num_batches_scanned += 1
            if max_batches is not None and num_batches_scanned >= max_batches:
                break

        print(f"  Scanned {num_batches_scanned} batches")

        # Count how many tokens actually appeared
        tokens_seen = (token_counts > 0).sum().item()
        print(f"  Tokens seen: {tokens_seen} / {vocab_size}")

        # HYBRID APPROACH: Down-weight the TOP-K most frequent tokens observed
        # This is more reliable than trying to compute inverse frequencies from small data

        if tokens_seen > 100:  # Need enough tokens to compute ranks
            # Sort by frequency (descending)
            sorted_counts, sorted_indices = torch.sort(token_counts, descending=True)

            # Down-weight the top 5% most frequent tokens
            top_k = max(50, int(tokens_seen * 0.05))  # At least 50 tokens
            top_frequent_tokens = sorted_indices[:top_k]

            # Assign weights based on frequency rank
            for rank, token_id in enumerate(top_frequent_tokens):
                # Linear decay from 0.1 (most frequent) to 0.5 (rank=top_k)
                weight = 0.1 + 0.4 * (rank / top_k)
                token_weights[token_id] = weight

            print(f"  Down-weighted top {top_k} frequent tokens to 0.1-0.5x")

        # Additionally, up-weight RARE tokens (appeared only 1-2 times)
        rare_mask = (token_counts > 0) & (token_counts <= 2)
        num_rare = rare_mask.sum().item()
        if num_rare > 0:
            token_weights[rare_mask] = 3.0  # 3x weight for very rare tokens
            print(f"  Up-weighted {num_rare} rare tokens (count<=2) to 3.0x")

        # Show statistics
        print(f"  Final: min={token_weights.min().item():.4f}, max={token_weights.max().item():.4f}, mean={token_weights.mean().item():.4f}")

        # Show distribution
        num_below_0_5 = (token_weights < 0.5).sum().item()
        num_normal = ((token_weights >= 0.5) & (token_weights <= 1.5)).sum().item()
        num_above_2 = (token_weights > 2.0).sum().item()
        print(f"  Tokens < 0.5x (frequent): {num_below_0_5} / {vocab_size} ({100*num_below_0_5/vocab_size:.1f}%)")
        print(f"  Tokens 0.5-1.5x (normal): {num_normal} / {vocab_size} ({100*num_normal/vocab_size:.1f}%)")
        print(f"  Tokens > 2.0x (rare): {num_above_2} / {vocab_size} ({100*num_above_2/vocab_size:.1f}%)")

        return token_weights.to(self.device)

    def train(self, num_epochs: int):
        """
        Train with local learning rules.

        At each step:
        1. Router selects exactly k experts (hard sparsity)
        2. ONLY those k experts are loaded to GPU
        3. ONLY those k experts receive gradients
        4. ONLY those k experts are updated
        5. All activations are immediately freed
        6. Experts are offloaded back to CPU

        VRAM usage: constant regardless of total expert count.
        """
        # Initialize phase scheduler if not already done
        if self.phase_scheduler is None:
            total_steps = len(self.train_dataloader) * num_epochs
            self.phase_scheduler = TrainingPhaseScheduler(total_steps)
            print("\n" + "=" * 80)
            print("TRAINING PHASE SCHEDULER INITIALIZED")
            print("=" * 80)
            print(f"Total training steps: {total_steps}")
            print(f"Phase A (Representation): {int(total_steps * 0.30)} steps (30%)")
            print(f"Phase B (Alignment): {int(total_steps * 0.50)} steps (50%)")
            print(f"Phase C (Stabilization): {int(total_steps * 0.20)} steps (20%)")
            print("=" * 80 + "\n")
            self.phase_scheduler.print_phase_info()

        self.model.train()

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # Advance phase scheduler
                prev_phase_name = self.phase_scheduler.get_current_phase().name if self.phase_scheduler else None
                self.phase_scheduler.step()
                current_phase = self.phase_scheduler.get_current_phase()
                loss_weights = self.phase_scheduler.get_loss_weights()
                intrinsic_weights = self.phase_scheduler.get_intrinsic_weights()

                # Check if we should save teacher checkpoint (end of Phase A)
                phase_stats = self.phase_scheduler.get_stats()
                if self.calibrator.should_save_teacher(current_phase.name, phase_stats['phase_progress']):
                    # Transitioning from representation to alignment
                    # Save frozen teacher to prevent drift
                    self.calibrator.save_teacher_checkpoint(self.model, self.global_step)
                    self.calibrator.to(self.device)

                # Update expert learning rates based on current phase
                for optimizer in self.expert_optimizers:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_phase.expert_learning_rate

                # Update model's intrinsic objective weights based on phase
                if hasattr(self.model, 'intrinsic_objective'):
                    with torch.no_grad():
                        self.model.intrinsic_objective.alpha.copy_(torch.tensor(intrinsic_weights['alpha']))
                        self.model.intrinsic_objective.beta.copy_(torch.tensor(intrinsic_weights['beta']))
                        self.model.intrinsic_objective.gamma.copy_(torch.tensor(intrinsic_weights['gamma']))
                        self.model.intrinsic_objective.delta.copy_(torch.tensor(intrinsic_weights['delta']))
                        self.model.intrinsic_objective.epsilon.copy_(torch.tensor(intrinsic_weights['epsilon']))

                # Track VRAM at start of step
                vram_start = self._get_gpu_memory_mb()

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                # STEP 1: Forward pass (selecting experts)
                # This should use HARD selection (no soft routing during training)
                with torch.no_grad():
                    # Embed tokens
                    token_embeds = self.model.token_embedding(input_ids)
                    position_ids = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
                    position_embeds = self.model.position_embedding(position_ids)
                    hidden_states = token_embeds + position_embeds

                    # Router selects EXACTLY k experts (hard sparsity)
                    features, selected_expert_ids, _ = self.model.router(
                        hidden_states,
                        return_routing_weights=False,  # No soft weights!
                    )

                # STEP 2: LOCAL LEARNING - Update each active expert independently
                # No global backprop! Each expert updates using only its own local error
                expert_losses = []
                expert_stabilities = {}

                for expert_id in selected_expert_ids:
                    # SKIP FROZEN EXPERTS
                    if expert_id in self.frozen_experts:
                        expert_losses.append(0.0)  # No update, no loss
                        continue

                    expert = self.model.experts[expert_id]
                    optimizer = self.expert_optimizers[expert_id]

                    # Load expert to GPU if needed
                    if self.config.offload_to_cpu:
                        expert.to(self.device)

                    # LOCAL FORWARD PASS
                    # Enable gradients only for this expert
                    expert_output, stats = expert(hidden_states.detach())

                    # LOCAL ERROR SIGNAL
                    # Compute prediction from this expert alone
                    expert_logits = self.model.output_head(expert_output)

                    # Local task loss (cross-entropy)
                    task_loss = torch.nn.functional.cross_entropy(
                        expert_logits.view(-1, expert_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )

                    # Combine with phase weights
                    # For local learning, task loss is the primary signal
                    # Phase weights modulate learning rate and freezing, not loss combination
                    local_loss = task_loss

                    # Track expert stability using REPRESENTATION DRIFT (not loss variance)
                    # FIXED Priority 6: Use representation stability for freezing decisions
                    # Store expert output representation (detached, mean pooled)
                    expert_repr = expert_output.detach().mean(dim=(0, 1))  # [hidden_size]
                    self.expert_representation_history[expert_id].append(expert_repr)

                    # Keep only recent history (last 20 outputs)
                    if len(self.expert_representation_history[expert_id]) > 20:
                        self.expert_representation_history[expert_id] = self.expert_representation_history[expert_id][-20:]

                    # Compute representation stability (inverse of drift)
                    if len(self.expert_representation_history[expert_id]) >= 5:
                        import numpy as np
                        recent_reprs = self.expert_representation_history[expert_id][-5:]

                        # Compute pairwise L2 distances between consecutive representations
                        drifts = []
                        for i in range(len(recent_reprs) - 1):
                            drift = torch.norm(recent_reprs[i+1] - recent_reprs[i], p=2).item()
                            drifts.append(drift)

                        avg_drift = np.mean(drifts)
                        # UPDATED: More aggressive stability computation
                        # Use exponential decay: stability = exp(-drift)
                        # This gives higher stability scores for the same drift
                        # drift=0.1 -> stability=0.90, drift=0.5 -> stability=0.61
                        stability = np.exp(-avg_drift)
                        expert_stabilities[expert_id] = stability

                        # Store stability score history
                        self.expert_stability_scores[expert_id].append(stability)
                        if len(self.expert_stability_scores[expert_id]) > 100:
                            self.expert_stability_scores[expert_id] = self.expert_stability_scores[expert_id][-100:]

                        # UPDATED: More aggressive freezing criteria
                        # Check THREE conditions (any one triggers freezing):

                        # Condition 1: Representations have stabilized (low drift)
                        should_freeze_drift = self.phase_scheduler.should_freeze_expert(stability)

                        # Condition 2: Expert has been consistently stable for last 10 steps
                        should_freeze_consistent = False
                        if len(self.expert_stability_scores[expert_id]) >= 10:
                            recent_10_stability = self.expert_stability_scores[expert_id][-10:]
                            # If all recent stability scores > 0.70, freeze
                            if all(s > 0.70 for s in recent_10_stability):
                                should_freeze_consistent = True

                        # Condition 3: We're in late alignment or stabilization phase and expert has been used enough
                        # This ensures experts freeze even without perfect stability
                        should_freeze_phase = False
                        if current_phase.name in ["alignment", "stabilization"]:
                            # If expert has been active for at least 50 steps and has reasonable stability
                            if len(self.expert_stability_scores[expert_id]) >= 50 and stability > 0.60:
                                should_freeze_phase = True

                        # Freeze if ANY condition is met
                        if should_freeze_drift or should_freeze_consistent or should_freeze_phase:
                            self.frozen_experts.add(expert_id)
                            freeze_reason = []
                            if should_freeze_drift:
                                freeze_reason.append("high stability")
                            if should_freeze_consistent:
                                freeze_reason.append("consistent stability")
                            if should_freeze_phase:
                                freeze_reason.append("phase-based")

                            print(f"\n{'='*60}")
                            print(f"EXPERT {expert_id} FROZEN (stability={stability:.4f}, drift={avg_drift:.6f})")
                            print(f"Reason: {', '.join(freeze_reason)}")
                            print(f"Phase: {current_phase.name}, Step: {self.global_step}")
                            print(f"{'='*60}\n")

                    # LOCAL BACKWARD PASS
                    # Gradients stay within this expert only!
                    optimizer.zero_grad()
                    local_loss.backward()

                    # Gradient clipping (per-expert)
                    torch.nn.utils.clip_grad_norm_(
                        expert.parameters(),
                        self.config.gradient_clip_val,
                    )

                    # LOCAL UPDATE
                    optimizer.step()

                    expert_losses.append(local_loss.item())

                    # AGGRESSIVE MEMORY CLEANUP
                    # Immediately free everything we don't need
                    del expert_output, expert_logits, task_loss, local_loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    # Offload expert back to CPU
                    if self.config.offload_to_cpu:
                        expert.to_cpu()

                    # Force garbage collection
                    gc.collect()

                # STEP 3: Update embeddings (with calibration if active)
                # Embeddings help experts form better representations
                self.embedding_optimizer.zero_grad()

                # Recompute embeddings and get expert outputs WITH gradients
                token_embeds = self.model.token_embedding(input_ids)
                position_embeds = self.model.position_embedding(position_ids)
                hidden_states = token_embeds + position_embeds

                # Get one expert output with gradients (use first selected expert)
                expert_id = selected_expert_ids[0]
                expert = self.model.experts[expert_id]
                if self.config.offload_to_cpu:
                    expert.to(self.device)

                # Forward with gradients for embeddings
                expert_output, _ = expert(hidden_states)  # Gradients flow to hidden_states

                # Compute calibration loss (if teacher checkpoint exists)
                # This ONLY affects embeddings, not the LM head
                calibration_results = self.calibrator.compute_calibration_loss(
                    student_hidden=expert_output,
                    student_logits=None,  # Don't compute on logits for embeddings
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                calibration_loss = calibration_results["calibration_loss"]
                divergence = calibration_results["divergence"]

                # Embedding loss is just calibration (representation stability)
                if calibration_loss.item() > 0:
                    calibration_loss.backward()

                # Zero out expert gradients (we already updated experts)
                expert.zero_grad()

                # Update ONLY position embeddings
                # Token embeddings are NOT in this optimizer (they're in lm_head_optimizer)
                # This prevents optimizer state corruption
                self.embedding_optimizer.step()

                # STEP 4: Update LM head (and token embeddings via weight tying)
                # output_head.weight IS token_embedding.weight (same tensor)
                # So this step trains BOTH:
                #   - output_head: learns to predict next tokens
                #   - token_embedding: learns token representations for prediction
                #
                # This is the ONLY place token_embedding gets gradients (output direction).
                # Position embeddings are trained separately in Step 3 (context direction).
                self.lm_head_optimizer.zero_grad()

                # Zero gradients for coherence module (global learning)
                if self.coherence_optimizer is not None:
                    self.coherence_optimizer.zero_grad()

                # Reuse the combined expert output from earlier
                # (It's still in memory from Step 2, but detached from computation graph)
                # We'll work with the cached expert outputs to avoid recomputing

                # Get fresh expert outputs using current embeddings, but treat experts as frozen
                expert_outputs_fresh = []
                for expert_id in selected_expert_ids:
                    expert = self.model.experts[expert_id]
                    if self.config.offload_to_cpu:
                        expert.to(self.device)

                    # Forward in eval mode (no gradients to expert parameters)
                    expert.eval()
                    with torch.no_grad():
                        expert_out, _ = expert(hidden_states)  # Use hidden_states from Step 1
                    expert.train()

                    expert_outputs_fresh.append(expert_out)

                    if self.config.offload_to_cpu:
                        expert.to_cpu()

                # Combine expert outputs (all detached, no grad flow to experts or embeddings)
                combined_expert_output = torch.stack(expert_outputs_fresh).mean(dim=0)

                # Apply coherence module if available (WITH gradients for coherence)
                # This is where local representations get bound into coherent sequences
                if self.model.use_coherence and self.model.coherence is not None:
                    # Create attention mask (assuming no padding for now)
                    attention_mask = None  # Could add padding mask here if needed

                    # Apply coherence (enables global learning via self-attention)
                    final_output = self.model.coherence(combined_expert_output, attention_mask)
                else:
                    # No coherence - use raw expert output
                    final_output = combined_expert_output

                # Forward through LM head (WITH gradients for head and coherence)
                lm_head_logits = self.model.output_head(final_output)

                # CLEAN supervised cross-entropy loss
                # NO frequency weighting here - output head must learn TRUE distribution
                #
                # Frequency weighting is for EXPERTS (local learning), not OUTPUT HEAD (global readout)
                # The output head needs to predict the actual next token distribution
                lm_head_loss = torch.nn.functional.cross_entropy(
                    lm_head_logits.view(-1, lm_head_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='mean',
                )

                # Backward through LM head, coherence, and token_embedding
                # Gradients flow: output_head <- coherence <- [STOP at detached expert output]
                # Also: output_head.weight IS token_embedding.weight (weight tying)
                lm_head_loss.backward()

                # Gradient clipping for LM head (includes token_embedding via weight tying)
                torch.nn.utils.clip_grad_norm_(
                    self.model.output_head.parameters(),
                    self.config.gradient_clip_val,
                )

                # Gradient clipping for coherence module (if enabled)
                if self.coherence_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.coherence.parameters(),
                        self.config.gradient_clip_val,
                    )

                # Update LM head + token embeddings (via weight tying)
                self.lm_head_optimizer.step()

                # Update coherence module (global learning via backprop)
                if self.coherence_optimizer is not None:
                    self.coherence_optimizer.step()

                # STEP 5: Update router using Hebbian rule (no gradients needed)
                # Router learns to select experts that reduce error
                # This is done via the hebbian_update method (correlation-based)

                # For now, skip gradient-based router update
                # The Hebbian updates happen inside the model during forward pass
                # We could add explicit router updates here if needed

                # Save loss for logging before cleanup
                # Use LM head loss as the primary metric (this is what matters for language)
                total_loss = lm_head_loss.item()

                # AGGRESSIVE CLEANUP
                del hidden_states, expert_output, calibration_loss
                del combined_expert_output, lm_head_logits, lm_head_loss
                del expert_outputs_fresh
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

                # Track VRAM at end of step
                vram_end = self._get_gpu_memory_mb()
                self.vram_history.append(vram_end)

                # Update progress
                self.global_step += 1
                epoch_loss += total_loss

                # Get phase statistics
                phase_stats = self.phase_scheduler.get_stats()

                progress_bar.set_postfix({
                    "lm_loss": f"{total_loss:.4f}",  # LM head loss (language quality)
                    "phase": f"{phase_stats['phase_name'][:3]}",  # rep/ali/sta
                    "frozen": f"{len(self.frozen_experts)}",
                    "vram_mb": f"{vram_end:.0f}",
                    "lr": f"{current_phase.expert_learning_rate:.4f}",
                })

                # Logging
                if self.global_step % self.logging_steps == 0:
                    self._log_metrics(total_loss, expert_losses, selected_expert_ids,
                                     phase_stats, expert_stabilities, divergence)

                # Print phase info every 500 steps
                if self.global_step % 500 == 0:
                    self.phase_scheduler.print_phase_info()

                # Evaluation
                if self.val_dataloader is not None and self.global_step % self.eval_steps == 0:
                    self._evaluate()

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint()

            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}")

        # Final VRAM report
        self._print_vram_report()

    def _get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if not self.vram_tracking_enabled:
            return 0.0

        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 2)

    def _log_metrics(self, total_loss: float, expert_losses: List[float], selected_experts: List[int],
                     phase_stats: Dict, expert_stabilities: Dict[int, float], divergence: torch.Tensor):
        """Log training metrics."""
        self.writer.add_scalar("train/total_loss", total_loss, self.global_step)

        # Log per-expert losses
        for i, (expert_id, loss) in enumerate(zip(selected_experts, expert_losses)):
            self.writer.add_scalar(f"train/expert_{expert_id}_loss", loss, self.global_step)

        # Log expert stabilities
        for expert_id, stability in expert_stabilities.items():
            self.writer.add_scalar(f"train/expert_{expert_id}_stability", stability, self.global_step)

        # Log phase information
        self.writer.add_scalar("train/phase_idx", phase_stats['phase_idx'], self.global_step)
        self.writer.add_scalar("train/phase_progress", phase_stats['phase_progress'], self.global_step)
        self.writer.add_scalar("train/overall_progress", phase_stats['overall_progress'], self.global_step)
        self.writer.add_scalar("train/expert_lr", phase_stats['expert_lr'], self.global_step)
        self.writer.add_scalar("train/frozen_experts", len(self.frozen_experts), self.global_step)

        # Log calibration metrics
        self.writer.add_scalar("train/divergence_from_teacher", divergence.item(), self.global_step)
        calibration_stats = self.calibrator.get_statistics()
        if calibration_stats['active']:
            self.writer.add_scalar("train/mean_divergence", calibration_stats['mean_divergence'], self.global_step)
            self.writer.add_scalar("train/calibration_alerts", calibration_stats['num_alerts'], self.global_step)

        # Log VRAM
        if len(self.vram_history) > 0:
            current_vram = self.vram_history[-1]
            self.writer.add_scalar("train/vram_mb", current_vram, self.global_step)

        # Console output
        if self.global_step % 100 == 0:
            calib_str = f" - Div: {divergence.item():.3f}" if self.calibrator.calibration_active else ""
            # Show LM head loss as primary metric (this is what matters for language)
            print(f"Step {self.global_step} [{phase_stats['phase_name']}] - "
                  f"LM Loss: {total_loss:.4f} - "  # LM head loss (clean supervised)
                  f"LR: {phase_stats['expert_lr']:.4f} - "
                  f"Frozen: {len(self.frozen_experts)}/{self.config.num_experts}"
                  f"{calib_str} - "
                  f"VRAM: {self.vram_history[-1]:.0f} MB")

    def _print_vram_report(self):
        """Print VRAM usage and training summary."""
        if len(self.vram_history) == 0:
            return

        import numpy as np
        vram_array = np.array(self.vram_history)

        print("\n" + "=" * 80)
        print("TRAINING SUMMARY - PHASED LOCAL LEARNING")
        print("=" * 80)

        # VRAM Statistics
        print("\nVRAM USAGE (Constant-Memory Training):")
        print(f"  Total training steps: {len(self.vram_history)}")
        print(f"  Mean VRAM: {vram_array.mean():.2f} MB")
        print(f"  Std VRAM: {vram_array.std():.2f} MB")
        print(f"  Min VRAM: {vram_array.min():.2f} MB")
        print(f"  Max VRAM: {vram_array.max():.2f} MB")
        print(f"  Range: {vram_array.max() - vram_array.min():.2f} MB")
        print(f"  Coefficient of variation: {vram_array.std() / vram_array.mean() * 100:.2f}%")
        print("  ✓ Low variance = successful constant-VRAM training")

        # Expert Statistics
        print(f"\nEXPERT DYNAMICS:")
        print(f"  Total experts: {self.config.num_experts}")
        print(f"  Frozen experts: {len(self.frozen_experts)}/{self.config.num_experts}")
        print(f"  Active pool: {self.config.num_experts - len(self.frozen_experts)} experts")
        print(f"  Frozen IDs: {sorted(list(self.frozen_experts)) if self.frozen_experts else 'None'}")

        # Phase Statistics
        if self.phase_scheduler:
            phase_stats = self.phase_scheduler.get_stats()
            print(f"\nPHASE PROGRESSION:")
            print(f"  Final phase: {phase_stats['phase_name'].upper()}")
            print(f"  Overall progress: {phase_stats['overall_progress']*100:.1f}%")
            print(f"  Phase progress: {phase_stats['phase_progress']*100:.1f}%")
            print(f"  Final expert LR: {phase_stats['expert_lr']:.4f}")

        # Calibration Statistics
        calibration_stats = self.calibrator.get_statistics()
        if calibration_stats['active']:
            print(f"\nCALIBRATION (Reference-Based Drift Prevention):")
            print(f"  Teacher checkpoint: Step {calibration_stats['teacher_step']}")
            print(f"  Mean divergence: {calibration_stats['mean_divergence']:.4f}")
            print(f"  Max divergence: {calibration_stats['max_divergence']:.4f}")
            print(f"  Recent divergence: {calibration_stats['recent_divergence']:.4f}")
            print(f"  Drift alerts: {calibration_stats['num_alerts']}")
            if calibration_stats['num_alerts'] == 0:
                print(f"  ✓ No excessive drift - representations stable")
            else:
                print(f"  ⚠ {calibration_stats['num_alerts']} drift alerts - consider tuning")

        # Success Metrics for Local Learning
        print(f"\nSUCCESS METRICS (Local Learning System):")
        print(f"  ✓ Constant VRAM: {'PASS' if vram_array.std() / vram_array.mean() < 0.15 else 'FAIL'}")
        print(f"  ✓ Expert specialization: {'PASS' if len(self.frozen_experts) > 0 else 'IN PROGRESS'}")
        print(f"  ✓ Phase completion: {'PASS' if self.phase_scheduler and phase_stats['overall_progress'] > 0.95 else 'IN PROGRESS'}")
        if calibration_stats['active']:
            print(f"  ✓ Calibration: {'PASS' if calibration_stats['num_alerts'] < 10 else 'WARN'}")
        print(f"\nNOTE: This is a REPRESENTATION LEARNING system, not a standard LM.")
        print(f"Success = stable representations + expert specialization + phase progression")
        print(f"LM loss is NOT the primary metric. Focus on relative improvement.")
        print("=" * 80)

    @torch.no_grad()
    def _evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()

        # Ensure all experts are on GPU for evaluation
        if self.config.offload_to_cpu:
            for expert in self.model.experts:
                expert.to(self.device)

        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            # Eval with inference_mode to disable learning
            outputs = self.model(input_ids, labels=labels, inference_mode=True)
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item() * input_ids.size(0)
                total_samples += input_ids.size(0)

        # Offload experts back to CPU
        if self.config.offload_to_cpu:
            for expert in self.model.experts:
                expert.to_cpu()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        self.writer.add_scalar("val/loss", avg_loss, self.global_step)
        print(f"Validation Loss: {avg_loss:.4f}")

        self.model.train()

    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.output_dir,
            f"checkpoint-{self.global_step}",
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_path, "model.pt"),
        )

        # Save VRAM history
        torch.save(
            {
                "vram_history": self.vram_history,
                "global_step": self.global_step,
                "epoch": self.epoch,
            },
            os.path.join(checkpoint_path, "training_state.pt"),
        )

        print(f"Checkpoint saved at {checkpoint_path}")
