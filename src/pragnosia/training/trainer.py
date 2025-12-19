"""Trainer for Pragnosia model."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
from tqdm import tqdm
import os

from ..models.pragnosia_model import PragnosiaModel
from ..utils.config import PragnosiaConfig
from ..utils.plasticity import PlasticityScheduler


class PragnosiaTrainer:
    """
    Trainer for Pragnosia model with continual learning support.

    Handles:
    - Training loop with intrinsic + homeostatic objectives
    - Hebbian router updates
    - Controlled neuroplasticity (growth/pruning)
    - Memory consolidation
    - Logging and checkpointing
    """

    def __init__(
        self,
        model: PragnosiaModel,
        config: PragnosiaConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./outputs",
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 5000,
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

        # Move model to device (router and active experts only)
        self.model.to(device)

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Plasticity scheduler
        total_steps = len(train_dataloader) * 100  # Assume 100 epochs max
        self.plasticity_scheduler = PlasticityScheduler(
            total_steps=total_steps,
            exploration_end=config.exploration_end,
            stabilization_end=config.stabilization_end,
            max_growth_rate=config.max_growth_rate,
            max_pruning_rate=config.max_pruning_rate,
            min_active_params=config.min_active_params,
            max_active_params=config.max_active_params,
            min_expert_entropy=config.min_expert_entropy,
        )

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def train(self, num_epochs: int):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs: Number of training epochs
        """
        self.model.train()

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs["loss"]
                intrinsic_loss = outputs["intrinsic_loss"]
                homeostatic_loss = outputs["homeostatic_loss"]

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )

                self.optimizer.step()

                # Apply neuroplasticity operations
                self._apply_neuroplasticity()

                # Update global step
                self.global_step += 1
                epoch_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "intrinsic": f"{intrinsic_loss.item():.4f}",
                    "homeostatic": f"{homeostatic_loss.item():.4f}",
                })

                # Logging
                if self.global_step % self.logging_steps == 0:
                    self._log_metrics(outputs)

                # Evaluation
                if self.val_dataloader is not None and self.global_step % self.eval_steps == 0:
                    self._evaluate()

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint()

            # End of epoch logging
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
            self.writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

    def _apply_neuroplasticity(self):
        """Apply neuroplasticity operations based on current phase."""
        # Check safety bounds BEFORE applying plasticity
        active_param_ratio = self._compute_active_param_ratio()
        expert_entropy = self.model.router.get_routing_entropy()

        violations = self.plasticity_scheduler.check_safety_bounds(
            active_param_ratio,
            expert_entropy,
        )

        # Only apply plasticity if no violations
        if not any(violations.values()):
            phase = self.plasticity_scheduler.step()

            if self.plasticity_scheduler.can_grow():
                growth_rate = self.plasticity_scheduler.get_growth_rate()
                for expert in self.model.experts:
                    expert.grow_neurons(growth_rate)

            elif self.plasticity_scheduler.can_prune():
                pruning_rate = self.plasticity_scheduler.get_pruning_rate()
                for expert in self.model.experts:
                    expert.prune_neurons(pruning_rate)
        else:
            # Skip plasticity this step due to safety violations
            if self.global_step % 100 == 0:
                print(f"Warning: Skipping plasticity - safety violations: {violations}")

    def _compute_active_param_ratio(self) -> float:
        """Compute ratio of active parameters across all experts."""
        total_active = 0
        total_params = 0

        for expert in self.model.experts:
            if hasattr(expert, "pruning_mask"):
                total_active += expert.pruning_mask.sum().item()
                total_params += expert.pruning_mask.numel()

        return total_active / total_params if total_params > 0 else 1.0

    def _log_metrics(self, outputs: Dict):
        """Log metrics to tensorboard."""
        # Loss components
        self.writer.add_scalar(
            "train/total_loss",
            outputs["loss"].item(),
            self.global_step,
        )
        self.writer.add_scalar(
            "train/intrinsic_loss",
            outputs["intrinsic_loss"].item(),
            self.global_step,
        )
        self.writer.add_scalar(
            "train/homeostatic_loss",
            outputs["homeostatic_loss"].item(),
            self.global_step,
        )

        # Component losses - detailed intrinsic breakdown
        if "component_losses" in outputs:
            for key, value in outputs["component_losses"].items():
                if isinstance(value, torch.Tensor):
                    self.writer.add_scalar(
                        f"train/component/{key}",
                        value.item(),
                        self.global_step,
                    )

            # Log to console every 100 steps for monitoring
            if self.global_step % 100 == 0:
                # Separate neuromodulation stats from other components
                neuro_stats = {k: v for k, v in outputs["component_losses"].items() if k.startswith("neuro_")}
                other_stats = {k: v for k, v in outputs["component_losses"].items() if not k.startswith("neuro_")}

                comp_str = ", ".join([
                    f"{k}={v.item():.4f}" if isinstance(v, torch.Tensor) else f"{k}={v:.4f}"
                    for k, v in other_stats.items()
                ])
                print(f"Step {self.global_step} - Intrinsic components: {comp_str}")

                if neuro_stats:
                    neuro_str = ", ".join([
                        f"{k.replace('neuro_', '')}={v:.4f}" if isinstance(v, (int, float)) else f"{k.replace('neuro_', '')}={v.item():.4f}"
                        for k, v in neuro_stats.items()
                    ])
                    print(f"Step {self.global_step} - Neuromodulation: {neuro_str}")

        # Routing statistics
        routing_stats = outputs["routing_stats"]
        self.writer.add_scalar(
            "routing/entropy",
            routing_stats["routing_entropy"],
            self.global_step,
        )
        self.writer.add_scalar(
            "routing/balance",
            routing_stats["expert_balance"],
            self.global_step,
        )

        # Plasticity statistics
        plasticity_stats = self.plasticity_scheduler.get_statistics()
        self.writer.add_scalar(
            "plasticity/phase",
            ["exploration", "stabilization", "exploitation"].index(plasticity_stats["phase"]),
            self.global_step,
        )
        self.writer.add_scalar(
            "plasticity/growth_rate",
            plasticity_stats["growth_rate"],
            self.global_step,
        )
        self.writer.add_scalar(
            "plasticity/pruning_rate",
            plasticity_stats["pruning_rate"],
            self.global_step,
        )

        # Memory statistics
        memory_stats = self.model.get_memory_statistics()
        self.writer.add_scalar(
            "memory/total_mb",
            memory_stats["total_mb"],
            self.global_step,
        )
        self.writer.add_scalar(
            "memory/hippocampus_utilization",
            memory_stats["hippocampus_stats"]["utilization"],
            self.global_step,
        )
        self.writer.add_scalar(
            "memory/neocortex_utilization",
            memory_stats["neocortex_stats"]["utilization"],
            self.global_step,
        )

    @torch.no_grad()
    def _evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs["loss"].item() * input_ids.size(0)
            total_samples += input_ids.size(0)

        avg_loss = total_loss / total_samples

        # Log validation metrics
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

        # Save optimizer
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer.pt"),
        )

        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
            },
            os.path.join(checkpoint_path, "training_state.pt"),
        )

        print(f"Checkpoint saved at {checkpoint_path}")
