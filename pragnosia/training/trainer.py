"""
Trainer for Pragnosia model.
"""

import os
import math
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from pragnosia.config import PragnosiaConfig
from pragnosia.model import Pragnosia
from pragnosia.training.config import TrainingConfig
from pragnosia.utils.memory import get_memory_stats, clear_memory, count_parameters

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Pragnosia model.

    Features:
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Logging (console, file, wandb)
    - Memory optimization
    """

    def __init__(
        self,
        model: Pragnosia,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup mixed precision
        self.scaler = None
        self.autocast_dtype = torch.float32
        if config.use_mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler("cuda")
            if config.mixed_precision_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
            else:
                self.autocast_dtype = torch.float16

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Logging
        self.log_history = []

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_yaml(str(self.output_dir / "training_config.yaml"))

        # Wandb
        self.wandb_run = None
        if config.log_to_wandb:
            self._setup_wandb()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        elif self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        elif self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # Calculate total steps
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.num_epochs

        warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(
                    self.config.min_learning_rate / self.config.learning_rate,
                    0.5 * (1 + math.cos(math.pi * progress))
                )
        elif self.config.lr_scheduler == "linear":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(
                    self.config.min_learning_rate / self.config.learning_rate,
                    1 - progress
                )
        else:  # constant
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config),
            )
        except ImportError:
            logger.warning("wandb not installed. Skipping wandb logging.")

    def _log(self, metrics: Dict[str, Any], step: int):
        """Log metrics."""
        self.log_history.append({"step": step, **metrics})

        # Console logging
        if step % self.config.logging_steps == 0:
            log_str = f"Step {step}"
            for k, v in metrics.items():
                if isinstance(v, float):
                    log_str += f" | {k}: {v:.4f}"
                else:
                    log_str += f" | {k}: {v}"
            logger.info(log_str)

        # Wandb logging
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _save_checkpoint(self, path: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass with mixed precision
        with autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.scaler is not None):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                energy_budget=self.config.energy_budget,
            )

            loss = outputs["loss"]
            moe_aux_loss = outputs.get("moe_aux_loss", torch.tensor(0.0))
            moe_z_loss = outputs.get("moe_z_loss", torch.tensor(0.0))

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "moe_aux_loss": moe_aux_loss.item(),
            "moe_z_loss": moe_z_loss.item(),
        }

    def _optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

        return grad_norm.item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.scaler is not None):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]

            total_loss += loss.item() * input_ids.shape[0]
            total_samples += input_ids.shape[0]

        self.model.train()

        avg_loss = total_loss / max(1, total_samples)
        perplexity = math.exp(min(avg_loss, 100))  # Cap perplexity

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.effective_batch_size()}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Mixed precision: {self.config.use_mixed_precision} ({self.config.mixed_precision_dtype})")

        params = count_parameters(self.model)
        logger.info(f"  Model parameters: {params['total_millions']:.2f}M")

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)

        self.model.train()
        accumulation_loss = 0.0
        accumulation_steps = 0

        # Calculate total steps
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.num_epochs

        progress_bar = tqdm(total=total_steps, initial=self.global_step, desc="Training")

        start_time = time.time()

        while self.global_step < total_steps:
            self.epoch += 1

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Training step
                step_metrics = self._train_step(batch)
                accumulation_loss += step_metrics["loss"]
                accumulation_steps += 1

                # Optimizer step after accumulation
                if accumulation_steps >= self.config.gradient_accumulation_steps:
                    grad_norm = self._optimizer_step()
                    self.global_step += 1

                    # Logging
                    metrics = {
                        "train_loss": accumulation_loss / accumulation_steps,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "moe_aux_loss": step_metrics["moe_aux_loss"],
                        "epoch": self.epoch,
                    }

                    if self.device.type == "cuda":
                        mem_stats = get_memory_stats()
                        metrics["gpu_memory_gb"] = mem_stats["allocated_gb"]

                    self._log(metrics, self.global_step)

                    # Reset accumulation
                    accumulation_loss = 0.0
                    accumulation_steps = 0

                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=metrics["train_loss"], lr=f"{metrics['learning_rate']:.2e}")

                    # Evaluation
                    if self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        if eval_metrics:
                            self._log(eval_metrics, self.global_step)

                            # Check for best model
                            if eval_metrics["val_loss"] < self.best_val_loss:
                                self.best_val_loss = eval_metrics["val_loss"]
                                self._save_checkpoint(
                                    str(self.output_dir / f"checkpoint-{self.global_step}.pt"),
                                    is_best=True
                                )

                    # Save checkpoint
                    if self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(
                            str(self.output_dir / f"checkpoint-{self.global_step}.pt")
                        )

                        # Cleanup old checkpoints
                        self._cleanup_checkpoints()

                    # Check if done
                    if self.global_step >= total_steps:
                        break

        progress_bar.close()

        # Final save
        self._save_checkpoint(str(self.output_dir / "final_model.pt"))

        # Training summary
        elapsed_time = time.time() - start_time
        logger.info(f"\nTraining complete!")
        logger.info(f"  Total steps: {self.global_step}")
        logger.info(f"  Total time: {elapsed_time / 60:.2f} minutes")
        logger.info(f"  Best validation loss: {self.best_val_loss:.4f}")

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.log_history, f, indent=2)

        if self.wandb_run is not None:
            self.wandb_run.finish()

        return self.log_history

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if self.config.save_total_limit <= 0:
            return

        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*.pt"),
            key=lambda p: int(p.stem.split("-")[1])
        )

        # Keep best model and final model
        checkpoints = [c for c in checkpoints if "best" not in c.stem and "final" not in c.stem]

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.debug(f"Removed old checkpoint: {oldest}")
