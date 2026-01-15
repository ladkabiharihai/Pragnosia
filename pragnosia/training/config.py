"""
Training configuration for Pragnosia.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import yaml


@dataclass
class TrainingConfig:
    """Configuration for training Pragnosia."""

    # Model
    model_config: str = "tiny"  # tiny, small, base, large or path to config

    # Data
    train_data: str = ""  # Path to training data (txt, jsonl, or HF dataset)
    val_data: Optional[str] = None  # Path to validation data
    max_seq_length: int = 512
    num_workers: int = 4

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Schedule
    num_epochs: int = 1
    max_steps: int = -1  # -1 means use epochs
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # float16, bfloat16
    use_cpu_offload: bool = False

    # MoE specific
    moe_aux_loss_weight: float = 0.01
    energy_budget: float = 1.0  # Training energy budget

    # Plasticity
    enable_plasticity: bool = False
    plasticity_check_frequency: int = 500

    # Checkpointing
    output_dir: str = "./outputs"
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    log_to_wandb: bool = False
    wandb_project: str = "pragnosia"
    wandb_run_name: Optional[str] = None

    # Seed
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def effective_batch_size(self) -> int:
        """Get effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
