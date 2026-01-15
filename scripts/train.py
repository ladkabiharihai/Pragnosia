#!/usr/bin/env python3
"""
Training script for Pragnosia model.

Usage:
    # Train on synthetic data (for testing)
    python scripts/train.py --synthetic --model-config tiny --max-steps 100

    # Train on text file
    python scripts/train.py --train-data data/train.txt --model-config small

    # Train with custom config
    python scripts/train.py --config configs/train_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pragnosia import Pragnosia, PragnosiaConfig
from pragnosia.training import TrainingConfig, Trainer, DataCollator
from pragnosia.training.data import TextDataset, SyntheticDataset, create_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pragnosia model")

    # Config file
    parser.add_argument("--config", type=str, help="Path to training config YAML")

    # Model
    parser.add_argument(
        "--model-config",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base", "large"],
        help="Model configuration preset"
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Data
    parser.add_argument("--train-data", type=str, help="Path to training data")
    parser.add_argument("--val-data", type=str, help="Path to validation data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")

    # Training
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 for epochs)")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")

    # Memory optimization
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing")

    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="pragnosia", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, help="Wandb run name")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load or create training config
    if args.config:
        train_config = TrainingConfig.from_yaml(args.config)
        logger.info(f"Loaded training config from {args.config}")
    else:
        train_config = TrainingConfig(
            model_config=args.model_config,
            train_data=args.train_data or "",
            val_data=args.val_data,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            use_mixed_precision=not args.no_mixed_precision,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
            output_dir=args.output_dir,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            log_to_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            seed=args.seed,
            resume_from_checkpoint=args.resume,
        )

    # Create model config
    model_config_fn = getattr(PragnosiaConfig, train_config.model_config)
    model_config = model_config_fn()
    model_config.use_gradient_checkpointing = train_config.use_gradient_checkpointing

    logger.info(f"Creating {train_config.model_config} Pragnosia model...")
    logger.info(f"  Hidden size: {model_config.hidden_size}")
    logger.info(f"  Layers: {model_config.num_hidden_layers}")
    logger.info(f"  Experts: {model_config.num_experts} (top-{model_config.num_experts_per_token})")

    # Create model
    model = Pragnosia(model_config)

    # Create dataloaders
    if args.synthetic:
        logger.info("Using synthetic data for training...")
        train_dataset = SyntheticDataset(
            num_samples=10000,
            seq_length=train_config.max_seq_length,
            vocab_size=model_config.vocab_size,
            seed=args.seed,
        )
        val_dataset = SyntheticDataset(
            num_samples=1000,
            seq_length=train_config.max_seq_length,
            vocab_size=model_config.vocab_size,
            seed=args.seed + 1,
        )

        collator = DataCollator(pad_token_id=model_config.pad_token_id)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collator,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
    elif train_config.train_data:
        logger.info(f"Loading training data from {train_config.train_data}...")
        train_dataloader = create_dataloader(
            data_path=train_config.train_data,
            tokenizer=None,  # Will use simple tokenization
            batch_size=train_config.batch_size,
            max_seq_length=train_config.max_seq_length,
            shuffle=True,
            pad_token_id=model_config.pad_token_id,
        )

        val_dataloader = None
        if train_config.val_data:
            logger.info(f"Loading validation data from {train_config.val_data}...")
            val_dataloader = create_dataloader(
                data_path=train_config.val_data,
                tokenizer=None,
                batch_size=train_config.batch_size,
                max_seq_length=train_config.max_seq_length,
                shuffle=False,
                pad_token_id=model_config.pad_token_id,
            )
    else:
        logger.error("No training data specified! Use --train-data or --synthetic")
        sys.exit(1)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    logger.info(f"\nTraining complete! Output saved to {train_config.output_dir}")

    # Print final stats
    if history:
        final_loss = history[-1].get("train_loss", "N/A")
        logger.info(f"Final training loss: {final_loss}")


if __name__ == "__main__":
    main()
