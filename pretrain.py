#!/usr/bin/env python3
"""
Pragnosia Pre-training Script

Pre-train Pragnosia on plain text for language modeling.
This is the FOUNDATION - it teaches the model what language IS.

After pre-training, you can fine-tune on specific tasks (chat, code, etc).

Usage:
    # Pre-train 350M model on WikiText-2 (fast, for testing)
    python pretrain.py --model-size 350M --dataset wikitext-2

    # Pre-train on WikiText-103 (better quality, ~1hr)
    python pretrain.py --model-size 350M --dataset wikitext-103 --epochs 2

    # Continue pre-training from checkpoint
    python pretrain.py --model-size 350M --resume ./outputs/pretrain_350M/checkpoint_1000.pt

Why pre-training matters:
    - Instruction-only training → learns template structure, not language
    - Pre-training first → learns language fundamentals
    - Then fine-tuning → learns specific task formats

    This is how ALL successful language models work (GPT, BERT, etc).
"""

import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datetime import datetime

from src.pragnosia import PragnosiaModel, LocalLearningTrainer
from src.pragnosia.utils.config import PragnosiaConfig
from src.pragnosia.data import LanguageModelingDataset


def get_preset_config(model_size_str):
    """Get preset configuration for common model sizes."""
    presets = {
        "350M": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_experts": 8,
            "description": "350M parameters - Good for testing"
        },
        "1B": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_experts": 16,
            "description": "1B parameters - Small production"
        },
        "3B": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_experts": 32,
            "description": "3B parameters - Medium production"
        },
    }

    if model_size_str.upper() in presets:
        return presets[model_size_str.upper()]
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-train Pragnosia on plain text for language modeling"
    )

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="350M",
        choices=["350M", "1B", "3B"],
        help="Preset model size"
    )
    parser.add_argument("--hidden-size", type=int, help="Hidden size (overrides preset)")
    parser.add_argument("--num-layers", type=int, help="Number of layers (overrides preset)")
    parser.add_argument("--num-experts", type=int, help="Number of experts (overrides preset)")
    parser.add_argument("--num-active-experts", type=int, default=2, help="Active experts per step")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size (GPT-2)")

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2",
        choices=["wikitext-2", "wikitext-103", "openwebtext"],
        help="Dataset to use for pre-training"
    )
    parser.add_argument("--max-samples", type=int, help="Max samples (for testing)")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--local-learning-rate", type=float, default=1e-3, help="Expert local learning rate")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping")

    # System configuration
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offload-to-cpu", action="store_true", default=True, help="Offload experts to CPU")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Get model configuration
    preset = get_preset_config(args.model_size)
    if preset is None:
        raise ValueError(f"Unknown model size: {args.model_size}")

    # Override with command line args
    hidden_size = args.hidden_size or preset["hidden_size"]
    num_layers = args.num_layers or preset["num_layers"]
    num_experts = args.num_experts or preset["num_experts"]

    print("\n" + "="*80)
    print("PRAGNOSIA PRE-TRAINING (Language Modeling)")
    print("="*80)
    print(f"Model: {args.model_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Offload to CPU: {args.offload_to_cpu}")
    print("="*80 + "\n")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset.replace("-", "_")
    output_dir = Path(args.output_dir) / f"pretrain_{args.model_size}_{dataset_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Print model configuration
    print("="*80)
    print(f"Model Configuration: {args.model_size}")
    print("="*80)
    print(f"Description: {preset['description']}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of experts: {num_experts}")
    print(f"Active experts: {args.num_active_experts}")
    print(f"Memory scaling: O(k) where k={args.num_active_experts}")
    print("="*80 + "\n")

    # Save training configuration
    config_to_save = {
        "timestamp": timestamp,
        "args": vars(args),
        "model_config": {
            "vocab_size": args.vocab_size,
            "hidden_size": hidden_size,
            "num_experts": num_experts,
            "num_active_experts": args.num_active_experts,
            "num_hidden_layers": num_layers,
            "max_position_embeddings": args.max_length,
        },
        "training_config": {
            "learning_rate": args.learning_rate,
            "gradient_clip_val": args.gradient_clip,
            "offload_to_cpu": args.offload_to_cpu,
        }
    }

    with open(output_dir / "pretrain_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    print(f"Pre-training configuration saved to: {output_dir}/pretrain_config.json\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}\n")

    # Map dataset names
    dataset_map = {
        "wikitext-2": "wikitext-2-raw-v1",
        "wikitext-103": "wikitext-103-raw-v1",
        "openwebtext": "openwebtext",
    }
    dataset_full_name = dataset_map[args.dataset]

    # Load datasets
    print("="*80)
    print("Loading Dataset")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print()

    train_dataset = LanguageModelingDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_full_name,
        max_length=args.max_length,
        split="train",
        cache_dir=args.data_dir,
        max_samples=args.max_samples,
    )

    val_dataset = LanguageModelingDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_full_name,
        max_length=args.max_length,
        split="validation",
        cache_dir=args.data_dir,
        max_samples=args.max_samples // 5 if args.max_samples else None,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    print("Train dataset stats:", train_dataset.get_stats())
    print("="*80 + "\n\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model configuration
    config = PragnosiaConfig(
        vocab_size=args.vocab_size,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_active_experts=args.num_active_experts,
        num_hidden_layers=num_layers,
        max_position_embeddings=args.max_length,
        offload_to_cpu=args.offload_to_cpu,
    )

    # Initialize model
    print("="*80)
    print("Initializing Model")
    print("="*80)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        config = checkpoint.get("config", config)
        model = PragnosiaModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded from checkpoint")
    else:
        model = PragnosiaModel(config)
        print("Model initialized from scratch")

    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.1f}M)")
    print("="*80 + "\n")

    # Initialize trainer
    trainer = LocalLearningTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=args.device,
        output_dir=output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )

    # Training
    print("\n" + "="*80)
    print("Starting Pre-training")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total steps: {args.epochs * len(train_loader)}")
    print("="*80 + "\n\n")

    # Train
    trainer.train(num_epochs=args.epochs)

    # Save final model
    print("\n" + "="*80)
    print("Pre-training Completed Successfully!")
    print("="*80)
    print(f"Model saved to: {output_dir}")
    print(f"Total steps: {trainer.global_step}")
    print("="*80)

    final_path = output_dir / "pretrained_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": trainer.global_step,
        "args": vars(args),
    }, final_path)

    print(f"\nPretrained model saved to: {final_path}")
    print("\nNext steps:")
    print(f"1. Test generation: python test_generation.py {final_path}")
    print(f"2. Fine-tune on tasks: python train.py --resume {final_path} --dataset chat")
    print()


if __name__ == "__main__":
    main()
