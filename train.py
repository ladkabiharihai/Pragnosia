#!/usr/bin/env python3
"""
Pragnosia Training Script

Train a Pragnosia model with configurable parameters for chat, code, and reasoning.

Usage:
    # Small model (1B parameters, 16 experts)
    python train.py --model-size 1B --num-experts 16 --dataset all

    # Medium model (3B parameters, 32 experts)
    python train.py --model-size 3B --num-experts 32 --dataset all

    # Large model (7B parameters, 64 experts)
    python train.py --model-size 7B --num-experts 64 --dataset all

    # Custom configuration
    python train.py --hidden-size 1024 --num-experts 32 --num-layers 24 \\
                    --dataset chat,code,reasoning --epochs 3
"""

import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer
from datetime import datetime

from src.pragnosia import PragnosiaModel, LocalLearningTrainer
from src.pragnosia.utils.config import PragnosiaConfig
from src.pragnosia.data.multitask_dataset import MultitaskDataset


def calculate_model_size(hidden_size, num_layers, vocab_size, num_experts):
    """Calculate approximate model parameters."""
    # Token embeddings
    token_emb = vocab_size * hidden_size
    pos_emb = 2048 * hidden_size  # Max position embeddings

    # Each expert (FFN)
    intermediate_size = hidden_size * 4
    expert_params = 2 * hidden_size * intermediate_size  # fc1 + fc2
    total_expert_params = expert_params * num_experts * num_layers

    # Router (constant size)
    router_size = 256
    router_params = hidden_size * router_size * 2  # 2-layer projection

    # Output head (tied with embeddings, so no extra params)

    total = token_emb + pos_emb + total_expert_params + router_params
    return total


def get_preset_config(model_size_str):
    """Get preset configuration for common model sizes."""
    presets = {
        "350M": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_experts": 8,
            "description": "350M parameters - Good for testing and small experiments"
        },
        "1B": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_experts": 16,
            "description": "1B parameters - Small production model"
        },
        "3B": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_experts": 32,
            "description": "3B parameters - Medium production model"
        },
        "7B": {
            "hidden_size": 1536,
            "num_layers": 32,
            "num_experts": 64,
            "description": "7B parameters - Large production model"
        },
        "13B": {
            "hidden_size": 2048,
            "num_layers": 40,
            "num_experts": 96,
            "description": "13B parameters - Very large model (requires 24GB+ VRAM for training)"
        },
    }

    if model_size_str.upper() in presets:
        return presets[model_size_str.upper()]
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Pragnosia model for chat, code, and reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--model-size",
        type=str,
        choices=["350M", "1B", "3B", "7B", "13B", "custom"],
        default="1B",
        help="Preset model size or 'custom' for manual configuration"
    )
    model_group.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size (only for custom model)"
    )
    model_group.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers (only for custom model)"
    )
    model_group.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Total number of experts"
    )
    model_group.add_argument(
        "--num-active-experts",
        type=int,
        default=2,
        help="Number of active experts per forward pass (default: 2)"
    )
    model_group.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Vocabulary size (default: 50257 for GPT-2)"
    )

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset(s) to use: 'all', 'chat', 'code', 'reasoning', or comma-separated"
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate for embeddings and router"
    )
    train_group.add_argument(
        "--local-learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for expert local updates"
    )
    train_group.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    train_group.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value"
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for caching datasets"
    )
    data_group.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for testing)"
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # System configuration
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    system_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    system_group.add_argument(
        "--offload-to-cpu",
        action="store_true",
        default=True,
        help="Offload inactive experts to CPU (essential for constant-VRAM)"
    )
    system_group.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Log every N steps"
    )
    system_group.add_argument(
        "--eval-steps",
        type=int,
        default=1000,
        help="Evaluate every N steps"
    )
    system_group.add_argument(
        "--save-steps",
        type=int,
        default=5000,
        help="Save checkpoint every N steps"
    )
    system_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    return parser.parse_args()


def setup_config(args):
    """Create PragnosiaConfig from arguments."""
    # Get preset or use custom values
    if args.model_size != "custom":
        preset = get_preset_config(args.model_size)
        if preset is None:
            raise ValueError(f"Unknown model size: {args.model_size}")

        hidden_size = preset["hidden_size"]
        num_layers = preset["num_layers"]
        num_experts = args.num_experts if args.num_experts else preset["num_experts"]

        print(f"\n{'='*80}")
        print(f"Model Configuration: {args.model_size}")
        print(f"{'='*80}")
        print(f"Description: {preset['description']}")
        print(f"Hidden size: {hidden_size}")
        print(f"Number of layers: {num_layers}")
        print(f"Number of experts: {num_experts}")
    else:
        if not all([args.hidden_size, args.num_layers, args.num_experts]):
            raise ValueError(
                "For custom model, must specify --hidden-size, --num-layers, and --num-experts"
            )
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        num_experts = args.num_experts

    # Calculate model size
    total_params = calculate_model_size(
        hidden_size, num_layers, args.vocab_size, num_experts
    )
    print(f"Estimated parameters: {total_params / 1e6:.1f}M ({total_params / 1e9:.2f}B)")
    print(f"Active experts: {args.num_active_experts}")
    print(f"Memory scaling: O(k) where k={args.num_active_experts}")
    print(f"{'='*80}\n")

    # Create config
    config = PragnosiaConfig(
        vocab_size=args.vocab_size,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_active_experts=args.num_active_experts,
        num_hidden_layers=num_layers,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=args.max_length,
        learning_rate=args.learning_rate,
        gradient_clip_val=args.gradient_clip,
        offload_to_cpu=args.offload_to_cpu,
        use_neuromodulation=True,
    )

    return config


def setup_datasets(args, tokenizer):
    """Load and prepare datasets."""
    print("\n" + "="*80)
    print("Loading Datasets")
    print("="*80)

    # Parse dataset selection
    if args.dataset.lower() == "all":
        dataset_types = ["chat", "code", "reasoning"]
    else:
        dataset_types = [d.strip() for d in args.dataset.split(",")]

    print(f"Selected datasets: {', '.join(dataset_types)}")

    # Load datasets
    train_datasets = []
    val_datasets = []

    for dataset_type in dataset_types:
        print(f"\nLoading {dataset_type} dataset...")

        train_ds = MultitaskDataset(
            dataset_type=dataset_type,
            tokenizer=tokenizer,
            max_length=args.max_length,
            split="train",
            cache_dir=args.data_dir,
            max_samples=args.max_samples,
        )

        val_ds = MultitaskDataset(
            dataset_type=dataset_type,
            tokenizer=tokenizer,
            max_length=args.max_length,
            split="validation",
            cache_dir=args.data_dir,
            max_samples=min(1000, args.max_samples) if args.max_samples else 1000,
        )

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples: {len(val_ds)}")

    # Combine datasets
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    print(f"\nTotal train samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print("="*80 + "\n")

    return train_dataset, val_dataset


def setup_dataloaders(train_dataset, val_dataset, args):
    """Create dataloaders."""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )

    return train_dataloader, val_dataloader


def save_training_config(args, config, output_dir):
    """Save training configuration for reproducibility."""
    config_dict = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "model_config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_experts": config.num_experts,
            "num_active_experts": config.num_active_experts,
            "num_hidden_layers": config.num_hidden_layers,
            "max_position_embeddings": config.max_position_embeddings,
        },
        "training_config": {
            "learning_rate": config.learning_rate,
            "gradient_clip_val": config.gradient_clip_val,
            "offload_to_cpu": config.offload_to_cpu,
        }
    }

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Training configuration saved to: {config_path}")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir,
        f"pragnosia_{args.model_size}_{args.num_experts}experts_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("PRAGNOSIA TRAINING")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Offload to CPU: {args.offload_to_cpu}")
    print("="*80)

    # Setup configuration
    config = setup_config(args)

    # Save configuration
    save_training_config(args, config, output_dir)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")

    # Setup datasets
    train_dataset, val_dataset = setup_datasets(args, tokenizer)
    train_dataloader, val_dataloader = setup_dataloaders(train_dataset, val_dataset, args)

    # Initialize model
    print("\n" + "="*80)
    print("Initializing Model")
    print("="*80)
    model = PragnosiaModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.1f}M)")
    print("="*80 + "\n")

    # Initialize trainer
    trainer = LocalLearningTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=args.device,
        output_dir=output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        local_learning_rate=args.local_learning_rate,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded successfully")

    # Train
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Steps per epoch: {len(train_dataloader)}")
    print(f"Total steps: {len(train_dataloader) * args.epochs}")
    print("="*80 + "\n")

    try:
        trainer.train(num_epochs=args.epochs)

        print("\n" + "="*80)
        print("Training Completed Successfully!")
        print("="*80)
        print(f"Model saved to: {output_dir}")
        print(f"Total steps: {trainer.global_step}")
        print("="*80 + "\n")

        # Save final model
        final_path = os.path.join(output_dir, "final_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "global_step": trainer.global_step,
            "epoch": trainer.epoch,
        }, final_path)
        print(f"Final model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")

        interrupt_path = os.path.join(output_dir, "interrupted_checkpoint.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "global_step": trainer.global_step,
            "epoch": trainer.epoch,
        }, interrupt_path)
        print(f"Checkpoint saved to: {interrupt_path}")

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Try to save emergency checkpoint
        try:
            emergency_path = os.path.join(output_dir, "emergency_checkpoint.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "global_step": trainer.global_step,
                "epoch": trainer.epoch,
            }, emergency_path)
            print(f"Emergency checkpoint saved to: {emergency_path}")
        except:
            print("Failed to save emergency checkpoint")


if __name__ == "__main__":
    main()
