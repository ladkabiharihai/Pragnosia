#!/usr/bin/env python3
"""
Pragnosia Instruction Fine-Tuning Script

Fine-tune a pre-trained Pragnosia model on instruction-following tasks.
This teaches the model to respond properly to user queries in a chat format.

Usage:
    # Fine-tune pre-trained model
    python finetune_instruction.py --resume pretrained_model.pt --epochs 3 --max-samples 5000

    # Fine-tune with specific configuration
    python finetune_instruction.py --resume pretrained_model.pt --epochs 3 --batch-size 8
"""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from datetime import datetime
import json

from src.pragnosia import PragnosiaModel, LocalLearningTrainer
from src.pragnosia.data import InstructionDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Pragnosia on instruction-following tasks"
    )

    # Model
    parser.add_argument("--resume", type=str, required=True, help="Path to pre-trained model checkpoint")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--local-learning-rate", type=float, default=1e-3, help="Expert local learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for testing)")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")

    # System
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "="*80)
    print("PRAGNOSIA INSTRUCTION FINE-TUNING")
    print("="*80)
    print(f"Resume from: {args.resume}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*80 + "\n")

    # Load checkpoint to get config
    print("Loading checkpoint for configuration...")
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)

    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain config. Cannot resume.")

    config = checkpoint["config"]
    print(f"✓ Config loaded from checkpoint")
    print(f"  Model: {config.hidden_size}d, {config.num_experts} experts")
    print(f"  Coherence: {'enabled' if config.use_coherence_module else 'disabled'}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"instruction_tuned_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save configuration
    config_dict = {
        "timestamp": timestamp,
        "args": vars(args),
        "model_config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_experts": config.num_experts,
            "use_coherence_module": config.use_coherence_module,
        }
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Load datasets
    print("\nLoading instruction datasets...")
    train_dataset = InstructionDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        split="train",
        cache_dir="./data",
        max_samples=args.max_samples,
    )

    val_dataset = InstructionDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        split="validation",
        cache_dir="./data",
        max_samples=args.max_samples // 5 if args.max_samples else None,
    )

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

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")

    # Initialize model
    print("\n" + "="*80)
    print("Initializing Model")
    print("="*80)
    model = PragnosiaModel(config)
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Load weights from checkpoint
    print("\nLoading pre-trained weights...")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✓ Weights loaded successfully")

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
        local_learning_rate=args.local_learning_rate,
    )

    # Train
    print("\n" + "="*80)
    print("Starting Instruction Fine-Tuning")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total steps: {args.epochs * len(train_loader)}")
    print("="*80 + "\n")

    try:
        trainer.train(num_epochs=args.epochs)

        print("\n" + "="*80)
        print("Fine-Tuning Completed Successfully!")
        print("="*80)

        # Save final model
        final_path = output_dir / "pragnosia_chat_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": trainer.global_step,
            "args": vars(args),
        }, final_path)

        print(f"\n✓ Chat model saved to: {final_path}")
        print("\nNext steps:")
        print(f"1. Test chat: python chat.py {final_path}")
        print(f"2. Generate text: python test_generation.py {final_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")

        checkpoint_path = output_dir / "interrupted_checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": trainer.global_step,
            "args": vars(args),
        }, checkpoint_path)

        print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
