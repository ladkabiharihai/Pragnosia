#!/usr/bin/env python3
"""
Stage 2 Training: Coherence-Only Fine-Tuning for Generation

This script takes a pretrained Pragnosia model (after Stage 1 representation learning)
and trains ONLY the coherence module and output head on pure language modeling.

Purpose:
- Stage 1 trained experts to learn rich representations via local learning
- Stage 2 trains coherence to bind those representations into coherent sequences

Freezes:
- All experts (keep learned representations)
- Router (keep routing behavior)
- Token/position embeddings (keep feature space)

Trains:
- Coherence module (learn sequential binding)
- Output head (learn token prediction from coherent representations)

Objective:
- Pure cross-entropy language modeling loss
- No intrinsic objectives (those are for representation learning)
- Focus: Grammatical, coherent, contextual generation
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from datetime import datetime

from src.pragnosia import PragnosiaModel
from src.pragnosia.data import InstructionDataset, LanguageModelingDataset


def freeze_for_generation(model: PragnosiaModel):
    """
    Freeze everything except coherence module and output head.

    This preserves the learned representations while allowing the model
    to learn how to generate coherent sequences from those representations.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze coherence module
    if model.coherence is not None:
        for param in model.coherence.parameters():
            param.requires_grad = True
        print("✓ Coherence module: TRAINABLE")
    else:
        raise ValueError("Cannot train coherence - coherence module is None!")

    # Unfreeze output head
    for param in model.output_head.parameters():
        param.requires_grad = True
    print("✓ Output head: TRAINABLE")

    # Verify what's frozen
    print("\n" + "="*80)
    print("PARAMETER STATUS")
    print("="*80)

    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"✓ TRAIN: {name:50s} {param.numel():>12,}")
        else:
            frozen_params += param.numel()

    print("="*80)
    print(f"Frozen parameters:    {frozen_params:>12,} ({frozen_params/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:>12,} ({trainable_params/1e6:.1f}M)")
    print(f"Total parameters:     {frozen_params+trainable_params:>12,} ({(frozen_params+trainable_params)/1e6:.1f}M)")
    print(f"Training fraction:    {trainable_params/(frozen_params+trainable_params)*100:.1f}%")
    print("="*80)


def train_coherence_stage(
    checkpoint_path: str,
    dataset_name: str = "alpaca",
    num_samples: int = 10000,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 0.0005,
    output_dir: str = None,
    device: str = "cuda",
):
    """Train coherence module only for generation."""

    print("="*80)
    print("STAGE 2: COHERENCE TRAINING FOR GENERATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {num_samples}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*80)
    print()

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/coherence_stage2_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Initialize model
    print("Initializing model...")
    model = PragnosiaModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"\n✓ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"  Coherence: {'enabled' if config.use_coherence_module else 'disabled'}")
    if model.coherence is not None:
        print(f"  Coherence size: {model.coherence.get_memory_size_mb():.1f} MB")

    # Freeze for generation training
    print("\nFreezing model for coherence training...")
    freeze_for_generation(model)

    # Setup optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    if dataset_name in ["alpaca", "dolly", "oasst1"]:
        # Instruction dataset
        train_dataset = InstructionDataset(
            # dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=512,
            split="train",
            max_samples=num_samples,
        )
        val_dataset = InstructionDataset(
            # dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=512,
            split="validation",
            max_samples=min(2000, num_samples // 5),
        )
    else:
        # Language modeling dataset
        train_dataset = LanguageModelingDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=512,
            split="train",
            max_samples=num_samples,
        )
        val_dataset = LanguageModelingDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=512,
            split="validation",
            max_samples=min(2000, num_samples // 5),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")

    # Training loop
    print("\n" + "="*80)
    print("TRAINING COHERENCE FOR GENERATION")
    print("="*80)
    print("Objective: Pure language modeling (cross-entropy loss)")
    print("Focus: Grammatical, coherent, contextual generation")
    print("="*80)
    print()

    model.train()
    global_step = 0

    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")

        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass (inference_mode=False to allow gradients in coherence)
            # But only coherence and output_head have requires_grad=True
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                inference_mode=False,  # Allow gradients
            )

            # Use PURE language modeling loss (ignore intrinsic loss)
            lm_loss = nn.functional.cross_entropy(
                outputs["logits"].view(-1, config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

            # Backward pass
            optimizer.zero_grad()
            lm_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()

            # Track metrics
            epoch_loss += lm_loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{lm_loss.item():.4f}",
                "avg_loss": f"{epoch_loss/(batch_idx+1):.4f}",
            })

            # Log every 100 steps
            if global_step % 100 == 0:
                print(f"\nStep {global_step}: Loss = {lm_loss.item():.4f}")

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Validation
        print("\nValidating...")
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, labels=labels, inference_mode=False)

                lm_loss = nn.functional.cross_entropy(
                    outputs["logits"].view(-1, config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                val_loss += lm_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        model.train()

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"coherence_epoch{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
        }, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Final save
    final_path = os.path.join(output_dir, "coherence_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_path)

    print("\n" + "="*80)
    print("COHERENCE TRAINING COMPLETED!")
    print("="*80)
    print(f"✓ Model saved: {final_path}")
    print(f"\nNext steps:")
    print(f"1. Test generation:")
    print(f"   python test_generation.py {final_path}")
    print(f"2. Interactive chat:")
    print(f"   python chat.py {final_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train coherence for generation")
    parser.add_argument("checkpoint", type=str, help="Path to Stage 1 checkpoint")
    parser.add_argument("--dataset", type=str, default="alpaca", help="Dataset name")
    parser.add_argument("--samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    train_coherence_stage(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
