"""Training script for large Pragnosia models (1-3B parameters)."""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import argparse

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("../src")

from pragnosia import PragnosiaModel, PragnosiaTrainer
from pragnosia.utils.config import PragnosiaConfig


def get_model_config(size="1B"):
    """Get configuration for different model sizes."""

    configs = {
        "1B": PragnosiaConfig(
            vocab_size=50257,
            hidden_size=2048,
            num_experts=16,
            num_active_experts=2,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=8192,
            max_position_embeddings=2048,
            learning_rate=0.0001,
            exploration_end=0.3,
            stabilization_end=0.7,
            hippocampus_capacity=50000,
            neocortex_capacity=200000,
            max_gpu_memory_gb=12.0,
            offload_to_cpu=True,
        ),
        "2B": PragnosiaConfig(
            vocab_size=50257,
            hidden_size=2560,
            num_experts=24,
            num_active_experts=2,
            num_hidden_layers=32,
            num_attention_heads=20,
            intermediate_size=10240,
            max_position_embeddings=2048,
            learning_rate=0.00008,
            exploration_end=0.3,
            stabilization_end=0.7,
            hippocampus_capacity=100000,
            neocortex_capacity=500000,
            max_gpu_memory_gb=16.0,
            offload_to_cpu=True,
        ),
        "3B": PragnosiaConfig(
            vocab_size=50257,
            hidden_size=3072,
            num_experts=32,
            num_active_experts=2,
            num_hidden_layers=40,
            num_attention_heads=24,
            intermediate_size=12288,
            max_position_embeddings=2048,
            learning_rate=0.00006,
            exploration_end=0.3,
            stabilization_end=0.7,
            hippocampus_capacity=150000,
            neocortex_capacity=750000,
            max_gpu_memory_gb=24.0,
            offload_to_cpu=True,
        ),
    }

    return configs.get(size, configs["1B"])


def prepare_dataset(tokenizer, dataset_name="wikitext", dataset_config="wikitext-103-raw-v1"):
    """Prepare dataset for training."""
    print(f"Loading {dataset_name} ({dataset_config})...")
    dataset = load_dataset(dataset_name, dataset_config)

    def tokenize_function(examples):
        # Filter out empty texts
        texts = [text for text in examples["text"] if text and len(text.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        return tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        desc="Filtering empty examples"
    )

    # Set format to torch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train large Pragnosia model")
    parser.add_argument("--size", type=str, default="1B", choices=["1B", "2B", "3B"],
                       help="Model size to train")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset to use")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1",
                       help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="./outputs/pragnosia_large",
                       help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Steps between logging")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Steps between evaluation")
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Steps between checkpoints")

    args = parser.parse_args()

    # Get configuration
    config = get_model_config(args.size)

    print("="*80)
    print(f"Training Pragnosia {args.size} Model")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model size: {args.size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num experts: {config.num_experts}")
    print(f"  Active experts: {config.num_active_experts}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")

    # Initialize model
    print("\nInitializing model...")
    model = PragnosiaModel(config)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e9:.2f}B")
    print(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")

    memory_stats = model.get_memory_statistics()
    print(f"  Estimated GPU memory: {memory_stats['total_mb']:.2f} MB")
    print(f"  Max GPU budget: {config.max_gpu_memory_gb} GB")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = prepare_dataset(tokenizer, args.dataset, args.dataset_config)

    print(f"\nDataset statistics:")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")

    # Data collator
    def collate_fn(batch):
        """Custom collate function to handle batching."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"  Training batches: {len(train_dataloader)}")
    print(f"  Validation batches: {len(val_dataloader)}")

    # Initialize trainer
    print("\nInitializing trainer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    trainer = PragnosiaTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    try:
        trainer.train(num_epochs=args.epochs)
        print("\n" + "="*80)
        print("Training complete!")
        print("="*80)

        # Print final statistics
        print("\nFinal Statistics:")
        memory_stats = model.get_memory_statistics()
        print(f"  GPU Memory: {memory_stats['total_mb']:.2f} MB")

        routing_stats = model.router.check_stability()
        print(f"  Routing Entropy: {routing_stats['routing_entropy']:.4f}")
        print(f"  Expert Balance: {routing_stats['expert_balance']:.4f}")
        print(f"  Routing Stable: {routing_stats['is_stable']}")

        plasticity_stats = trainer.plasticity_scheduler.get_statistics()
        print(f"  Plasticity Phase: {plasticity_stats['phase']}")
        print(f"  Safety Violations: {plasticity_stats['safety_violations']}")

        hippo_stats = model.hippocampus.get_statistics()
        print(f"  Hippocampus: {hippo_stats['size']}/{hippo_stats['capacity']}")

        neo_stats = model.neocortex.get_statistics()
        print(f"  Neocortex: {neo_stats['size']}/{neo_stats['capacity']}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer._save_checkpoint()

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
