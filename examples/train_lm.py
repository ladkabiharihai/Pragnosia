"""Example training script for language modeling with Pragnosia."""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("../src")

from pragnosia import PragnosiaModel, PragnosiaTrainer
from pragnosia.utils.config import PragnosiaConfig


def prepare_dataset(tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1"):
    """Prepare dataset for training."""
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)

    def tokenize_function(examples):
        # Filter out empty texts
        texts = [text for text in examples["text"] if text and len(text.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        return tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    # Set format to torch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized_dataset


def main():
    # Configuration
    config = PragnosiaConfig(
        vocab_size=50257,
        hidden_size=768,
        num_experts=8,
        num_active_experts=2,
        num_hidden_layers=12,
        max_position_embeddings=256,
        learning_rate=0.0001,
        exploration_end=0.3,
        stabilization_end=0.7,
    )

    # Initialize model
    print("Initializing Pragnosia model...")
    model = PragnosiaModel(config)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    memory_stats = model.get_memory_statistics()
    print(f"Estimated GPU memory: {memory_stats['total_mb']:.2f} MB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    print("Loading dataset...")
    dataset = prepare_dataset(tokenizer)

    # Data collator for language modeling
    def collate_fn(batch):
        """Custom collate function to handle batching."""
        # Stack tensors properly
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = PragnosiaTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="./outputs/pragnosia_lm",
        logging_steps=50,
        eval_steps=500,
        save_steps=1000,
    )

    # Train
    print("Starting training...")
    trainer.train(num_epochs=3)

    print("Training complete!")


if __name__ == "__main__":
    main()
