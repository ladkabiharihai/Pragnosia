#!/usr/bin/env python3
"""
Fine-tuning script for Pragnosia model.

Supports:
- Full fine-tuning
- LoRA fine-tuning (parameter-efficient)
- Instruction tuning
- Task-specific fine-tuning

Usage:
    # Full fine-tuning
    python scripts/finetune.py --checkpoint outputs/model.pt --data data/instructions.jsonl

    # LoRA fine-tuning
    python scripts/finetune.py --checkpoint outputs/model.pt --data data/instructions.jsonl --lora --lora-r 8

    # From HuggingFace dataset
    python scripts/finetune.py --checkpoint outputs/model.pt --hf-dataset tatsu-lab/alpaca --lora
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from pragnosia import Pragnosia, PragnosiaConfig
from pragnosia.training import TrainingConfig, Trainer, DataCollator
from pragnosia.training.data import TextDataset, SyntheticDataset
from pragnosia.modules.lora import (
    LoRAConfig,
    apply_lora,
    get_lora_state_dict,
    count_lora_parameters,
    merge_lora_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class InstructionDataset(torch.utils.data.Dataset):
    """Dataset for instruction fine-tuning."""

    def __init__(
        self,
        data_path: str,
        max_seq_length: int = 512,
        tokenizer=None,
    ):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.samples = []

        self._load_data(data_path)

    def _load_data(self, path: str):
        """Load instruction data."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    text = self._format_instruction(item)
                    if text:
                        self.samples.append(text)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.samples)} instruction samples")

    def _format_instruction(self, item: dict) -> str:
        """Format instruction into training text."""
        # Support various instruction formats
        if "instruction" in item and "output" in item:
            # Alpaca format
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item["output"]

            if input_text:
                return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        elif "prompt" in item and "completion" in item:
            # OpenAI format
            return f"{item['prompt']}\n{item['completion']}"

        elif "text" in item:
            return item["text"]

        elif "question" in item and "answer" in item:
            return f"Question: {item['question']}\nAnswer: {item['answer']}"

        return ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text = self.samples[idx]

        # Tokenize
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text)
        else:
            tokens = [hash(w) % 32000 for w in text.split()]

        # Truncate
        tokens = tokens[:self.max_seq_length]
        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }


def load_hf_instruction_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: int = None,
    max_seq_length: int = 512,
):
    """Load instruction dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Install with: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading HuggingFace dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    samples = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Format instruction
        text = ""
        if "instruction" in item:
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item.get("output", item.get("response", ""))

            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        elif "text" in item:
            text = item["text"]

        if text:
            tokens = [hash(w) % 32000 for w in text.split()][:max_seq_length]
            samples.append(torch.tensor(tokens, dtype=torch.long))

    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            tokens = self.data[idx]
            return {"input_ids": tokens, "labels": tokens.clone()}

    return SimpleDataset(samples)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Pragnosia model")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-config", type=str, help="Model config (if no checkpoint)")

    # Data
    parser.add_argument("--data", type=str, help="Path to fine-tuning data (JSONL)")
    parser.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to use")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")

    # LoRA
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, nargs="+",
                        default=["q_proj", "v_proj"], help="LoRA target modules")

    # Training
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")

    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs/finetune", help="Output directory")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA weights after training")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Get model config
    if "config" in checkpoint:
        # Load config from checkpoint
        config_dict = checkpoint["config"]
        if isinstance(config_dict, dict):
            model_config = PragnosiaConfig(**config_dict)
        else:
            model_config = config_dict
    elif args.model_config:
        model_config_fn = getattr(PragnosiaConfig, args.model_config)
        model_config = model_config_fn()
    else:
        logger.error("No model config found. Specify --model-config")
        sys.exit(1)

    # Create model
    logger.info("Creating model...")
    model = Pragnosia(model_config)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Apply LoRA if requested
    if args.lora:
        logger.info(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        lora_config = LoRAConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = apply_lora(model, lora_config)

        # Log parameter counts
        param_counts = count_lora_parameters(model)
        logger.info(f"  Total parameters: {param_counts['total_params'] / 1e6:.2f}M")
        logger.info(f"  Trainable parameters: {param_counts['trainable_params'] / 1e6:.2f}M")
        logger.info(f"  LoRA parameters: {param_counts['lora_params'] / 1e6:.2f}M")
        logger.info(f"  Trainable percentage: {param_counts['trainable_percentage']:.2f}%")

    # Load data
    if args.data:
        logger.info(f"Loading data from {args.data}")
        train_dataset = InstructionDataset(
            data_path=args.data,
            max_seq_length=args.max_seq_length,
        )
    elif args.hf_dataset:
        train_dataset = load_hf_instruction_dataset(
            dataset_name=args.hf_dataset,
            max_samples=args.max_samples,
            max_seq_length=args.max_seq_length,
        )
    else:
        logger.error("No data specified. Use --data or --hf-dataset")
        sys.exit(1)

    # Create dataloader
    collator = DataCollator(pad_token_id=model_config.pad_token_id)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
    )

    # Create training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        logging_steps=10,
        eval_steps=0,  # No eval for fine-tuning
        use_mixed_precision=True,
        seed=args.seed,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
    )

    # Train
    logger.info("Starting fine-tuning...")
    history = trainer.train()

    # Save final model
    output_path = Path(args.output_dir)

    if args.lora:
        # Save LoRA weights separately
        lora_state = get_lora_state_dict(model)
        lora_path = output_path / "lora_weights.pt"
        torch.save(lora_state, lora_path)
        logger.info(f"Saved LoRA weights to {lora_path}")

        if args.merge_lora:
            # Merge LoRA weights and save full model
            logger.info("Merging LoRA weights...")
            merge_lora_weights(model)
            merged_path = output_path / "merged_model.pt"
            torch.save(model.state_dict(), merged_path)
            logger.info(f"Saved merged model to {merged_path}")

    logger.info(f"\nFine-tuning complete! Output saved to {args.output_dir}")

    # Print final stats
    if history:
        final_loss = history[-1].get("train_loss", "N/A")
        logger.info(f"Final training loss: {final_loss}")


if __name__ == "__main__":
    main()
