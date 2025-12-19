#!/usr/bin/env python3
"""
Pragnosia Evaluation Script

Evaluate a trained Pragnosia model on validation/test sets.

Usage:
    # Evaluate on all tasks
    python evaluate.py --checkpoint ./outputs/final_model.pt --dataset all

    # Evaluate on specific task
    python evaluate.py --checkpoint ./outputs/final_model.pt --dataset code

    # Evaluate with generation examples
    python evaluate.py --checkpoint ./outputs/final_model.pt --show-examples 10
"""

import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.pragnosia import PragnosiaModel
from src.pragnosia.data.multitask_dataset import MultitaskDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Pragnosia model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset(s) to evaluate: 'all', 'chat', 'code', 'reasoning'"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for caching datasets"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Number of generation examples to show"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for quick testing)"
    )

    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint does not contain config")

    # Initialize model
    model = PragnosiaModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num experts: {config.num_experts}")
    print(f"  Num layers: {config.num_hidden_layers}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    return model, config


@torch.no_grad()
def evaluate_dataset(model, dataloader, device, dataset_name="Dataset"):
    """Evaluate model on a dataset."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_tokens = 0
    # FIXED Priority 8: Add token-level accuracy tracking
    total_correct_tokens = 0

    pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")

    for batch in pbar:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass in INFERENCE MODE (disable learning during evaluation)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inference_mode=True,  # CRITICAL: No learning during eval
        )

        # Accumulate loss
        batch_loss = outputs["loss"].item()
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Count actual tokens (non-padding, non-ignore)
        valid_tokens = (labels != -100).sum().item()

        total_loss += batch_loss * valid_tokens
        total_samples += batch_size
        total_tokens += valid_tokens

        # FIXED Priority 8: Compute token-level accuracy
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)
        # Only count correct predictions for valid tokens (labels != -100)
        valid_mask = labels != -100
        correct_predictions = (predictions == labels) & valid_mask
        total_correct_tokens += correct_predictions.sum().item()

        # Update progress bar with accuracy
        current_avg_loss = total_loss / max(total_tokens, 1)
        current_accuracy = total_correct_tokens / max(total_tokens, 1) * 100
        # FIXED Priority 8: Cap perplexity at 10000 to avoid inf display
        current_ppl = min(np.exp(current_avg_loss), 10000.0)
        pbar.set_postfix({
            "loss": f"{current_avg_loss:.4f}",
            "ppl": f"{current_ppl:.1f}",
            "acc": f"{current_accuracy:.2f}%"
        })

    # Calculate final metrics
    avg_loss = total_loss / max(total_tokens, 1)
    # FIXED Priority 8: Capped perplexity (avoid inf from very high loss)
    perplexity = min(np.exp(avg_loss), 10000.0)
    # Token-level accuracy
    token_accuracy = (total_correct_tokens / max(total_tokens, 1)) * 100

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "perplexity_uncapped": np.exp(avg_loss),  # Keep uncapped for reference
        "token_accuracy": token_accuracy,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "correct_tokens": total_correct_tokens,
    }


@torch.no_grad()
def generate_examples(model, tokenizer, dataset, num_examples, max_length, device):
    """Generate text examples from prompts."""
    model.eval()

    examples = []

    for i in range(min(num_examples, len(dataset))):
        # Get item
        item = dataset[i]
        input_ids = item["input_ids"].unsqueeze(0).to(device)

        # Find the end of the prompt (before padding)
        # We'll generate from a truncated prompt
        prompt_length = min(50, (input_ids[0] != tokenizer.pad_token_id).sum().item())
        prompt_ids = input_ids[:, :prompt_length]

        # Decode prompt
        prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

        # Generate
        generated_ids = prompt_ids.clone()

        for _ in range(min(100, max_length - prompt_length)):
            # Forward pass
            outputs = model(generated_ids)
            logits = outputs["logits"]

            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Append
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

        # Decode generation
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        examples.append({
            "prompt": prompt_text,
            "generated": generated_text,
        })

    return examples


def main():
    """Main evaluation function."""
    args = parse_args()

    print("\n" + "="*80)
    print("PRAGNOSIA EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")

    # Load model
    model, config = load_model(args.checkpoint, args.device)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    if args.dataset.lower() == "all":
        dataset_types = ["chat", "code", "reasoning"]
    else:
        dataset_types = [d.strip() for d in args.dataset.split(",")]

    # Evaluate on each dataset
    results = {}

    for dataset_type in dataset_types:
        print(f"\n{'='*80}")
        print(f"Evaluating on {dataset_type.upper()} dataset")
        print("="*80)

        # Load dataset
        eval_dataset = MultitaskDataset(
            dataset_type=dataset_type,
            tokenizer=tokenizer,
            max_length=args.max_length,
            split="validation",
            cache_dir=args.data_dir,
            max_samples=args.max_samples,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Evaluate
        metrics = evaluate_dataset(model, eval_dataloader, args.device, dataset_type)

        print(f"\n{dataset_type.upper()} Results:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f} (capped at 10k)")
        if metrics['perplexity_uncapped'] > 10000:
            print(f"    (Uncapped: {metrics['perplexity_uncapped']:.2e})")
        print(f"  Token Accuracy: {metrics['token_accuracy']:.2f}%")
        print(f"  Correct tokens: {metrics['correct_tokens']:,} / {metrics['total_tokens']:,}")
        print(f"  Total samples: {metrics['total_samples']:,}")

        results[dataset_type] = metrics

        # Generate examples if requested
        if args.show_examples > 0:
            print(f"\nGenerating {args.show_examples} examples...")
            examples = generate_examples(
                model, tokenizer, eval_dataset,
                args.show_examples, args.max_length, args.device
            )

            print(f"\n{'-'*80}")
            print(f"GENERATION EXAMPLES ({dataset_type})")
            print("-"*80)

            for i, ex in enumerate(examples, 1):
                print(f"\nExample {i}:")
                print(f"Prompt: {ex['prompt'][:200]}...")
                print(f"Generated: {ex['generated'][len(ex['prompt']):200]}...")
                print("-"*40)

            results[f"{dataset_type}_examples"] = examples

    # Overall summary with improved metrics
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print("="*80)

    # FIXED Priority 8: Include new metrics in summary
    avg_loss = np.mean([r["loss"] for r in results.values() if isinstance(r, dict) and "loss" in r])
    avg_perplexity = np.mean([r["perplexity"] for r in results.values() if isinstance(r, dict) and "perplexity" in r])
    avg_accuracy = np.mean([r["token_accuracy"] for r in results.values() if isinstance(r, dict) and "token_accuracy" in r])

    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Perplexity: {avg_perplexity:.2f}")
    print(f"Average Token Accuracy: {avg_accuracy:.2f}%")

    for dataset_type in dataset_types:
        metrics = results[dataset_type]
        print(f"\n{dataset_type}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Token Accuracy: {metrics['token_accuracy']:.2f}%")

    print("="*80)

    # Save results if requested
    if args.output_file:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint,
            "results": results,
            "summary": {
                "avg_loss": avg_loss,
                "avg_perplexity": avg_perplexity,
                "avg_token_accuracy": avg_accuracy,
            }
        }

        # Remove examples from JSON (too large)
        output_data_for_json = {
            k: v for k, v in output_data.items()
            if not isinstance(v, dict) or not any("examples" in str(k2) for k2 in v.keys())
        }

        with open(args.output_file, "w") as f:
            json.dump(output_data_for_json, f, indent=2)

        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
