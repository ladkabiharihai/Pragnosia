#!/usr/bin/env python3
"""
Pragnosia Text Generation Utility

Simple script for generating text from prompts.

Usage:
    # Generate from command line
    python generate.py --checkpoint model.pt --prompt "Write a Python function to"

    # Generate from file
    python generate.py --checkpoint model.pt --input-file prompts.txt --output-file outputs.txt

    # Batch generation
    python generate.py --checkpoint model.pt --batch-mode
"""

import argparse
import torch
from transformers import AutoTokenizer
from pathlib import Path

from src.pragnosia import PragnosiaModel


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint does not contain config")

    model = PragnosiaModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    device,
    max_length=512,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_return_sequences=1,
):
    """Generate text from a prompt."""
    model.eval()

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length - max_new_tokens,
    )
    input_ids = inputs["input_ids"].to(device)

    results = []

    for _ in range(num_return_sequences):
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward
            outputs = model(generated_ids)
            logits = outputs["logits"]

            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Append
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

            if generated_ids.size(1) >= max_length:
                break

        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results.append(generated_text)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with Pragnosia model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to save outputs"
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Interactive batch mode (read prompts from stdin)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Number of sequences to generate per prompt"
    )

    return parser.parse_args()


def main():
    """Main generation function."""
    args = parse_args()

    print("Loading model...")
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Determine mode
    if args.prompt:
        # Single prompt mode
        prompts = [args.prompt]
    elif args.input_file:
        # File mode
        with open(args.input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.batch_mode:
        # Batch mode
        prompts = []
        print("\nBatch mode: Enter prompts (Ctrl+D or empty line to finish)")
        try:
            while True:
                prompt = input("Prompt: ").strip()
                if not prompt:
                    break
                prompts.append(prompt)
        except EOFError:
            pass
    else:
        print("Error: Must provide --prompt, --input-file, or --batch-mode")
        return

    if not prompts:
        print("No prompts provided")
        return

    print(f"\nGenerating for {len(prompts)} prompt(s)...")
    print("="*80)

    # Generate
    all_outputs = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}/{len(prompts)}:")
        print(f">>> {prompt}")
        print()

        outputs = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
        )

        for j, output in enumerate(outputs, 1):
            if args.num_return_sequences > 1:
                print(f"Output {j}:")
            print(output)
            print("-"*80)

            all_outputs.append({
                "prompt": prompt,
                "output": output,
            })

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for item in all_outputs:
                f.write(f"Prompt: {item['prompt']}\n")
                f.write(f"Output: {item['output']}\n")
                f.write("-"*80 + "\n\n")

        print(f"\nOutputs saved to: {args.output_file}")

    print(f"\nGeneration complete!")


if __name__ == "__main__":
    main()
