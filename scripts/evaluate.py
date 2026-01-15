#!/usr/bin/env python3
"""
Evaluation script for Pragnosia model.

Supports multiple evaluation benchmarks:
- Perplexity evaluation
- Text generation quality
- Memory and inference profiling
- Task-specific evaluations

Usage:
    # Evaluate perplexity
    python scripts/evaluate.py --checkpoint outputs/model.pt --eval-perplexity --data data/test.txt

    # Profile memory usage
    python scripts/evaluate.py --checkpoint outputs/model.pt --profile-memory

    # Full evaluation
    python scripts/evaluate.py --checkpoint outputs/model.pt --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pragnosia import Pragnosia, PragnosiaConfig
from pragnosia.training.data import TextDataset, DataCollator, SyntheticDataset
from pragnosia.utils.memory import get_memory_stats, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for Pragnosia models."""

    def __init__(
        self,
        model: Pragnosia,
        config: PragnosiaConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate perplexity on dataset."""
        logger.info("Evaluating perplexity...")

        total_loss = 0.0
        total_tokens = 0
        num_samples = 0

        for batch in tqdm(dataloader, desc="Perplexity"):
            if max_samples and num_samples >= max_samples:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Accumulate loss
            loss = outputs["loss"]
            batch_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_samples += input_ids.shape[0]

        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_samples": num_samples,
            "num_tokens": total_tokens,
        }

    @torch.no_grad()
    def evaluate_generation(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
    ) -> Dict[str, any]:
        """Evaluate text generation quality."""
        logger.info("Evaluating generation...")

        results = []
        total_time = 0.0
        total_tokens = 0

        for prompt in tqdm(prompts, desc="Generation"):
            # Simple tokenization
            input_ids = torch.tensor(
                [[hash(w) % self.config.vocab_size for w in prompt.split()]],
                dtype=torch.long,
                device=self.device,
            )

            start_time = time.time()
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=50,
            )
            gen_time = time.time() - start_time

            new_tokens = output_ids.shape[1] - input_ids.shape[1]
            total_time += gen_time
            total_tokens += new_tokens

            results.append({
                "prompt_length": input_ids.shape[1],
                "generated_tokens": new_tokens,
                "time": gen_time,
                "tokens_per_second": new_tokens / gen_time if gen_time > 0 else 0,
            })

        avg_speed = total_tokens / total_time if total_time > 0 else 0

        return {
            "num_prompts": len(prompts),
            "total_tokens_generated": total_tokens,
            "total_time": total_time,
            "average_tokens_per_second": avg_speed,
            "results": results,
        }

    def profile_memory(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        seq_lengths: List[int] = [128, 256, 512, 1024],
    ) -> Dict[str, any]:
        """Profile memory usage at different batch sizes and sequence lengths."""
        logger.info("Profiling memory usage...")

        results = []
        device = self.device

        if device.type != "cuda":
            logger.warning("Memory profiling is most accurate on CUDA devices")

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # Clear cache
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                try:
                    # Create dummy input
                    input_ids = torch.randint(
                        1, self.config.vocab_size,
                        (batch_size, seq_len),
                        device=device,
                    )

                    # Forward pass
                    with torch.no_grad():
                        _ = self.model(input_ids)

                    # Get memory stats
                    if device.type == "cuda":
                        mem_stats = get_memory_stats()
                        peak_memory = mem_stats["peak_gb"]
                        allocated = mem_stats["allocated_gb"]
                    else:
                        peak_memory = 0
                        allocated = 0

                    results.append({
                        "batch_size": batch_size,
                        "seq_length": seq_len,
                        "peak_memory_gb": peak_memory,
                        "allocated_memory_gb": allocated,
                        "status": "success",
                    })

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        results.append({
                            "batch_size": batch_size,
                            "seq_length": seq_len,
                            "status": "OOM",
                        })
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise e

        return {
            "device": str(device),
            "results": results,
        }

    def profile_inference_speed(
        self,
        batch_size: int = 1,
        seq_length: int = 512,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """Profile inference speed."""
        logger.info("Profiling inference speed...")

        device = self.device
        input_ids = torch.randint(
            1, self.config.vocab_size,
            (batch_size, seq_length),
            device=device,
        )

        # Warmup
        logger.info("Warming up...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(input_ids)

        # Benchmark
        if device.type == "cuda":
            torch.cuda.synchronize()

        logger.info(f"Running {num_iterations} iterations...")
        start_time = time.time()

        for _ in tqdm(range(num_iterations), desc="Speed"):
            with torch.no_grad():
                _ = self.model(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        throughput = (batch_size * seq_length) / avg_time

        return {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "num_iterations": num_iterations,
            "total_time": total_time,
            "avg_time_per_forward": avg_time,
            "tokens_per_second": throughput,
        }

    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        params = count_parameters(self.model)

        return {
            "total_parameters": params["total"],
            "total_parameters_millions": params["total_millions"],
            "trainable_parameters": params["trainable"],
            "trainable_parameters_millions": params["trainable_millions"],
            "num_experts": self.config.num_experts,
            "experts_per_token": self.config.num_experts_per_token,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_hidden_layers,
            "num_heads": self.config.num_attention_heads,
            "vocab_size": self.config.vocab_size,
        }


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    if "config" in checkpoint:
        config_data = checkpoint["config"]
        if isinstance(config_data, dict):
            config = PragnosiaConfig(**config_data)
        else:
            config = config_data
    else:
        logger.warning("No config in checkpoint, using tiny config")
        config = PragnosiaConfig.tiny()

    # Create model
    model = Pragnosia(config)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pragnosia model")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")

    # Evaluation modes
    parser.add_argument("--eval-perplexity", action="store_true", help="Evaluate perplexity")
    parser.add_argument("--eval-generation", action="store_true", help="Evaluate generation")
    parser.add_argument("--profile-memory", action="store_true", help="Profile memory usage")
    parser.add_argument("--profile-speed", action="store_true", help="Profile inference speed")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")

    # Data
    parser.add_argument("--data", type=str, help="Evaluation data path")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples for perplexity")

    # Output
    parser.add_argument("--output", type=str, default="./eval_results.json", help="Output file")

    # Profiling options
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length for profiling")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Create evaluator
    evaluator = ModelEvaluator(model, config, device)

    # Results
    results = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "model_info": evaluator.get_model_info(),
    }

    # Run evaluations
    if args.all or args.eval_perplexity:
        if args.data:
            dataset = TextDataset(
                data_path=args.data,
                tokenizer=None,
                max_seq_length=args.seq_length,
            )
        else:
            # Use synthetic data
            dataset = SyntheticDataset(
                num_samples=args.max_samples,
                seq_length=args.seq_length,
                vocab_size=config.vocab_size,
            )

        collator = DataCollator(pad_token_id=config.pad_token_id)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
        )

        perplexity_results = evaluator.evaluate_perplexity(
            dataloader, max_samples=args.max_samples
        )
        results["perplexity"] = perplexity_results
        logger.info(f"Perplexity: {perplexity_results['perplexity']:.2f}")

    if args.all or args.eval_generation:
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In the beginning",
            "The meaning of life is",
            "To be or not to be",
        ]
        generation_results = evaluator.evaluate_generation(prompts)
        results["generation"] = generation_results
        logger.info(f"Generation speed: {generation_results['average_tokens_per_second']:.2f} tok/s")

    if args.all or args.profile_memory:
        memory_results = evaluator.profile_memory()
        results["memory"] = memory_results

    if args.all or args.profile_speed:
        speed_results = evaluator.profile_inference_speed(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
        )
        results["speed"] = speed_results
        logger.info(f"Inference speed: {speed_results['tokens_per_second']:.2f} tok/s")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Model: {results['model_info']['total_parameters_millions']:.2f}M parameters")
    if "perplexity" in results:
        logger.info(f"Perplexity: {results['perplexity']['perplexity']:.2f}")
    if "speed" in results:
        logger.info(f"Speed: {results['speed']['tokens_per_second']:.2f} tokens/sec")


if __name__ == "__main__":
    main()
