#!/usr/bin/env python3
"""
Demo script for Pragnosia model.

Shows basic model creation, forward pass, and text generation.
"""

import torch
import argparse
import logging

from pragnosia import Pragnosia, PragnosiaConfig
from pragnosia.utils.memory import get_memory_stats, count_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pragnosia Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base", "large"],
        help="Model configuration preset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for demo"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run text generation demo"
    )
    parser.add_argument(
        "--energy-budget",
        type=float,
        default=1.0,
        help="Energy budget (0.0-1.0) for MoE routing"
    )
    args = parser.parse_args()

    # Create config
    config_fn = getattr(PragnosiaConfig, args.config)
    config = config_fn()

    logger.info(f"Creating Pragnosia model with {args.config} config...")
    logger.info(f"  Hidden size: {config.hidden_size}")
    logger.info(f"  Num layers: {config.num_hidden_layers}")
    logger.info(f"  Num experts: {config.num_experts}")
    logger.info(f"  Experts per token: {config.num_experts_per_token}")

    # Create model
    model = Pragnosia(config)
    model = model.to(args.device)
    model.eval()

    # Show parameter count
    params = count_parameters(model)
    logger.info(f"Model parameters: {params['total_millions']:.2f}M")

    # Memory stats
    if args.device == "cuda":
        mem = get_memory_stats()
        logger.info(f"GPU memory used: {mem['allocated_gb']:.2f}GB")

    # Demo forward pass
    logger.info(f"\n--- Forward Pass Demo ---")
    input_ids = torch.randint(1, config.vocab_size, (1, args.seq_len), device=args.device)

    with torch.no_grad():
        outputs = model(input_ids, energy_budget=args.energy_budget)

    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Output logits shape: {outputs['logits'].shape}")
    logger.info(f"Energy budget used: {outputs['energy_budget_used']:.2f}")
    logger.info(f"MoE aux loss: {outputs['moe_aux_loss'].item():.6f}")

    # Demo generation
    if args.generate:
        logger.info(f"\n--- Generation Demo ---")
        prompt = torch.randint(1, config.vocab_size, (1, 5), device=args.device)
        logger.info(f"Prompt tokens: {prompt.tolist()}")

        with torch.no_grad():
            generated = model.generate(
                prompt,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50,
                do_sample=True,
                energy_budget=args.energy_budget,
            )

        logger.info(f"Generated tokens: {generated.tolist()}")
        logger.info(f"Generated {generated.shape[1] - prompt.shape[1]} new tokens")

    logger.info("\nDemo complete!")


if __name__ == "__main__":
    main()
