#!/usr/bin/env python3
"""
Pragnosia Interactive CLI

A comprehensive interactive tool for:
- Training models
- Fine-tuning with LoRA
- Chatting with trained models
- Evaluating model performance
- Testing inference

Usage:
    python scripts/pragnosia_cli.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch

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
from pragnosia.utils.memory import get_memory_stats, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


def print_menu(title: str, options: list):
    """Print a menu with numbered options."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}{title}{Colors.END}")
    print(f"{Colors.GREEN}{'-'*40}{Colors.END}")
    for i, (name, desc) in enumerate(options, 1):
        print(f"  {Colors.YELLOW}[{i}]{Colors.END} {Colors.BOLD}{name}{Colors.END}")
        print(f"      {Colors.CYAN}{desc}{Colors.END}")
    print(f"  {Colors.YELLOW}[0]{Colors.END} {Colors.BOLD}Exit / Go Back{Colors.END}")
    print()


def get_choice(prompt: str, max_choice: int) -> int:
    """Get user's menu choice."""
    while True:
        try:
            choice = input(f"{Colors.BOLD}{prompt}{Colors.END} ")
            choice = int(choice)
            if 0 <= choice <= max_choice:
                return choice
            print(f"{Colors.RED}Please enter a number between 0 and {max_choice}{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number{Colors.END}")


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default."""
    if default:
        result = input(f"{Colors.BOLD}{prompt}{Colors.END} [{Colors.CYAN}{default}{Colors.END}]: ")
        return result if result else default
    return input(f"{Colors.BOLD}{prompt}{Colors.END}: ")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input."""
    default_str = "Y/n" if default else "y/N"
    result = input(f"{Colors.BOLD}{prompt}{Colors.END} [{Colors.CYAN}{default_str}{Colors.END}]: ").lower()
    if not result:
        return default
    return result in ['y', 'yes', '1', 'true']


class PragnosiaCLI:
    """Interactive CLI for Pragnosia."""

    def __init__(self):
        self.model: Optional[Pragnosia] = None
        self.config: Optional[PragnosiaConfig] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path: Optional[str] = None
        self.tokenizer = None  # Simple tokenizer

    def simple_tokenize(self, text: str) -> torch.Tensor:
        """Simple word-based tokenization."""
        tokens = [hash(w) % (self.config.vocab_size - 3) + 3 for w in text.split()]
        return torch.tensor([tokens], dtype=torch.long)

    def simple_detokenize(self, tokens: torch.Tensor) -> str:
        """Convert tokens back to placeholder text."""
        # Since we use hash-based tokenization, we can't truly decode
        # This is a placeholder - in production, use a real tokenizer
        return f"[Generated {tokens.shape[1]} tokens]"

    def run(self):
        """Main CLI loop."""
        print_header("PRAGNOSIA - Brain-Inspired LLM")
        print(f"{Colors.CYAN}A modular, energy-efficient language model{Colors.END}")
        print(f"{Colors.CYAN}Device: {self.device}{Colors.END}")

        if self.device.type == "cuda":
            mem = get_memory_stats()
            print(f"{Colors.CYAN}GPU Memory: {mem['total_gb']:.1f}GB total, {mem['free_gb']:.1f}GB free{Colors.END}")

        while True:
            self.main_menu()

    def main_menu(self):
        """Display main menu."""
        options = [
            ("Create New Model", "Initialize a new Pragnosia model from scratch"),
            ("Load Model", "Load a trained model from checkpoint"),
            ("Train Model", "Train the current model on data"),
            ("Fine-tune with LoRA", "Parameter-efficient fine-tuning"),
            ("Chat with Model", "Interactive conversation with the model"),
            ("Evaluate Model", "Run benchmarks and evaluate performance"),
            ("Test Inference", "Quick inference tests"),
            ("Model Info", "View current model information"),
        ]

        print_menu("MAIN MENU", options)
        choice = get_choice("Select option:", len(options))

        if choice == 0:
            print(f"\n{Colors.GREEN}Goodbye!{Colors.END}\n")
            sys.exit(0)
        elif choice == 1:
            self.create_model_menu()
        elif choice == 2:
            self.load_model_menu()
        elif choice == 3:
            self.train_menu()
        elif choice == 4:
            self.finetune_menu()
        elif choice == 5:
            self.chat_menu()
        elif choice == 6:
            self.evaluate_menu()
        elif choice == 7:
            self.test_inference_menu()
        elif choice == 8:
            self.model_info()

    def create_model_menu(self):
        """Create a new model."""
        print_header("CREATE NEW MODEL")

        options = [
            ("Tiny (~125M params)", "Fast training, good for testing. Fits easily on 4GB GPU."),
            ("Small (~350M active / ~1B total)", "Good balance of quality and speed."),
            ("Base (~1B active / ~3B total)", "Better quality, requires more memory."),
            ("Large (~3B active / ~7B total)", "High quality, needs GPU offloading."),
            ("Efficient 4GB", "Optimized for inference on 4GB GPU."),
            ("Custom", "Configure your own model architecture."),
        ]

        print_menu("SELECT MODEL SIZE", options)
        choice = get_choice("Select model size:", len(options))

        if choice == 0:
            return

        config_map = {
            1: PragnosiaConfig.tiny,
            2: PragnosiaConfig.small,
            3: PragnosiaConfig.base,
            4: PragnosiaConfig.large,
            5: PragnosiaConfig.efficient_4gb,
        }

        if choice in config_map:
            self.config = config_map[choice]()
        elif choice == 6:
            self.config = self.custom_config_menu()
            if self.config is None:
                return

        print(f"\n{Colors.YELLOW}Creating model...{Colors.END}")
        self.model = Pragnosia(self.config)
        self.model = self.model.to(self.device)

        params = count_parameters(self.model)
        print(f"\n{Colors.GREEN}Model created successfully!{Colors.END}")
        print(f"  Total parameters: {Colors.BOLD}{params['total_millions']:.2f}M{Colors.END}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Experts: {self.config.num_experts} (top-{self.config.num_experts_per_token})")

    def custom_config_menu(self) -> Optional[PragnosiaConfig]:
        """Configure custom model."""
        print(f"\n{Colors.BOLD}Custom Model Configuration{Colors.END}")

        hidden_size = int(get_input("Hidden size", "1024"))
        num_layers = int(get_input("Number of layers", "12"))
        num_heads = int(get_input("Number of attention heads", "8"))
        num_experts = int(get_input("Number of experts", "8"))
        experts_per_token = int(get_input("Experts per token (top-k)", "2"))

        return PragnosiaConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=max(1, num_heads // 4),
            num_experts=num_experts,
            num_experts_per_token=experts_per_token,
        )

    def load_model_menu(self):
        """Load a model from checkpoint."""
        print_header("LOAD MODEL")

        checkpoint_path = get_input("Checkpoint path", "./outputs/final_model.pt")

        if not Path(checkpoint_path).exists():
            print(f"{Colors.RED}Checkpoint not found: {checkpoint_path}{Colors.END}")
            return

        print(f"\n{Colors.YELLOW}Loading model...{Colors.END}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Get config
            if "config" in checkpoint:
                config_data = checkpoint["config"]
                if isinstance(config_data, dict):
                    self.config = PragnosiaConfig(**config_data)
                else:
                    self.config = config_data
            else:
                print(f"{Colors.YELLOW}No config in checkpoint, using tiny config{Colors.END}")
                self.config = PragnosiaConfig.tiny()

            # Create and load model
            self.model = Pragnosia(self.config)

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model = self.model.to(self.device)
            self.model.eval()
            self.checkpoint_path = checkpoint_path

            params = count_parameters(self.model)
            print(f"\n{Colors.GREEN}Model loaded successfully!{Colors.END}")
            print(f"  Parameters: {params['total_millions']:.2f}M")
            print(f"  Checkpoint: {checkpoint_path}")

        except Exception as e:
            print(f"{Colors.RED}Error loading model: {e}{Colors.END}")

    def train_menu(self):
        """Training menu."""
        if self.model is None:
            print(f"{Colors.RED}No model loaded. Please create or load a model first.{Colors.END}")
            return

        print_header("TRAIN MODEL")

        options = [
            ("Train on Synthetic Data", "Quick test with random data (no real learning)"),
            ("Train on Text File", "Train on a .txt or .jsonl file"),
            ("Train with Custom Config", "Advanced training configuration"),
        ]

        print_menu("TRAINING OPTIONS", options)
        choice = get_choice("Select option:", len(options))

        if choice == 0:
            return
        elif choice == 1:
            self.train_synthetic()
        elif choice == 2:
            self.train_on_file()
        elif choice == 3:
            self.train_custom()

    def train_synthetic(self):
        """Train on synthetic data."""
        print(f"\n{Colors.BOLD}Training on Synthetic Data{Colors.END}")
        print(f"{Colors.CYAN}This is for testing the training pipeline, not real learning.{Colors.END}\n")

        num_samples = int(get_input("Number of samples", "1000"))
        max_steps = int(get_input("Max training steps", "100"))
        batch_size = int(get_input("Batch size", "8"))
        learning_rate = float(get_input("Learning rate", "3e-4"))

        self._run_training(
            dataset_type="synthetic",
            num_samples=num_samples,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

    def train_on_file(self):
        """Train on a text file."""
        print(f"\n{Colors.BOLD}Train on Text File{Colors.END}\n")

        data_path = get_input("Path to training data")
        if not Path(data_path).exists():
            print(f"{Colors.RED}File not found: {data_path}{Colors.END}")
            return

        max_steps = int(get_input("Max training steps (-1 for full epoch)", "-1"))
        batch_size = int(get_input("Batch size", "8"))
        learning_rate = float(get_input("Learning rate", "3e-4"))
        max_seq_length = int(get_input("Max sequence length", "512"))

        self._run_training(
            dataset_type="file",
            data_path=data_path,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
        )

    def train_custom(self):
        """Train with custom configuration."""
        print(f"\n{Colors.BOLD}Custom Training Configuration{Colors.END}\n")

        # Data
        use_synthetic = get_yes_no("Use synthetic data?", True)

        if use_synthetic:
            num_samples = int(get_input("Number of samples", "10000"))
            data_path = None
        else:
            data_path = get_input("Path to training data")
            num_samples = 0

        # Training params
        max_steps = int(get_input("Max training steps", "1000"))
        batch_size = int(get_input("Batch size", "8"))
        gradient_accumulation = int(get_input("Gradient accumulation steps", "4"))
        learning_rate = float(get_input("Learning rate", "3e-4"))
        warmup_steps = int(get_input("Warmup steps", "100"))
        max_seq_length = int(get_input("Max sequence length", "512"))

        # Memory optimization
        use_mixed_precision = get_yes_no("Use mixed precision (FP16)?", True)
        use_gradient_checkpointing = get_yes_no("Use gradient checkpointing?", True)

        # Output
        output_dir = get_input("Output directory", "./outputs")

        self._run_training(
            dataset_type="synthetic" if use_synthetic else "file",
            data_path=data_path,
            num_samples=num_samples,
            max_steps=max_steps,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_seq_length=max_seq_length,
            use_mixed_precision=use_mixed_precision,
            use_gradient_checkpointing=use_gradient_checkpointing,
            output_dir=output_dir,
        )

    def _run_training(
        self,
        dataset_type: str,
        data_path: str = None,
        num_samples: int = 1000,
        max_steps: int = 100,
        batch_size: int = 8,
        gradient_accumulation: int = 4,
        learning_rate: float = 3e-4,
        warmup_steps: int = 100,
        max_seq_length: int = 512,
        use_mixed_precision: bool = True,
        use_gradient_checkpointing: bool = True,
        output_dir: str = "./outputs",
    ):
        """Run training."""
        from torch.utils.data import DataLoader

        print(f"\n{Colors.YELLOW}Preparing training...{Colors.END}")

        # Enable gradient checkpointing
        if use_gradient_checkpointing:
            self.config.use_gradient_checkpointing = True

        # Create dataset
        if dataset_type == "synthetic":
            dataset = SyntheticDataset(
                num_samples=num_samples,
                seq_length=max_seq_length,
                vocab_size=self.config.vocab_size,
            )
        else:
            dataset = TextDataset(
                data_path=data_path,
                tokenizer=None,
                max_seq_length=max_seq_length,
            )

        collator = DataCollator(pad_token_id=self.config.pad_token_id)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            drop_last=True,
        )

        # Create training config
        train_config = TrainingConfig(
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            use_mixed_precision=use_mixed_precision,
            use_gradient_checkpointing=use_gradient_checkpointing,
            output_dir=output_dir,
            logging_steps=10,
            save_steps=max(100, max_steps // 5),
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            config=train_config,
            train_dataloader=dataloader,
        )

        # Train
        print(f"\n{Colors.GREEN}Starting training...{Colors.END}")
        print(f"  Steps: {max_steps}")
        print(f"  Batch size: {batch_size} x {gradient_accumulation} = {batch_size * gradient_accumulation}")
        print(f"  Learning rate: {learning_rate}")
        print()

        try:
            history = trainer.train()
            self.checkpoint_path = str(Path(output_dir) / "final_model.pt")

            print(f"\n{Colors.GREEN}Training complete!{Colors.END}")
            print(f"  Model saved to: {self.checkpoint_path}")

            if history:
                final_loss = history[-1].get("train_loss", "N/A")
                print(f"  Final loss: {final_loss:.4f}" if isinstance(final_loss, float) else f"  Final loss: {final_loss}")

            # Post-training options
            self.post_training_menu()

        except Exception as e:
            print(f"{Colors.RED}Training error: {e}{Colors.END}")

    def post_training_menu(self):
        """Menu shown after training completes."""
        print_header("TRAINING COMPLETE")

        options = [
            ("Chat with Model", "Test the model with interactive conversation"),
            ("Evaluate Model", "Run evaluation benchmarks"),
            ("Test Inference", "Quick inference speed test"),
            ("Continue Training", "Train for more steps"),
            ("Return to Main Menu", "Go back to main menu"),
        ]

        print_menu("WHAT WOULD YOU LIKE TO DO?", options)
        choice = get_choice("Select option:", len(options))

        if choice == 1:
            self.chat_menu()
        elif choice == 2:
            self.evaluate_menu()
        elif choice == 3:
            self.test_inference_menu()
        elif choice == 4:
            self.train_menu()
        # choice 0 or 5 returns to main menu

    def finetune_menu(self):
        """Fine-tuning with LoRA."""
        if self.model is None:
            print(f"{Colors.RED}No model loaded. Please create or load a model first.{Colors.END}")
            return

        print_header("FINE-TUNE WITH LoRA")

        print(f"{Colors.CYAN}LoRA (Low-Rank Adaptation) enables efficient fine-tuning{Colors.END}")
        print(f"{Colors.CYAN}by training only a small number of additional parameters.{Colors.END}\n")

        # LoRA config
        lora_r = int(get_input("LoRA rank (r)", "8"))
        lora_alpha = float(get_input("LoRA alpha", "16"))
        lora_dropout = float(get_input("LoRA dropout", "0.05"))

        # Data
        data_path = get_input("Path to fine-tuning data (JSONL)")
        if not Path(data_path).exists():
            print(f"{Colors.RED}File not found: {data_path}{Colors.END}")
            return

        # Training params
        max_steps = int(get_input("Max training steps", "500"))
        batch_size = int(get_input("Batch size", "4"))
        learning_rate = float(get_input("Learning rate", "2e-5"))

        print(f"\n{Colors.YELLOW}Applying LoRA...{Colors.END}")

        lora_config = LoRAConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        self.model = apply_lora(self.model, lora_config)

        param_counts = count_lora_parameters(self.model)
        print(f"{Colors.GREEN}LoRA applied!{Colors.END}")
        print(f"  Trainable: {param_counts['trainable_params'] / 1e6:.2f}M ({param_counts['trainable_percentage']:.2f}%)")

        # Run training
        self._run_training(
            dataset_type="file",
            data_path=data_path,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir="./outputs/finetune",
        )

    def chat_menu(self):
        """Interactive chat with the model."""
        if self.model is None:
            print(f"{Colors.RED}No model loaded. Please create or load a model first.{Colors.END}")
            return

        print_header("CHAT WITH MODEL")

        print(f"{Colors.CYAN}Type your message and press Enter. Type 'quit' to exit.{Colors.END}")
        print(f"{Colors.CYAN}Note: Using simple tokenization - output shows token count.{Colors.END}\n")

        # Chat settings
        max_new_tokens = int(get_input("Max new tokens per response", "100"))
        temperature = float(get_input("Temperature (0.1-2.0)", "0.7"))
        top_k = int(get_input("Top-K sampling", "50"))

        self.model.eval()

        print(f"\n{Colors.GREEN}Chat started! Type 'quit' to exit.{Colors.END}\n")

        while True:
            try:
                user_input = input(f"{Colors.BOLD}{Colors.BLUE}You:{Colors.END} ")

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"\n{Colors.GREEN}Chat ended.{Colors.END}")
                    break

                if not user_input.strip():
                    continue

                # Tokenize input
                input_ids = self.simple_tokenize(user_input).to(self.device)

                # Generate
                print(f"{Colors.YELLOW}Generating...{Colors.END}", end="\r")

                with torch.no_grad():
                    start_time = time.time()
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    gen_time = time.time() - start_time

                new_tokens = output_ids.shape[1] - input_ids.shape[1]
                tokens_per_sec = new_tokens / gen_time if gen_time > 0 else 0

                print(f"{Colors.BOLD}{Colors.GREEN}Model:{Colors.END} Generated {new_tokens} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
                print()

            except KeyboardInterrupt:
                print(f"\n{Colors.GREEN}Chat ended.{Colors.END}")
                break

    def evaluate_menu(self):
        """Evaluation menu."""
        if self.model is None:
            print(f"{Colors.RED}No model loaded. Please create or load a model first.{Colors.END}")
            return

        print_header("EVALUATE MODEL")

        options = [
            ("Perplexity Test", "Measure model's prediction quality on test data"),
            ("Generation Speed", "Test text generation throughput"),
            ("Memory Profile", "Profile GPU memory usage at different settings"),
            ("Full Evaluation", "Run all evaluations"),
        ]

        print_menu("EVALUATION OPTIONS", options)
        choice = get_choice("Select option:", len(options))

        if choice == 0:
            return
        elif choice == 1:
            self.eval_perplexity()
        elif choice == 2:
            self.eval_generation_speed()
        elif choice == 3:
            self.eval_memory()
        elif choice == 4:
            self.eval_perplexity()
            self.eval_generation_speed()
            self.eval_memory()

    def eval_perplexity(self):
        """Evaluate perplexity."""
        from torch.utils.data import DataLoader
        import math

        print(f"\n{Colors.BOLD}Perplexity Evaluation{Colors.END}\n")

        num_samples = int(get_input("Number of test samples", "500"))
        seq_length = int(get_input("Sequence length", "512"))
        batch_size = int(get_input("Batch size", "8"))

        # Create test dataset
        dataset = SyntheticDataset(
            num_samples=num_samples,
            seq_length=seq_length,
            vocab_size=self.config.vocab_size,
            seed=12345,  # Different seed for test
        )

        collator = DataCollator(pad_token_id=self.config.pad_token_id)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        print(f"{Colors.YELLOW}Evaluating...{Colors.END}")

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

                batch_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 100))

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Tokens evaluated: {total_tokens}")

    def eval_generation_speed(self):
        """Evaluate generation speed."""
        print(f"\n{Colors.BOLD}Generation Speed Test{Colors.END}\n")

        num_prompts = int(get_input("Number of prompts", "10"))
        max_new_tokens = int(get_input("Tokens to generate per prompt", "50"))
        prompt_length = int(get_input("Prompt length (tokens)", "20"))

        self.model.eval()
        total_tokens = 0
        total_time = 0

        print(f"\n{Colors.YELLOW}Testing generation speed...{Colors.END}")

        with torch.no_grad():
            for i in range(num_prompts):
                # Create random prompt
                input_ids = torch.randint(
                    1, self.config.vocab_size,
                    (1, prompt_length),
                    device=self.device,
                )

                start_time = time.time()
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                )
                gen_time = time.time() - start_time

                new_tokens = output_ids.shape[1] - input_ids.shape[1]
                total_tokens += new_tokens
                total_time += gen_time

                print(f"  Prompt {i+1}/{num_prompts}: {new_tokens} tokens in {gen_time:.2f}s")

        avg_speed = total_tokens / total_time if total_time > 0 else 0

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average speed: {avg_speed:.2f} tokens/second")

    def eval_memory(self):
        """Profile memory usage."""
        if self.device.type != "cuda":
            print(f"{Colors.YELLOW}Memory profiling works best on CUDA devices{Colors.END}")
            return

        print(f"\n{Colors.BOLD}Memory Profile{Colors.END}\n")

        batch_sizes = [1, 2, 4, 8]
        seq_lengths = [128, 256, 512, 1024]

        print(f"{Colors.YELLOW}Testing different configurations...{Colors.END}\n")

        results = []
        for bs in batch_sizes:
            for seq_len in seq_lengths:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                try:
                    input_ids = torch.randint(
                        1, self.config.vocab_size,
                        (bs, seq_len),
                        device=self.device,
                    )

                    with torch.no_grad():
                        _ = self.model(input_ids)

                    mem_stats = get_memory_stats()
                    print(f"  batch={bs}, seq={seq_len}: {mem_stats['peak_gb']:.2f}GB peak")
                    results.append((bs, seq_len, mem_stats['peak_gb'], "OK"))

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  batch={bs}, seq={seq_len}: {Colors.RED}OOM{Colors.END}")
                        results.append((bs, seq_len, 0, "OOM"))
                        torch.cuda.empty_cache()
                    else:
                        raise

        print(f"\n{Colors.GREEN}Memory Profile Complete{Colors.END}")

    def test_inference_menu(self):
        """Quick inference test."""
        if self.model is None:
            print(f"{Colors.RED}No model loaded. Please create or load a model first.{Colors.END}")
            return

        print_header("TEST INFERENCE")

        batch_size = int(get_input("Batch size", "1"))
        seq_length = int(get_input("Sequence length", "512"))
        num_iterations = int(get_input("Number of iterations", "10"))

        print(f"\n{Colors.YELLOW}Running inference test...{Colors.END}")

        self.model.eval()

        # Warmup
        input_ids = torch.randint(
            1, self.config.vocab_size,
            (batch_size, seq_length),
            device=self.device,
        )

        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_ids)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()

        with torch.no_grad():
            for i in range(num_iterations):
                _ = self.model(input_ids)
                print(f"  Iteration {i+1}/{num_iterations}", end="\r")

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        throughput = (batch_size * seq_length) / avg_time

        print(f"\n\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Average forward pass: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {throughput:.0f} tokens/second")

        if self.device.type == "cuda":
            mem_stats = get_memory_stats()
            print(f"  GPU Memory: {mem_stats['allocated_gb']:.2f}GB allocated")

    def model_info(self):
        """Display model information."""
        if self.model is None:
            print(f"{Colors.RED}No model loaded.{Colors.END}")
            return

        print_header("MODEL INFORMATION")

        params = count_parameters(self.model)

        print(f"{Colors.BOLD}Architecture:{Colors.END}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Intermediate size: {self.config.intermediate_size}")
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Attention heads: {self.config.num_attention_heads}")
        print(f"  KV heads (GQA): {self.config.num_key_value_heads}")
        print(f"  Max position: {self.config.max_position_embeddings}")

        print(f"\n{Colors.BOLD}MoE Configuration:{Colors.END}")
        print(f"  Experts: {self.config.num_experts}")
        print(f"  Experts per token: {self.config.num_experts_per_token}")
        print(f"  Energy gating: {self.config.enable_energy_gating}")

        print(f"\n{Colors.BOLD}Parameters:{Colors.END}")
        print(f"  Total: {params['total_millions']:.2f}M")
        print(f"  Trainable: {params['trainable_millions']:.2f}M")

        print(f"\n{Colors.BOLD}Memory Optimization:{Colors.END}")
        print(f"  Gradient checkpointing: {self.config.use_gradient_checkpointing}")
        print(f"  CPU offload: {self.config.use_cpu_offload}")
        print(f"  4-bit quantization: {self.config.load_in_4bit}")

        if self.checkpoint_path:
            print(f"\n{Colors.BOLD}Checkpoint:{Colors.END}")
            print(f"  Path: {self.checkpoint_path}")


def main():
    cli = PragnosiaCLI()
    cli.run()


if __name__ == "__main__":
    main()
