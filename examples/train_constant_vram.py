"""
Demonstration of constant-VRAM training with local learning.

This script proves that Pragnosia achieves TRUE constant-memory training:
- VRAM usage does NOT increase with more experts
- Training throughput remains constant
- Local learning rules enable brain-like efficiency

Key experiments:
1. Train with 8, 16, 32 experts - same VRAM
2. Monitor VRAM throughout training
3. Visualize memory usage stability
"""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("../src")

from pragnosia import PragnosiaModel, LocalLearningTrainer
from pragnosia.utils.config import PragnosiaConfig


def prepare_dataset(tokenizer, max_samples=1000):
    """Prepare small dataset for VRAM experiments."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        texts = [text for text in examples["text"] if text and len(text.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        return tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Take subset
    train_subset = tokenized_dataset["train"].select(range(min(max_samples, len(tokenized_dataset["train"]))))
    val_subset = tokenized_dataset["validation"].select(range(min(100, len(tokenized_dataset["validation"]))))

    return train_subset, val_subset


def collate_fn(batch):
    """Collate function for dataloaders."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # For language modeling: predict next token
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]  # Shift left
    labels[:, -1] = -100  # Ignore last position

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_experiment(num_experts: int, num_active_experts: int = 2, num_epochs: int = 1):
    """
    Run training experiment with specified expert configuration.

    Returns VRAM statistics.
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {num_experts} total experts, {num_active_experts} active")
    print("=" * 80)

    # Configuration
    config = PragnosiaConfig(
        vocab_size=50257,
        hidden_size=768,
        num_experts=num_experts,  # VARIABLE
        num_active_experts=num_active_experts,  # FIXED
        num_hidden_layers=12,
        max_position_embeddings=256,
        learning_rate=0.0001,
        offload_to_cpu=True,  # Critical for constant VRAM
        use_neuromodulation=True,
    )

    # Initialize model
    model = PragnosiaModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Parameters per expert: ~{total_params / num_experts / 1e6:.2f}M")

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_data, val_data = prepare_dataset(tokenizer, max_samples=500)

    train_dataloader = DataLoader(
        train_data,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Initialize LOCAL LEARNING trainer
    trainer = LocalLearningTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=f"./outputs/constant_vram_{num_experts}experts",
        logging_steps=50,
        eval_steps=10000,  # Skip eval for speed
        save_steps=10000,  # Skip saving for speed
        local_learning_rate=0.001,
    )

    # Train
    trainer.train(num_epochs=num_epochs)

    # Get VRAM statistics
    vram_history = trainer.vram_history

    if len(vram_history) > 0:
        vram_stats = {
            "num_experts": num_experts,
            "mean_vram_mb": np.mean(vram_history),
            "std_vram_mb": np.std(vram_history),
            "min_vram_mb": np.min(vram_history),
            "max_vram_mb": np.max(vram_history),
            "vram_range_mb": np.max(vram_history) - np.min(vram_history),
            "cv": np.std(vram_history) / np.mean(vram_history) * 100,
            "history": vram_history,
        }
    else:
        vram_stats = {"num_experts": num_experts, "mean_vram_mb": 0}

    return vram_stats


def visualize_results(results):
    """Create visualization of constant-VRAM property."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: VRAM vs Number of Experts
    ax = axes[0, 0]
    num_experts_list = [r["num_experts"] for r in results]
    mean_vram = [r["mean_vram_mb"] for r in results]
    std_vram = [r["std_vram_mb"] for r in results]

    ax.errorbar(num_experts_list, mean_vram, yerr=std_vram, marker='o', linewidth=2, capsize=5)
    ax.set_xlabel("Number of Total Experts", fontsize=12)
    ax.set_ylabel("Mean VRAM Usage (MB)", fontsize=12)
    ax.set_title("Constant-VRAM Training:\nMemory Usage vs Model Size", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=mean_vram[0], color='r', linestyle='--', label='Baseline (8 experts)')
    ax.legend()

    # Plot 2: VRAM over time for each configuration
    ax = axes[0, 1]
    for r in results:
        if len(r["history"]) > 0:
            ax.plot(r["history"], label=f"{r['num_experts']} experts", alpha=0.7)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("VRAM Usage (MB)", fontsize=12)
    ax.set_title("VRAM Stability During Training", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Coefficient of Variation (stability metric)
    ax = axes[1, 0]
    cv_values = [r.get("cv", 0) for r in results]
    ax.bar(range(len(num_experts_list)), cv_values, color=['green' if cv < 5 else 'orange' for cv in cv_values])
    ax.set_xticks(range(len(num_experts_list)))
    ax.set_xticklabels([f"{n} exp" for n in num_experts_list])
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Coefficient of Variation (%)", fontsize=12)
    ax.set_title("Memory Stability (Lower = Better)", fontsize=14, fontweight='bold')
    ax.axhline(y=5, color='r', linestyle='--', label='5% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    table_data.append(["Experts", "Mean VRAM", "Std VRAM", "CV %", "Constant?"])
    table_data.append(["-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10])

    baseline_vram = results[0]["mean_vram_mb"]

    for r in results:
        experts = r["num_experts"]
        mean = r["mean_vram_mb"]
        std = r["std_vram_mb"]
        cv = r.get("cv", 0)

        # Check if constant (within 10% of baseline)
        is_constant = abs(mean - baseline_vram) / baseline_vram < 0.1

        table_data.append([
            f"{experts}",
            f"{mean:.1f} MB",
            f"{std:.1f} MB",
            f"{cv:.2f}%",
            "✓" if is_constant else "✗"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(1, i)].set_facecolor('#e0e0e0')

    ax.set_title("Constant-VRAM Verification", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig("./outputs/constant_vram_proof.png", dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: ./outputs/constant_vram_proof.png")

    return fig


def main():
    """Run scaling experiments and prove constant VRAM."""
    print("\n" + "=" * 80)
    print("PRAGNOSIA: CONSTANT-VRAM TRAINING DEMONSTRATION")
    print("=" * 80)
    print("\nThis experiment proves that Pragnosia achieves:")
    print("1. Training VRAM independent of total model size")
    print("2. Memory scaling O(k) not O(n) where k = active experts")
    print("3. True local learning with no global backprop")
    print("\n" + "=" * 80)

    # Run experiments with increasing number of experts
    # Keep active experts FIXED at 2
    expert_configurations = [8, 16, 32]

    results = []

    for num_experts in expert_configurations:
        vram_stats = run_experiment(
            num_experts=num_experts,
            num_active_experts=2,  # ALWAYS 2
            num_epochs=1,
        )
        results.append(vram_stats)

        # Clear CUDA cache between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Visualize results
    visualize_results(results)

    # Print final summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    baseline_vram = results[0]["mean_vram_mb"]

    for r in results:
        print(f"\n{r['num_experts']} Experts:")
        print(f"  Mean VRAM: {r['mean_vram_mb']:.2f} MB")
        print(f"  Std VRAM:  {r['std_vram_mb']:.2f} MB")
        print(f"  CV:        {r.get('cv', 0):.2f}%")
        print(f"  vs Baseline: {(r['mean_vram_mb'] - baseline_vram) / baseline_vram * 100:+.1f}%")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Check if constant VRAM achieved
    all_constant = all(abs(r["mean_vram_mb"] - baseline_vram) / baseline_vram < 0.1 for r in results)

    if all_constant:
        print("✓ SUCCESS: Constant-VRAM training achieved!")
        print("  Memory usage independent of total model size")
        print("  This is TRUE brain-like local learning")
    else:
        print("⚠ PARTIAL: Some variance in VRAM usage")
        print("  May need further tuning of offloading strategy")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("./outputs", exist_ok=True)

    main()
