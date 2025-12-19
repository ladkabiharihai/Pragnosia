"""Example evaluation script for Pragnosia model."""
import torch
from transformers import AutoTokenizer

import sys
sys.path.append("../src")

from pragnosia import PragnosiaModel
from pragnosia.utils.config import PragnosiaConfig


def generate_text(model, tokenizer, prompt, max_length=100, device="cuda"):
    """Generate text using the Pragnosia model."""
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_ids=generated)
            logits = outputs["logits"]

            # Get next token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


def main():
    # Configuration (should match training config)
    config = PragnosiaConfig(
        vocab_size=50257,
        hidden_size=768,
        num_experts=8,
        num_active_experts=2,
    )

    # Initialize model
    print("Loading Pragnosia model...")
    model = PragnosiaModel(config)

    # Load checkpoint
    checkpoint_path = "./outputs/pragnosia_lm/checkpoint-10000/model.pt"
    model.load_state_dict(torch.load(checkpoint_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Print model statistics
    memory_stats = model.get_memory_statistics()
    print(f"\nMemory Statistics:")
    print(f"  Router: {memory_stats['router_mb']:.2f} MB")
    print(f"  Active Experts: {memory_stats['active_experts_mb']:.2f} MB")
    print(f"  Hippocampus: {memory_stats['hippocampus_mb']:.2f} MB")
    print(f"  Neocortex: {memory_stats['neocortex_mb']:.2f} MB")
    print(f"  Total: {memory_stats['total_mb']:.2f} MB")

    # Routing statistics
    routing_stats = model.router.check_stability()
    print(f"\nRouting Statistics:")
    print(f"  Entropy: {routing_stats['routing_entropy']:.4f}")
    print(f"  Balance: {routing_stats['expert_balance']:.4f}")
    print(f"  Stable: {routing_stats['is_stable']}")

    # Generate text
    prompts = [
        "The future of artificial intelligence is",
        "In a world where machines can learn,",
        "The key to continual learning is",
    ]

    print("\n" + "=" * 80)
    print("Text Generation Examples")
    print("=" * 80)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_text(model, tokenizer, prompt, max_length=50, device=device)
        print(f"Generated: {generated}")
        print("-" * 80)


if __name__ == "__main__":
    main()
