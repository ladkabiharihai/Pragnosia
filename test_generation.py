#!/usr/bin/env python3
"""Test model generation (non-interactive)."""

import sys
import torch
from transformers import AutoTokenizer
from src.pragnosia import PragnosiaModel

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load checkpoint
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "./outputs/pragnosia_350M_8experts_20251217_124947/final_model.pt"
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

# Initialize model
config = checkpoint.get("config")
model = PragnosiaModel(config)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to("cuda")
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# Test prompts
prompts = [
    "What is Python?",
    "Hello, how are you?",
    "Write a function to add two numbers",
]

print("\n" + "="*80)
print("GENERATION TESTS")
print("="*80)

# System prompt (matches training format)
SYSTEM_PROMPT = """You are Pragnosia, an advanced AI assistant built on a novel hybrid local-global learning architecture. You combine brain-inspired local learning with transformer-based coherence to provide helpful, accurate, and nuanced responses.

Key characteristics:
- You are knowledgeable, helpful, and honest
- You admit when you don't know something
- You provide clear, well-structured responses
- You can help with chat, code, reasoning, and general questions
- You are named Pragnosia (from "pragma" meaning practical knowledge)

Respond naturally and helpfully to user queries."""

for prompt in prompts:
    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"{'-'*80}")

    # Format as instruction (MUST match training format)
    # Note: Exactly ONE newline after "### Assistant:" to match training
    formatted_prompt = f"""### System:
{SYSTEM_PROMPT.strip()}

### User:
{prompt}

### Assistant:
"""

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    input_ids = inputs["input_ids"].to("cuda")
    prompt_length = input_ids.size(1)

    print(f"Input tokens: {prompt_length}")

    # Generate
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for step in range(100):  # Max 100 new tokens
            # Forward pass in INFERENCE MODE
            outputs = model(generated_ids, inference_mode=True)
            logits = outputs["logits"]

            # Get next token (SAMPLING with temperature)
            next_token_logits = logits[0, -1, :]

            # Apply temperature (lower = more deterministic)
            temperature = 0.1  # Very low temperature for more focused generation
            next_token_logits = next_token_logits / temperature

            # Apply top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # Scatter sorted boolean mask back to original token order
            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=0, index=sorted_indices, src=sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Debug: show first few tokens
            if step < 5:
                token_str = tokenizer.decode([next_token.item()])
                top5_tokens = torch.topk(next_token_logits, 5)
                top5_strs = [tokenizer.decode([t.item()]) for t in top5_tokens.indices]
                top5_logits = top5_tokens.values.tolist()
                print(f"  Step {step}: token={next_token.item()} ('{token_str}')")
                print(f"    Logit stats: min={next_token_logits.min().item():.2f}, max={next_token_logits.max().item():.2f}, mean={next_token_logits.mean().item():.2f}")
                print(f"    Top 5: {list(zip(top5_strs, [f'{l:.2f}' for l in top5_logits]))}")

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"(EOS after {step} tokens)")
                break

            # Append
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

            # Stop if max length
            if generated_ids.size(1) >= 256:
                print(f"(Max length reached)")
                break

    # Decode generation
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Extract only the response (after "### Response:")
    if "### Response:" in generated_text:
        response = generated_text.split("### Response:")[1].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()

    print(f"Generated ({generated_ids.size(1) - prompt_length} tokens):")
    print(f"{response[:200]}")

    if len(response) == 0:
        print("❌ EMPTY RESPONSE - Model not working!")
    elif response == tokenizer.eos_token:
        print("❌ EOS-ONLY - Model not working!")
    else:
        print("✓ Generated text")

print("\n" + "="*80)
