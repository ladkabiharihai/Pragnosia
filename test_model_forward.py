#!/usr/bin/env python3
"""Quick test to verify model forward pass works correctly."""

import torch
from transformers import AutoTokenizer
from src.pragnosia import PragnosiaModel

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load checkpoint
checkpoint_path = "./outputs/pragnosia_350M_8experts_20251217_124947/final_model.pt"
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Get config
config = checkpoint.get("config")
if config is None:
    print("ERROR: No config in checkpoint")
    exit(1)

print(f"Config: hidden_size={config.hidden_size}, num_experts={config.num_experts}")

# Initialize model
print("Initializing model...")
model = PragnosiaModel(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Test forward pass
print("\nTest 1: Forward pass with single token")
input_ids = torch.tensor([[tokenizer.encode("Hello")[0]]])
print(f"Input shape: {input_ids.shape}")

with torch.no_grad():
    outputs = model(input_ids, inference_mode=True)
    logits = outputs["logits"]
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: (1, 1, {config.vocab_size})")

print("\nTest 2: Forward pass with multiple tokens")
text = "Hello world"
input_ids = torch.tensor([tokenizer.encode(text)])
print(f"Input: '{text}'")
print(f"Input shape: {input_ids.shape}")

with torch.no_grad():
    outputs = model(input_ids, inference_mode=True)
    logits = outputs["logits"]
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: (1, {input_ids.shape[1]}, {config.vocab_size})")

    # Try to sample next token
    next_token_logits = logits[0, -1, :]
    print(f"Next token logits shape: {next_token_logits.shape}")
    next_token = torch.argmax(next_token_logits)
    next_word = tokenizer.decode([next_token.item()])
    print(f"Next token: {next_token.item()} ('{next_word}')")

print("\n✓ All tests passed!")
