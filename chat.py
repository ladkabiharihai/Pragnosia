#!/usr/bin/env python3
"""
Pragnosia Interactive Chat Interface

Chat with your trained Pragnosia model interactively.

Usage:
    # Basic chat
    python chat.py --checkpoint ./outputs/final_model.pt

    # With custom parameters
    python chat.py --checkpoint ./outputs/final_model.pt --max-length 512 --temperature 0.8
"""

import argparse
import torch
from transformers import AutoTokenizer
import sys

from src.pragnosia import PragnosiaModel


class PragnosiaChat:
    """Interactive chat interface for Pragnosia."""

    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        max_length=512,
        temperature=0.8,  # Lower - trust the model's learned distribution
        top_p=0.9,  # Standard nucleus sampling
        top_k=0,  # Disabled - let top-p handle it
        repetition_penalty=1.2,  # Gentle penalty
        presence_penalty=0.3,  # Gentle discouragement
        frequency_penalty=0.3,  # Gentle discouragement
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

        self.conversation_history = []

    @torch.no_grad()
    def generate_response(self, prompt, max_new_tokens=64):
        """Generate a response to the prompt.

        Args:
            prompt: Input text
            max_new_tokens: Maximum new tokens to generate (default 64 for local learning models)
        """
        self.model.eval()

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
        )
        input_ids = inputs["input_ids"].to(self.device)
        prompt_length = input_ids.size(1)

        # Generate
        generated_ids = input_ids.clone()

        # Track token frequencies for penalties
        token_counts = {}

        # Track n-grams for blocking (CRITICAL for local learning models)
        ngram_history = {2: set(), 3: set(), 4: set()}  # Track 2-grams, 3-grams, 4-grams

        for step_idx in range(max_new_tokens):
            # Forward pass in INFERENCE MODE
            # This disables ALL learning (routing, intrinsic, neuromodulation)
            # CRITICAL for stable generation
            outputs = self.model(generated_ids, inference_mode=True)
            logits = outputs["logits"]

            # Get next token logits (before temperature)
            next_token_logits = logits[0, -1, :].clone()

            # CRITICAL FOR LOCAL LEARNING MODELS:
            # Apply multiple layers of repetition suppression

            # 1. Presence penalty: flat penalty for ANY token that appeared
            if self.presence_penalty > 0 and generated_ids.size(1) > prompt_length:
                unique_tokens = set(generated_ids[0, prompt_length:].tolist())
                for token_id in unique_tokens:
                    next_token_logits[token_id] -= self.presence_penalty

            # 2. Frequency penalty: penalty proportional to count
            if self.frequency_penalty > 0 and generated_ids.size(1) > prompt_length:
                for token_id in generated_ids[0, prompt_length:].tolist():
                    token_counts[token_id] = token_counts.get(token_id, 0) + 1

                for token_id, count in token_counts.items():
                    next_token_logits[token_id] -= self.frequency_penalty * count

            # 3. Repetition penalty: stronger for recent tokens
            # Check last 20 tokens for repetition
            if self.repetition_penalty != 1.0 and generated_ids.size(1) > prompt_length:
                recent_window = min(20, generated_ids.size(1) - prompt_length)
                recent_tokens = generated_ids[0, -recent_window:].tolist()

                # Apply decay: more recent = stronger penalty
                for i, token_id in enumerate(recent_tokens):
                    # Recency weight: 1.0 for most recent, decays to 0.2
                    recency = 0.2 + 0.8 * (i / max(1, recent_window - 1))
                    penalty = self.repetition_penalty * recency

                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= penalty
                    else:
                        next_token_logits[token_id] *= penalty

            # 4. Advanced loop detection and suppression
            # This is CRITICAL for local learning models which lack sequence control
            if generated_ids.size(1) >= prompt_length + 2:
                recent_gen = generated_ids[0, prompt_length:].tolist()

                # Check for immediate repetition (same token 2+ times)
                if len(recent_gen) >= 2 and recent_gen[-1] == recent_gen[-2]:
                    # Nuclear penalty: prevent third repetition
                    loop_token = recent_gen[-1]
                    next_token_logits[loop_token] -= 50.0

                # Check for pattern loops (ABC ABC pattern)
                if len(recent_gen) >= 6:
                    # Check if last 3 tokens == previous 3 tokens
                    last_3 = recent_gen[-3:]
                    prev_3 = recent_gen[-6:-3]
                    if last_3 == prev_3:
                        # Block all 3 tokens from continuing the pattern
                        for token in last_3:
                            next_token_logits[token] -= 30.0

                # Check for any token appearing > 30% of output
                if len(recent_gen) >= 10:
                    for token, count in token_counts.items():
                        ratio = count / len(recent_gen)
                        if ratio > 0.3:  # Token dominates output
                            # Progressive penalty based on dominance
                            next_token_logits[token] -= 20.0 * ratio

            # 5. N-gram blocking (DISABLED FOR NOW - was too destructive)
            # The model has good perplexity (3.61), it knows what it's doing
            # Only block EXACT repetition loops (same 3+ tokens in a row)
            if generated_ids.size(1) >= prompt_length + 6:
                recent_gen = generated_ids[0, prompt_length:].tolist()

                # Only check for EXACT loops (ABC ABC pattern)
                # Don't block all n-grams, just immediate repetition
                for n in [4, 3]:  # Only check 3-gram and 4-gram loops
                    if len(recent_gen) >= 2 * n:
                        # Check if last N tokens == previous N tokens
                        last_n = tuple(recent_gen[-n:])
                        prev_n = tuple(recent_gen[-2*n:-n])

                        if last_n == prev_n:
                            # LOOP DETECTED - block continuation
                            # Block only the tokens that would continue the loop
                            for token_id in last_n:
                                next_token_logits[token_id] -= 20.0  # Strong but not nuclear

            # Apply temperature AFTER penalties (so penalties are absolute)
            next_token_logits = next_token_logits / max(self.temperature, 0.5)

            # Apply top-k filtering
            if self.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, self.top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution (NEVER greedy!)
            probs = torch.softmax(next_token_logits, dim=-1)

            # Compute entropy for early stopping
            entropy = -(probs * torch.log(probs + 1e-10)).sum()

            # Early stop if model is very confident (entropy < 0.5) for 3+ consecutive tokens
            # This indicates the model has "finished its thought"
            if entropy < 0.5 and step_idx > 10:  # After at least 10 tokens
                # Model is very certain - likely found a good endpoint
                # Don't force more generation
                break

            # Add entropy floor only if entropy is EXTREMELY low (< 0.1)
            if entropy < 0.1:
                # Distribution is too peaked - add slight randomness
                probs = probs ** 0.9
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)

            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

            # Record n-grams for future blocking
            if generated_ids.size(1) > prompt_length:
                recent_gen = generated_ids[0, prompt_length:].tolist()
                for n in [2, 3, 4]:
                    if len(recent_gen) >= n:
                        ngram = tuple(recent_gen[-n:])
                        ngram_history[n].add(ngram)

            # Stop if max length reached
            if generated_ids.size(1) >= self.max_length:
                break

        # Decode only the newly generated tokens
        if generated_ids.size(1) > prompt_length:
            new_tokens = generated_ids[0, prompt_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            response = ""

        return response

    def chat(self, user_input):
        """Process user input and return response."""
        # Format as instruction
        formatted_prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"

        # Add conversation history for context (last 3 turns)
        if self.conversation_history:
            context = "\n\n".join(self.conversation_history[-3:])
            formatted_prompt = context + "\n\n" + formatted_prompt

        # Generate response (shorter for local learning models)
        response = self.generate_response(formatted_prompt, max_new_tokens=64)

        # Update conversation history
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")

        return response

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with Pragnosia model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
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
        "--temperature",
        type=float,
        default=0.8,  # LOWER - model has good perplexity, don't add noise
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,  # Standard nucleus sampling
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,  # DISABLE - let top-p handle it
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,  # MUCH LOWER - gentle penalty only
        help="Repetition penalty (1.0 = no penalty, higher = less repetition)"
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.3,  # MUCH LOWER - gentle discouragement
        help="Presence penalty - flat penalty for tokens that appeared (0.0 = no penalty)"
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.3,  # MUCH LOWER - gentle discouragement
        help="Frequency penalty - penalty proportional to token count (0.0 = no penalty)"
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    return model, config


def print_help():
    """Print help message."""
    print("\nAvailable commands:")
    print("  /help    - Show this help message")
    print("  /clear   - Clear conversation history")
    print("  /exit    - Exit chat")
    print("  /params  - Show current generation parameters")
    print("  /temp <value> - Set temperature (e.g., /temp 0.8)")
    print("  /code    - Switch to code mode")
    print("  /chat    - Switch to chat mode")
    print()


def main():
    """Main chat function."""
    args = parse_args()

    print("\n" + "="*80)
    print("PRAGNOSIA INTERACTIVE CHAT")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")

    # Load model
    model, config = load_model(args.checkpoint, args.device)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize chat
    chat = PragnosiaChat(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
    )

    print("\n" + "="*80)
    print("Chat initialized! Type '/help' for commands or start chatting.")
    print("="*80 + "\n")

    mode = "chat"  # chat or code

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command == "/exit":
                    print("\nGoodbye!")
                    break

                elif command == "/help":
                    print_help()
                    continue

                elif command == "/clear":
                    chat.reset_conversation()
                    print("Conversation history cleared.\n")
                    continue

                elif command == "/params":
                    print(f"\nCurrent parameters:")
                    print(f"  Temperature: {chat.temperature}")
                    print(f"  Top-p: {chat.top_p}")
                    print(f"  Top-k: {chat.top_k}")
                    print(f"  Max length: {chat.max_length}")
                    print(f"  Mode: {mode}")
                    print()
                    continue

                elif command.startswith("/temp "):
                    try:
                        temp = float(command.split()[1])
                        chat.temperature = temp
                        print(f"Temperature set to {temp}\n")
                    except:
                        print("Invalid temperature value\n")
                    continue

                elif command == "/code":
                    mode = "code"
                    print("Switched to code mode\n")
                    continue

                elif command == "/chat":
                    mode = "chat"
                    print("Switched to chat mode\n")
                    continue

                else:
                    print(f"Unknown command: {command}")
                    print("Type '/help' for available commands\n")
                    continue

            # Format input based on mode
            if mode == "code":
                formatted_input = f"### Code Task:\n{user_input}\n\n### Solution:"
            else:
                formatted_input = user_input

            # Generate response
            print("Assistant: ", end="", flush=True)
            response = chat.chat(formatted_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nUse /exit to quit\n")
            continue

        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
