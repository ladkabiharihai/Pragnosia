#!/usr/bin/env python3
"""
Interactive Chat Interface for Pragnosia

Chat with your trained Pragnosia model in an interactive terminal session.

Usage:
    python chat.py path/to/model.pt
    python chat.py path/to/model.pt --temperature 0.7 --max-length 256
"""

import argparse
import torch
from transformers import AutoTokenizer
from src.pragnosia import PragnosiaModel
import sys


class PragnosiaChat:
    """Interactive chat interface for Pragnosia."""

    SYSTEM_PROMPT = """You are Pragnosia, an advanced AI assistant built on a novel hybrid local-global learning architecture. You combine brain-inspired local learning with transformer-based coherence to provide helpful, accurate, and nuanced responses.

Key characteristics:
- You are knowledgeable, helpful, and honest
- You admit when you don't know something
- You provide clear, well-structured responses
- You can help with chat, code, reasoning, and general questions
- You are named Pragnosia (from "pragma" meaning practical knowledge)

Respond naturally and helpfully to user queries."""

    def __init__(self, model_path, device="cuda", temperature=0.8, top_p=0.9, max_length=200):
        """Initialize chat interface."""
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length

        print("Loading Pragnosia...")
        print(f"Device: {device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        self.model = PragnosiaModel(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()

        print(f"✓ Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
        print(f"✓ Coherence: {'enabled' if config.use_coherence_module else 'disabled'}")
        print()

    def format_prompt(self, user_message, conversation_history=None):
        """Format user message with system prompt and history."""
        parts = [
            "### System:",
            self.SYSTEM_PROMPT.strip(),
            ""
        ]

        # Add conversation history if available
        if conversation_history:
            for turn in conversation_history:
                parts.extend([
                    "### User:",
                    turn["user"],
                    "",
                    "### Assistant:",
                    turn["assistant"],
                    ""
                ])

        # Add current user message
        parts.extend([
            "### User:",
            user_message.strip(),
            "",
            "### Assistant:",
        ])

        return "\n".join(parts)

    def generate_response(self, prompt):
        """Generate response from model."""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Discourage repetition
            )

        # Decode
        # Get only the newly generated tokens (skip the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][prompt_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def chat(self):
        """Run interactive chat session."""
        print("="*80)
        print("PRAGNOSIA CHAT")
        print("="*80)
        print("Type your message and press Enter to chat.")
        print("Commands: 'exit' or 'quit' to end, 'clear' to reset conversation")
        print("="*80)
        print()

        conversation_history = []

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Handle commands
                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "clear":
                    conversation_history = []
                    print("\n[Conversation cleared]\n")
                    continue

                if not user_input:
                    continue

                # Format prompt
                prompt = self.format_prompt(user_input, conversation_history)

                # Generate response
                print("Pragnosia: ", end="", flush=True)
                response = self.generate_response(prompt)
                print(response)
                print()

                # Add to history
                conversation_history.append({
                    "user": user_input,
                    "assistant": response
                })

                # Keep only last 5 turns to avoid context overflow
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-5:]

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Pragnosia")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--max-length", type=int, default=200, help="Max response length")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        chat = PragnosiaChat(
            model_path=args.model_path,
            device=args.device,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length
        )
        chat.chat()
    except FileNotFoundError:
        print(f"❌ Error: Model file not found: {args.model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
