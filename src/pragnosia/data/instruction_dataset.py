"""Instruction-following dataset with proper formatting for Pragnosia."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional
import os


class InstructionDataset(Dataset):
    """
    Instruction-following dataset formatted for Pragnosia.

    Uses Alpaca-style format with proper system prompts.
    """

    # Pragnosia's identity and capabilities
    SYSTEM_PROMPT = """You are Pragnosia, an advanced AI assistant built on a novel hybrid local-global learning architecture. You combine brain-inspired local learning with transformer-based coherence to provide helpful, accurate, and nuanced responses.

Key characteristics:
- You are knowledgeable, helpful, and honest
- You admit when you don't know something
- You provide clear, well-structured responses
- You can help with chat, code, reasoning, and general questions
- You are named Pragnosia (from "pragma" meaning practical knowledge)

Respond naturally and helpfully to user queries."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            split: Dataset split ("train" or "validation")
            cache_dir: Directory to cache downloaded datasets
            max_samples: Maximum number of samples (for testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Disable tokenizer parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load dataset
        self.data = self._load_dataset(split, cache_dir, max_samples)

        print(f"\n📚 Instruction Dataset Loaded:")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.data)}")
        print(f"  Max length: {max_length}")

    def _load_dataset(self, split, cache_dir, max_samples):
        """Load instruction-following dataset."""
        try:
            # Try Alpaca dataset
            print(f"Loading Alpaca instruction dataset...")
            dataset = load_dataset(
                "tatsu-lab/alpaca",
                split="train",
                cache_dir=cache_dir,
            )

            # Create train/val split
            if split == "train":
                dataset = dataset.select(range(0, int(len(dataset) * 0.95)))
            else:
                dataset = dataset.select(range(int(len(dataset) * 0.95), len(dataset)))

        except Exception as e:
            print(f"Warning: Could not load Alpaca ({e})")
            print("Creating synthetic instruction data...")
            return self._create_synthetic_data(max_samples or 1000)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    def _create_synthetic_data(self, num_samples):
        """Create synthetic instruction data for testing."""
        synthetic_data = []

        templates = [
            {
                "instruction": "What is Python?",
                "input": "",
                "output": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, artificial intelligence, automation, and more. Python emphasizes code readability with its notable use of significant indentation."
            },
            {
                "instruction": "Write a function to add two numbers",
                "input": "",
                "output": "Here's a simple function to add two numbers:\n\n```python\ndef add_numbers(a, b):\n    return a + b\n\n# Example usage\nresult = add_numbers(5, 3)\nprint(result)  # Output: 8\n```"
            },
            {
                "instruction": "Explain what machine learning is",
                "input": "",
                "output": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed. Instead of following predefined rules, ML algorithms identify patterns in data and make predictions or decisions based on those patterns. Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles."
            },
            {
                "instruction": "How do I reverse a string in Python?",
                "input": "",
                "output": "There are several ways to reverse a string in Python:\n\n1. Using slicing (most Pythonic):\n```python\ntext = 'hello'\nreversed_text = text[::-1]\nprint(reversed_text)  # 'olleh'\n```\n\n2. Using reversed() function:\n```python\ntext = 'hello'\nreversed_text = ''.join(reversed(text))\nprint(reversed_text)  # 'olleh'\n```\n\nThe slicing method is generally preferred for its simplicity and readability."
            },
        ]

        # Repeat templates to create more samples
        for i in range(num_samples):
            synthetic_data.append(templates[i % len(templates)])

        return synthetic_data

    def _format_instruction(self, instruction, input_text, output):
        """
        Format instruction in a chat-like format with system prompt.

        Format:
        ### System:
        {system_prompt}

        ### User:
        {instruction}
        {input_text if provided}

        ### Assistant:
        {output}
        """
        # Build prompt
        prompt_parts = [
            "### System:",
            self.SYSTEM_PROMPT.strip(),
            "",
            "### User:",
            instruction.strip()
        ]

        if input_text and input_text.strip():
            prompt_parts.append(input_text.strip())

        prompt_parts.extend([
            "",
            "### Assistant:",
            output.strip()
        ])

        return "\n".join(prompt_parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get fields
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        # Format as chat
        text = self._format_instruction(instruction, input_text, output)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()

        # CRITICAL FIX: Only compute loss on assistant's response (not the prompt or initial whitespace)
        # Strategy: Find "### Assistant:" and skip initial whitespace/newlines to find where actual text starts

        # Find where actual response content starts (after "### Assistant:\n")
        # We want to mask the prompt AND the newlines/whitespace between "### Assistant:" and the response
        text_decoded = self.tokenizer.decode(input_ids)
        assistant_marker = "### Assistant:"

        if assistant_marker in text_decoded:
            # Find where the assistant marker ends
            marker_end = text_decoded.index(assistant_marker) + len(assistant_marker)
            # Skip whitespace/newlines after the marker
            content_start = marker_end
            while content_start < len(text_decoded) and text_decoded[content_start] in ['\n', ' ', '\t']:
                content_start += 1

            # Find the corresponding token position
            # Encode the text up to content_start to see how many tokens that is
            prefix_tokens = self.tokenizer.encode(text_decoded[:content_start], add_special_tokens=False)
            response_start_idx = len(prefix_tokens)

            # Mask everything before the actual response content
            if response_start_idx < len(labels):
                labels[:response_start_idx] = -100
            else:
                # Fallback
                labels[:len(labels)//2] = -100
        else:
            # Fallback: if we can't find the marker, mask the first half
            labels[:len(labels)//2] = -100

        # Mask padding tokens in labels (-100 so they're ignored in loss)
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_stats(self):
        """Get dataset statistics."""
        return {
            "num_samples": len(self),
            "max_length": self.max_length,
            "split": self.split,
        }
