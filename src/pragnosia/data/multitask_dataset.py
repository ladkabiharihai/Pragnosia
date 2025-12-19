"""Multi-task dataset for chat, code, and reasoning."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional, Dict
import os


class MultitaskDataset(Dataset):
    """
    Dataset that combines chat, code, and reasoning tasks.

    Supports:
    - Chat: Instruction following, conversational AI
    - Code: Code generation, completion, debugging
    - Reasoning: Math, logic, problem solving
    """

    def __init__(
        self,
        dataset_type: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset_type: Type of dataset - "chat", "code", or "reasoning"
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            split: Dataset split ("train" or "validation")
            cache_dir: Directory to cache downloaded datasets
            max_samples: Maximum number of samples (for testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.split = split

        # Set environment variable to disable parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load appropriate dataset
        self.data = self._load_dataset(dataset_type, split, cache_dir, max_samples)

    def _load_dataset(self, dataset_type, split, cache_dir, max_samples):
        """Load the appropriate dataset based on type."""

        if dataset_type == "chat":
            return self._load_chat_dataset(split, cache_dir, max_samples)
        elif dataset_type == "code":
            return self._load_code_dataset(split, cache_dir, max_samples)
        elif dataset_type == "reasoning":
            return self._load_reasoning_dataset(split, cache_dir, max_samples)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _load_chat_dataset(self, split, cache_dir, max_samples):
        """Load chat/instruction-following dataset."""
        try:
            # Try to load Alpaca-style instruction dataset
            dataset = load_dataset(
                "tatsu-lab/alpaca",
                split="train",
                cache_dir=cache_dir,
            )

            # Alpaca doesn't have a validation split, so we create one
            if split == "train":
                dataset = dataset.select(range(0, int(len(dataset) * 0.95)))
            else:
                dataset = dataset.select(range(int(len(dataset) * 0.95), len(dataset)))

        except Exception as e:
            print(f"Warning: Could not load Alpaca dataset ({e}), using fallback")
            # Fallback to a simpler instruction dataset
            try:
                dataset = load_dataset(
                    "HuggingFaceH4/instruction-dataset",
                    split=split if split == "train" else "train[:5%]",
                    cache_dir=cache_dir,
                )
            except:
                print("Warning: Could not load instruction dataset, creating synthetic examples")
                return self._create_synthetic_chat_data(max_samples or 100)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    def _load_code_dataset(self, split, cache_dir, max_samples):
        """Load code generation/completion dataset."""
        try:
            # Load CodeAlpaca or similar
            dataset = load_dataset(
                "sahil2801/CodeAlpaca-20k",
                split="train",
                cache_dir=cache_dir,
            )

            # Create train/val split
            if split == "train":
                dataset = dataset.select(range(0, int(len(dataset) * 0.95)))
            else:
                dataset = dataset.select(range(int(len(dataset) * 0.95), len(dataset)))

        except Exception as e:
            print(f"Warning: Could not load CodeAlpaca ({e}), trying alternative")
            try:
                # Alternative: code_search_net
                dataset = load_dataset(
                    "code_search_net",
                    "python",
                    split=split if split == "train" else "validation",
                    cache_dir=cache_dir,
                )
            except:
                print("Warning: Could not load code dataset, creating synthetic examples")
                return self._create_synthetic_code_data(max_samples or 100)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    def _load_reasoning_dataset(self, split, cache_dir, max_samples):
        """Load math/reasoning dataset."""
        try:
            # Load GSM8K (Grade School Math)
            dataset = load_dataset(
                "gsm8k",
                "main",
                split=split if split == "train" else "test",
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"Warning: Could not load GSM8K ({e}), trying alternative")
            try:
                # Alternative: MATH dataset
                dataset = load_dataset(
                    "competition_math",
                    split="train" if split == "train" else "test",
                    cache_dir=cache_dir,
                )
            except:
                print("Warning: Could not load reasoning dataset, creating synthetic examples")
                return self._create_synthetic_reasoning_data(max_samples or 100)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    def _create_synthetic_chat_data(self, num_samples):
        """Create synthetic chat examples as fallback."""
        examples = []
        templates = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Explain photosynthesis.", "Photosynthesis is the process by which plants convert sunlight into energy."),
            ("How do I bake a cake?", "To bake a cake, you need flour, eggs, sugar, butter, and baking powder. Mix the ingredients and bake at 350°F for 30 minutes."),
            ("What is machine learning?", "Machine learning is a subset of AI where systems learn from data to improve their performance."),
            ("Tell me a joke.", "Why did the programmer quit his job? Because he didn't get arrays!"),
        ]

        for i in range(num_samples):
            instruction, response = templates[i % len(templates)]
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response,
            })

        return examples

    def _create_synthetic_code_data(self, num_samples):
        """Create synthetic code examples as fallback."""
        examples = []
        templates = [
            ("Write a function to reverse a string", "def reverse_string(s):\n    return s[::-1]"),
            ("Create a function to check if a number is prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
            ("Write a function to find the factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"),
        ]

        for i in range(num_samples):
            instruction, code = templates[i % len(templates)]
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": code,
            })

        return examples

    def _create_synthetic_reasoning_data(self, num_samples):
        """Create synthetic reasoning examples as fallback."""
        examples = []
        templates = [
            ("If John has 5 apples and gives 2 to Mary, how many does he have left?", "John has 5 - 2 = 3 apples left."),
            ("What is 15% of 200?", "15% of 200 = 0.15 × 200 = 30"),
            ("If a train travels 60 mph for 2 hours, how far does it go?", "Distance = Speed × Time = 60 × 2 = 120 miles"),
        ]

        for i in range(num_samples):
            question, answer = templates[i % len(templates)]
            examples.append({
                "question": question,
                "answer": answer,
            })

        return examples

    def _format_prompt(self, item: Dict) -> str:
        """Format item into instruction-following prompt."""

        if self.dataset_type == "chat":
            # Alpaca-style format
            if "instruction" in item:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")

                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            else:
                # Fallback for other chat formats
                prompt = item.get("text", str(item))

        elif self.dataset_type == "code":
            # Code generation format
            if "instruction" in item:
                instruction = item.get("instruction", "")
                output = item.get("output", "")
                prompt = f"### Code Task:\n{instruction}\n\n### Solution:\n{output}"
            elif "func_documentation_string" in item:
                # code_search_net format
                doc = item.get("func_documentation_string", "")
                code = item.get("whole_func_string", "")
                prompt = f"### Documentation:\n{doc}\n\n### Code:\n{code}"
            else:
                prompt = str(item)

        elif self.dataset_type == "reasoning":
            # Math/reasoning format
            if "question" in item:
                question = item.get("question", "")
                answer = item.get("answer", "")
                prompt = f"### Problem:\n{question}\n\n### Solution:\n{answer}"
            else:
                prompt = str(item)

        return prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get tokenized item."""
        item = self.data[idx]

        # Format prompt
        prompt = self._format_prompt(item)

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels for next-token prediction
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left
        labels[-1] = -100  # Ignore last position

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
