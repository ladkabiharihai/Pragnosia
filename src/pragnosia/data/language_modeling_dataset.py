"""Language modeling dataset for pre-training on plain text.

This dataset loader is designed for pre-training Pragnosia on raw text
without instruction templates. This teaches the model fundamental language
patterns before fine-tuning on specific tasks.

Key principle: Learn language first, learn tasks later.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional
import os


class LanguageModelingDataset(Dataset):
    """
    Plain text dataset for language modeling pre-training.

    No instruction templates - just raw text for next-token prediction.
    This is the foundation for all language understanding.

    Supported datasets:
    - WikiText-2: ~2M tokens, good for testing (default)
    - WikiText-103: ~100M tokens, better for real pre-training
    - OpenWebText: ~8B tokens, production-grade (requires more time)

    Usage:
        >>> dataset = LanguageModelingDataset(
        ...     tokenizer=tokenizer,
        ...     dataset_name="wikitext-2-raw-v1",
        ...     max_length=512,
        ...     split="train"
        ... )
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str = "wikitext-2-raw-v1",
        max_length: int = 512,
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer to use
            dataset_name: Name of dataset to load:
                - "wikitext-2-raw-v1" (2M tokens, fast, good for testing)
                - "wikitext-103-raw-v1" (100M tokens, better quality)
                - "openwebtext" (8B tokens, production)
            max_length: Maximum sequence length
            split: Dataset split ("train", "validation", "test")
            cache_dir: Directory to cache downloaded datasets
            max_samples: Maximum number of samples (for testing)
            stride: Stride for sliding window (default: max_length, no overlap)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length  # No overlap by default

        # Disable parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load dataset
        print(f"Loading {dataset_name} ({split} split)...")
        self.raw_data = self._load_dataset(
            dataset_name, split, cache_dir, max_samples
        )

        # Pre-tokenize all text into chunks
        print("Tokenizing dataset...")
        self.samples = self._prepare_samples()
        print(f"Created {len(self.samples)} training samples")

    def _load_dataset(self, dataset_name, split, cache_dir, max_samples):
        """Load the raw text dataset."""
        try:
            if dataset_name in ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]:
                # WikiText datasets
                dataset = load_dataset(
                    "wikitext",
                    dataset_name,
                    split=split,
                    cache_dir=cache_dir,
                )
            elif dataset_name == "openwebtext":
                # OpenWebText (much larger)
                dataset = load_dataset(
                    "openwebtext",
                    split="train",  # Only has train split
                    cache_dir=cache_dir,
                )
                # Create validation split manually
                if split == "validation":
                    total = len(dataset)
                    dataset = dataset.select(range(int(total * 0.95), total))
                elif split == "train":
                    total = len(dataset)
                    dataset = dataset.select(range(0, int(total * 0.95)))
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            return dataset

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            print("Falling back to synthetic data for testing...")
            return self._create_synthetic_data(max_samples or 100)

    def _create_synthetic_data(self, num_samples):
        """Create synthetic text data as fallback."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a high-level programming language.",
            "The capital of France is Paris.",
            "Natural language processing enables computers to understand human language.",
        ] * (num_samples // 5 + 1)

        return [{"text": text} for text in texts[:num_samples]]

    def _prepare_samples(self):
        """
        Tokenize all text and create fixed-length samples.

        Uses sliding window with configurable stride to create training samples.
        This is more efficient than tokenizing on-the-fly.
        """
        samples = []

        for item in self.raw_data:
            text = item.get("text", "").strip()

            # Skip empty lines and very short text
            if len(text) < 10:
                continue

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Create sliding window samples
            for i in range(0, len(tokens), self.stride):
                chunk = tokens[i:i + self.max_length]

                # Only use chunks that are at least 50% of max_length
                # This avoids too many short sequences at the end
                if len(chunk) >= self.max_length // 2:
                    samples.append(chunk)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            dict with:
            - input_ids: token IDs (max_length,)
            - attention_mask: attention mask (max_length,)
            - labels: same as input_ids (for next-token prediction)
        """
        tokens = self.samples[idx]

        # Pad to max_length if needed
        if len(tokens) < self.max_length:
            num_pad = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * num_pad
            attention_mask = [1] * len(self.samples[idx]) + [0] * num_pad
        else:
            attention_mask = [1] * self.max_length

        # For language modeling, labels = input_ids
        # The model learns to predict next token given previous tokens
        input_ids = tokens
        labels = tokens.copy()

        # Set padding tokens to -100 so they're ignored in loss
        labels = [
            -100 if attention_mask[i] == 0 else labels[i]
            for i in range(len(labels))
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def get_stats(self):
        """Get dataset statistics."""
        total_tokens = sum(len(s) for s in self.samples)
        avg_length = total_tokens / len(self.samples) if self.samples else 0

        return {
            "num_samples": len(self.samples),
            "total_tokens": total_tokens,
            "avg_length": avg_length,
            "max_length": self.max_length,
            "stride": self.stride,
        }
