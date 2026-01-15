"""
Data loading utilities for Pragnosia training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Optional, Union, Iterator
from pathlib import Path
import json
import random
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.

    Supports:
    - Plain text files (.txt)
    - JSONL files with 'text' field
    - Pre-tokenized files
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
        stride: int = 256,  # Overlap between chunks
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride

        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load and tokenize data."""
        logger.info(f"Loading data from {self.data_path}")

        if self.data_path.suffix == ".txt":
            self._load_txt()
        elif self.data_path.suffix == ".jsonl":
            self._load_jsonl()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        logger.info(f"Loaded {len(self.samples)} samples")

    def _load_txt(self):
        """Load plain text file."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize entire text
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text)
        else:
            # Simple whitespace tokenization for testing
            tokens = [hash(w) % 32000 for w in text.split()]

        # Create overlapping chunks
        for i in range(0, len(tokens) - self.max_seq_length + 1, self.stride):
            chunk = tokens[i:i + self.max_seq_length]
            if len(chunk) == self.max_seq_length:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def _load_jsonl(self):
        """Load JSONL file with 'text' field."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text", "")

                if self.tokenizer is not None:
                    tokens = self.tokenizer.encode(text)
                else:
                    tokens = [hash(w) % 32000 for w in text.split()]

                # Truncate or pad
                if len(tokens) >= self.max_seq_length:
                    tokens = tokens[:self.max_seq_length]
                    self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.samples[idx]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large text corpora.

    Memory efficient - doesn't load entire dataset into memory.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through dataset with shuffling."""
        buffer = []
        rng = random.Random(self.seed)

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                # Parse line
                if self.data_path.suffix == ".jsonl":
                    item = json.loads(line)
                    text = item.get("text", "")
                else:
                    text = line.strip()

                # Tokenize
                if self.tokenizer is not None:
                    tokens = self.tokenizer.encode(text)
                else:
                    tokens = [hash(w) % 32000 for w in text.split()]

                if len(tokens) < 10:  # Skip very short samples
                    continue

                # Truncate
                tokens = tokens[:self.max_seq_length]
                tokens = torch.tensor(tokens, dtype=torch.long)

                # Add to buffer
                buffer.append({"input_ids": tokens, "labels": tokens.clone()})

                # Yield from buffer when full
                if len(buffer) >= self.shuffle_buffer:
                    rng.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

        # Yield remaining items
        if buffer:
            rng.shuffle(buffer)
            for item in buffer:
                yield item


class DataCollator:
    """
    Collator for batching samples with padding.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_seq_length: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch with padding."""
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Find max length in batch
        max_len = max(len(ids) for ids in input_ids)
        if self.max_seq_length is not None:
            max_len = min(max_len, self.max_seq_length)

        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        attention_mask = []

        for ids, lbls in zip(input_ids, labels):
            # Truncate if needed
            ids = ids[:max_len]
            lbls = lbls[:max_len]

            # Calculate padding
            pad_len = max_len - len(ids)

            # Pad
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                lbls = torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)])  # -100 = ignore
                mask = torch.cat([torch.ones(max_len - pad_len), torch.zeros(pad_len)])
            else:
                mask = torch.ones(max_len)

            padded_input_ids.append(ids)
            padded_labels.append(lbls)
            attention_mask.append(mask)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_mask).long(),
        }


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 8,
    max_seq_length: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
    streaming: bool = False,
    pad_token_id: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for training.

    Args:
        data_path: Path to data file
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_seq_length: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        streaming: Use streaming dataset (for large files)
        pad_token_id: Padding token ID

    Returns:
        DataLoader instance
    """
    if streaming:
        dataset = StreamingTextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        # Streaming datasets handle their own shuffling
        shuffle = False
        num_workers = 0  # Streaming doesn't support multiple workers well
    else:
        dataset = TextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )

    collator = DataCollator(
        pad_token_id=pad_token_id,
        max_seq_length=max_seq_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not streaming,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing training pipeline.

    Generates random token sequences.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 512,
        vocab_size: int = 32000,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Generate synthetic data
        torch.manual_seed(seed)
        self.data = torch.randint(1, vocab_size, (num_samples, seq_length))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }
