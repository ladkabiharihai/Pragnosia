#!/usr/bin/env python3
"""
Dataset preprocessing script for Pragnosia training.

Supports multiple data formats:
- Plain text files (.txt)
- JSONL files with 'text' field
- Parquet files
- Hugging Face datasets

Usage:
    # Preprocess text file
    python scripts/data/preprocess.py --input data/train.txt --output data/processed

    # Preprocess JSONL
    python scripts/data/preprocess.py --input data/train.jsonl --output data/processed

    # Preprocess HuggingFace dataset
    python scripts/data/preprocess.py --hf-dataset "allenai/c4" --output data/processed

    # With tokenization
    python scripts/data/preprocess.py --input data/train.txt --output data/processed --tokenize
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import multiprocessing as mp
from functools import partial

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def read_text_file(path: str, chunk_size: int = 1000000) -> Iterator[str]:
    """Read text file in chunks."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def read_jsonl_file(path: str) -> Iterator[str]:
    """Read JSONL file line by line."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                text = item.get("text", "")
                if text:
                    yield text
            except json.JSONDecodeError:
                continue


def read_parquet_file(path: str, text_column: str = "text") -> Iterator[str]:
    """Read parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow not installed. Install with: pip install pyarrow")
        return

    table = pq.read_table(path, columns=[text_column])
    for batch in table.to_batches():
        for text in batch[text_column]:
            if text:
                yield str(text)


def create_chunks(
    texts: Iterator[str],
    max_length: int,
    stride: int,
    tokenizer=None,
) -> Iterator[Dict]:
    """Create overlapping chunks from texts."""
    current_tokens = []

    for text in texts:
        # Tokenize
        if tokenizer is not None:
            tokens = tokenizer.encode(text)
        else:
            # Simple whitespace tokenization for testing
            tokens = [hash(w) % 32000 for w in text.split()]

        current_tokens.extend(tokens)

        # Yield chunks when we have enough
        while len(current_tokens) >= max_length:
            chunk = current_tokens[:max_length]
            yield {
                "input_ids": chunk,
                "length": len(chunk),
            }
            current_tokens = current_tokens[stride:]

    # Yield remaining if substantial
    if len(current_tokens) >= max_length // 2:
        # Pad to max_length
        padding = [0] * (max_length - len(current_tokens))
        chunk = current_tokens + padding
        yield {
            "input_ids": chunk,
            "length": len(current_tokens),
        }


def process_file(
    input_path: str,
    output_dir: str,
    max_length: int = 512,
    stride: int = 256,
    shard_size: int = 10000,
    tokenizer=None,
) -> Dict:
    """Process a single file."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine file type and read
    if input_path.suffix == ".txt":
        texts = read_text_file(str(input_path))
    elif input_path.suffix == ".jsonl":
        texts = read_jsonl_file(str(input_path))
    elif input_path.suffix == ".parquet":
        texts = read_parquet_file(str(input_path))
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Create chunks
    chunks = create_chunks(texts, max_length, stride, tokenizer)

    # Write shards
    shard_idx = 0
    samples_in_shard = []
    total_samples = 0

    for chunk in tqdm(chunks, desc=f"Processing {input_path.name}"):
        samples_in_shard.append(chunk)
        total_samples += 1

        if len(samples_in_shard) >= shard_size:
            shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
            torch.save(samples_in_shard, shard_path)
            logger.info(f"Saved shard {shard_idx} with {len(samples_in_shard)} samples")
            samples_in_shard = []
            shard_idx += 1

    # Save remaining samples
    if samples_in_shard:
        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(samples_in_shard, shard_path)
        logger.info(f"Saved shard {shard_idx} with {len(samples_in_shard)} samples")
        shard_idx += 1

    # Save metadata
    metadata = {
        "source": str(input_path),
        "total_samples": total_samples,
        "num_shards": shard_idx,
        "max_length": max_length,
        "stride": stride,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def process_hf_dataset(
    dataset_name: str,
    output_dir: str,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    max_length: int = 512,
    stride: int = 256,
    shard_size: int = 10000,
    tokenizer=None,
) -> Dict:
    """Process HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Install with: pip install datasets")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    def text_generator():
        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break
            text = item.get(text_column, "")
            if text:
                yield text
                count += 1

    # Create chunks
    chunks = create_chunks(text_generator(), max_length, stride, tokenizer)

    # Write shards
    shard_idx = 0
    samples_in_shard = []
    total_samples = 0

    for chunk in tqdm(chunks, desc=f"Processing {dataset_name}"):
        samples_in_shard.append(chunk)
        total_samples += 1

        if len(samples_in_shard) >= shard_size:
            shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
            torch.save(samples_in_shard, shard_path)
            logger.info(f"Saved shard {shard_idx} with {len(samples_in_shard)} samples")
            samples_in_shard = []
            shard_idx += 1

    # Save remaining
    if samples_in_shard:
        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(samples_in_shard, shard_path)
        shard_idx += 1

    # Save metadata
    metadata = {
        "source": dataset_name,
        "split": split,
        "total_samples": total_samples,
        "num_shards": shard_idx,
        "max_length": max_length,
        "stride": stride,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


class ShardedDataset(torch.utils.data.Dataset):
    """Dataset that reads from preprocessed shards."""

    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)

        # Load metadata
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        # Find all shards
        self.shard_files = sorted(self.data_dir.glob("shard_*.pt"))

        # Build index
        self.samples = []
        for shard_path in tqdm(self.shard_files, desc="Loading shards"):
            shard_data = torch.load(shard_path)
            self.samples.extend(shard_data)
            if max_samples and len(self.samples) >= max_samples:
                self.samples = self.samples[:max_samples]
                break

        logger.info(f"Loaded {len(self.samples)} samples from {len(self.shard_files)} shards")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def create_mixed_dataset(
    data_dirs: List[str],
    weights: Optional[List[float]] = None,
) -> torch.utils.data.Dataset:
    """Create mixed dataset from multiple preprocessed directories."""
    datasets = [ShardedDataset(d) for d in data_dirs]

    if weights is None:
        # Equal weighting
        return torch.utils.data.ConcatDataset(datasets)
    else:
        # Weighted sampling
        from torch.utils.data import WeightedRandomSampler

        # Expand weights to sample level
        sample_weights = []
        for dataset, weight in zip(datasets, weights):
            sample_weights.extend([weight] * len(dataset))

        concat_dataset = torch.utils.data.ConcatDataset(datasets)
        return concat_dataset, sample_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for Pragnosia training")

    # Input
    parser.add_argument("--input", type=str, help="Path to input file")
    parser.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--text-column", type=str, default="text", help="Text column name")

    # Output
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    # Processing
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=256, help="Stride for overlapping chunks")
    parser.add_argument("--shard-size", type=int, default=10000, help="Samples per shard")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")

    # Tokenization
    parser.add_argument("--tokenize", action="store_true", help="Apply tokenization")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer or HF tokenizer name")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup tokenizer
    tokenizer = None
    if args.tokenize and args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            logger.info(f"Loaded tokenizer: {args.tokenizer}")
        except ImportError:
            logger.warning("transformers not installed. Using simple tokenization.")

    # Process data
    if args.input:
        metadata = process_file(
            input_path=args.input,
            output_dir=args.output,
            max_length=args.max_length,
            stride=args.stride,
            shard_size=args.shard_size,
            tokenizer=tokenizer,
        )
    elif args.hf_dataset:
        metadata = process_hf_dataset(
            dataset_name=args.hf_dataset,
            output_dir=args.output,
            split=args.split,
            text_column=args.text_column,
            max_samples=args.max_samples,
            max_length=args.max_length,
            stride=args.stride,
            shard_size=args.shard_size,
            tokenizer=tokenizer,
        )
    else:
        logger.error("No input specified. Use --input or --hf-dataset")
        sys.exit(1)

    logger.info(f"\nPreprocessing complete!")
    logger.info(f"  Total samples: {metadata['total_samples']}")
    logger.info(f"  Num shards: {metadata['num_shards']}")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
