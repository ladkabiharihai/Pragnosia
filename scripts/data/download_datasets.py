#!/usr/bin/env python3
"""
Download and prepare datasets for Pragnosia training.

Supported datasets:
- Text: C4, The Pile, RedPajama, GneissWeb
- Math: OpenWebMath, GSM8K
- Code: The Stack V2, StarCoder
- Multimodal: COCO, LAION (image-text pairs)

Usage:
    # Download C4 subset
    python scripts/data/download_datasets.py --dataset c4 --split train --max-samples 100000

    # Download OpenWebMath
    python scripts/data/download_datasets.py --dataset openwebmath --output data/math

    # Download The Stack (specific language)
    python scripts/data/download_datasets.py --dataset the-stack --language python
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    # Text datasets
    "c4": {
        "hf_name": "allenai/c4",
        "config": "en",
        "text_column": "text",
        "description": "Colossal Clean Crawled Corpus",
    },
    "pile": {
        "hf_name": "EleutherAI/pile",
        "text_column": "text",
        "description": "The Pile - diverse 800GB dataset",
    },
    "redpajama": {
        "hf_name": "togethercomputer/RedPajama-Data-1T-Sample",
        "text_column": "text",
        "description": "RedPajama - LLaMA training data replica",
    },
    "fineweb": {
        "hf_name": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "text_column": "text",
        "description": "FineWeb - high quality web text",
    },

    # Math datasets
    "openwebmath": {
        "hf_name": "open-web-math/open-web-math",
        "text_column": "text",
        "description": "OpenWebMath - mathematical web text",
    },
    "gsm8k": {
        "hf_name": "gsm8k",
        "config": "main",
        "text_column": "question",
        "description": "GSM8K - grade school math problems",
    },
    "metamath": {
        "hf_name": "meta-math/MetaMathQA",
        "text_column": "query",
        "description": "MetaMath - math reasoning dataset",
    },

    # Code datasets
    "the-stack": {
        "hf_name": "bigcode/the-stack",
        "text_column": "content",
        "description": "The Stack - code dataset",
    },
    "starcoder": {
        "hf_name": "bigcode/starcoderdata",
        "text_column": "content",
        "description": "StarCoder training data",
    },

    # Instruction datasets
    "alpaca": {
        "hf_name": "tatsu-lab/alpaca",
        "text_column": "text",
        "description": "Alpaca instruction dataset",
    },
    "dolly": {
        "hf_name": "databricks/databricks-dolly-15k",
        "text_column": "instruction",
        "description": "Dolly instruction dataset",
    },
}


def download_hf_dataset(
    dataset_name: str,
    output_dir: str,
    split: str = "train",
    max_samples: int = None,
    language: str = None,
    streaming: bool = True,
):
    """Download dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Install with: pip install datasets")
        sys.exit(1)

    dataset_config = DATASETS.get(dataset_name)
    if dataset_config is None:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {list(DATASETS.keys())}")
        sys.exit(1)

    hf_name = dataset_config["hf_name"]
    config = dataset_config.get("config")
    text_column = dataset_config["text_column"]

    # Handle language-specific datasets
    if language and dataset_name in ["the-stack", "starcoder"]:
        config = language

    logger.info(f"Downloading {dataset_name} ({hf_name})...")
    logger.info(f"  Description: {dataset_config['description']}")

    # Load dataset
    if streaming:
        if config:
            dataset = load_dataset(hf_name, config, split=split, streaming=True)
        else:
            dataset = load_dataset(hf_name, split=split, streaming=True)
    else:
        if config:
            dataset = load_dataset(hf_name, config, split=split)
        else:
            dataset = load_dataset(hf_name, split=split)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to JSONL
    output_file = output_path / f"{dataset_name}_{split}.jsonl"
    logger.info(f"Saving to {output_file}...")

    import json
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            if max_samples and count >= max_samples:
                break

            text = item.get(text_column, "")
            if not text:
                continue

            f.write(json.dumps({"text": text}) + "\n")
            count += 1

            if count % 10000 == 0:
                logger.info(f"  Processed {count} samples...")

    logger.info(f"Downloaded {count} samples to {output_file}")
    return str(output_file)


def download_multimodal_dataset(
    dataset_name: str,
    output_dir: str,
    max_samples: int = None,
):
    """Download multimodal dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == "coco":
        logger.info("Downloading COCO captions...")
        dataset = load_dataset("HuggingFaceM4/COCO", split="train", streaming=True)

        import json
        output_file = output_path / "coco_captions.jsonl"
        count = 0

        with open(output_file, "w") as f:
            for item in dataset:
                if max_samples and count >= max_samples:
                    break

                # COCO has image and sentences
                captions = item.get("sentences", [])
                if captions:
                    for caption in captions:
                        f.write(json.dumps({
                            "text": caption.get("raw", ""),
                            "image_id": item.get("imgid", ""),
                        }) + "\n")
                        count += 1

        logger.info(f"Downloaded {count} captions to {output_file}")

    elif dataset_name == "laion":
        logger.info("Downloading LAION-400M subset...")
        dataset = load_dataset("laion/laion400m", split="train", streaming=True)

        import json
        output_file = output_path / "laion_captions.jsonl"
        count = 0

        with open(output_file, "w") as f:
            for item in dataset:
                if max_samples and count >= max_samples:
                    break

                text = item.get("TEXT", "")
                if text:
                    f.write(json.dumps({
                        "text": text,
                        "url": item.get("URL", ""),
                    }) + "\n")
                    count += 1

        logger.info(f"Downloaded {count} samples to {output_file}")


def create_curriculum_config(
    datasets: list,
    weights: list,
    output_path: str,
):
    """Create curriculum learning config file."""
    import yaml

    config = {
        "curriculum": {
            "stages": [
                {
                    "name": "pretrain",
                    "datasets": datasets,
                    "weights": weights,
                    "steps": 100000,
                },
            ]
        }
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Created curriculum config: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets for Pragnosia")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()) + ["coco", "laion"],
        help="Dataset to download",
    )
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to download")
    parser.add_argument("--language", type=str, help="Language for code datasets")
    parser.add_argument("--no-streaming", action="store_true", help="Download full dataset")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset in ["coco", "laion"]:
        download_multimodal_dataset(
            dataset_name=args.dataset,
            output_dir=args.output,
            max_samples=args.max_samples,
        )
    else:
        download_hf_dataset(
            dataset_name=args.dataset,
            output_dir=args.output,
            split=args.split,
            max_samples=args.max_samples,
            language=args.language,
            streaming=not args.no_streaming,
        )


if __name__ == "__main__":
    main()
