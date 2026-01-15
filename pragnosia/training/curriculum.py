"""
Curriculum training for Pragnosia.

Implements curriculum learning strategies:
- Data difficulty scheduling
- Multi-dataset mixing with dynamic weights
- Progressive sequence length
- Domain-specific curriculum stages
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import json
import yaml

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str
    datasets: List[str]
    weights: List[float]
    steps: int
    seq_length: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    energy_budget: float = 1.0

    def __post_init__(self):
        if len(self.datasets) != len(self.weights):
            raise ValueError("datasets and weights must have same length")
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]


@dataclass
class CurriculumConfig:
    """Configuration for curriculum training."""
    stages: List[CurriculumStage] = field(default_factory=list)
    difficulty_metric: str = "length"  # length, perplexity, complexity
    progressive_seq_length: bool = False
    min_seq_length: int = 128
    max_seq_length: int = 2048
    seq_length_warmup_steps: int = 10000

    @classmethod
    def from_yaml(cls, path: str) -> "CurriculumConfig":
        """Load curriculum config from YAML."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        stages = []
        for stage_data in data.get("stages", []):
            stages.append(CurriculumStage(**stage_data))

        return cls(
            stages=stages,
            difficulty_metric=data.get("difficulty_metric", "length"),
            progressive_seq_length=data.get("progressive_seq_length", False),
            min_seq_length=data.get("min_seq_length", 128),
            max_seq_length=data.get("max_seq_length", 2048),
            seq_length_warmup_steps=data.get("seq_length_warmup_steps", 10000),
        )

    def to_yaml(self, path: str):
        """Save curriculum config to YAML."""
        data = {
            "stages": [
                {
                    "name": s.name,
                    "datasets": s.datasets,
                    "weights": s.weights,
                    "steps": s.steps,
                    "seq_length": s.seq_length,
                    "batch_size": s.batch_size,
                    "learning_rate": s.learning_rate,
                    "energy_budget": s.energy_budget,
                }
                for s in self.stages
            ],
            "difficulty_metric": self.difficulty_metric,
            "progressive_seq_length": self.progressive_seq_length,
            "min_seq_length": self.min_seq_length,
            "max_seq_length": self.max_seq_length,
            "seq_length_warmup_steps": self.seq_length_warmup_steps,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


class DifficultyScorer:
    """Score sample difficulty for curriculum learning."""

    def __init__(self, metric: str = "length"):
        self.metric = metric

    def score(self, sample: Dict[str, torch.Tensor]) -> float:
        """Score sample difficulty (higher = harder)."""
        if self.metric == "length":
            return len(sample["input_ids"])
        elif self.metric == "complexity":
            # Simple complexity: number of unique tokens
            return len(set(sample["input_ids"].tolist()))
        else:
            return 0.0


class CurriculumDataset(Dataset):
    """Dataset with curriculum-aware sampling."""

    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float],
        difficulty_scorer: Optional[DifficultyScorer] = None,
        sort_by_difficulty: bool = False,
    ):
        self.datasets = datasets
        self.weights = weights
        self.difficulty_scorer = difficulty_scorer

        # Build combined index
        self.samples = []
        for dataset_idx, dataset in enumerate(datasets):
            for sample_idx in range(len(dataset)):
                self.samples.append((dataset_idx, sample_idx))

        # Compute sample weights for weighted sampling
        self.sample_weights = []
        for dataset_idx, _ in self.samples:
            self.sample_weights.append(weights[dataset_idx])

        # Sort by difficulty if requested
        if sort_by_difficulty and difficulty_scorer:
            self._sort_by_difficulty()

    def _sort_by_difficulty(self):
        """Sort samples by difficulty."""
        scored_samples = []
        for dataset_idx, sample_idx in self.samples:
            sample = self.datasets[dataset_idx][sample_idx]
            score = self.difficulty_scorer.score(sample)
            scored_samples.append((dataset_idx, sample_idx, score))

        scored_samples.sort(key=lambda x: x[2])
        self.samples = [(d, s) for d, s, _ in scored_samples]

        # Recalculate weights
        self.sample_weights = []
        for dataset_idx, _ in self.samples:
            self.sample_weights.append(self.weights[dataset_idx])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_idx, sample_idx = self.samples[idx]
        return self.datasets[dataset_idx][sample_idx]

    def get_sampler(self) -> WeightedRandomSampler:
        """Get weighted random sampler for this dataset."""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.samples),
            replacement=True,
        )


class CurriculumScheduler:
    """Manages curriculum training schedule."""

    def __init__(
        self,
        config: CurriculumConfig,
        datasets: Dict[str, Dataset],
        collator: Callable,
        batch_size: int = 8,
    ):
        self.config = config
        self.datasets = datasets
        self.collator = collator
        self.batch_size = batch_size

        self.current_stage_idx = 0
        self.steps_in_stage = 0
        self.total_steps = 0

        self.difficulty_scorer = DifficultyScorer(config.difficulty_metric)

    @property
    def current_stage(self) -> Optional[CurriculumStage]:
        """Get current curriculum stage."""
        if self.current_stage_idx < len(self.config.stages):
            return self.config.stages[self.current_stage_idx]
        return None

    def get_seq_length(self) -> int:
        """Get current sequence length (for progressive training)."""
        if not self.config.progressive_seq_length:
            if self.current_stage and self.current_stage.seq_length:
                return self.current_stage.seq_length
            return self.config.max_seq_length

        # Progressive increase
        progress = min(1.0, self.total_steps / self.config.seq_length_warmup_steps)
        seq_length = int(
            self.config.min_seq_length +
            (self.config.max_seq_length - self.config.min_seq_length) * progress
        )
        return seq_length

    def get_energy_budget(self) -> float:
        """Get current energy budget."""
        if self.current_stage:
            return self.current_stage.energy_budget
        return 1.0

    def get_learning_rate_multiplier(self) -> float:
        """Get learning rate multiplier for current stage."""
        if self.current_stage and self.current_stage.learning_rate:
            return 1.0  # Use stage-specific LR
        return 1.0

    def get_dataloader(self) -> Optional[DataLoader]:
        """Get dataloader for current stage."""
        stage = self.current_stage
        if stage is None:
            return None

        # Get datasets for this stage
        stage_datasets = []
        for dataset_name in stage.datasets:
            if dataset_name in self.datasets:
                stage_datasets.append(self.datasets[dataset_name])
            else:
                logger.warning(f"Dataset not found: {dataset_name}")

        if not stage_datasets:
            return None

        # Create curriculum dataset
        curriculum_dataset = CurriculumDataset(
            datasets=stage_datasets,
            weights=stage.weights,
            difficulty_scorer=self.difficulty_scorer,
        )

        # Create dataloader with weighted sampler
        batch_size = stage.batch_size or self.batch_size
        sampler = curriculum_dataset.get_sampler()

        return DataLoader(
            curriculum_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=4,
            pin_memory=True,
        )

    def step(self) -> bool:
        """
        Advance curriculum by one step.

        Returns True if stage changed.
        """
        self.steps_in_stage += 1
        self.total_steps += 1

        # Check if we should advance to next stage
        if self.current_stage and self.steps_in_stage >= self.current_stage.steps:
            self.current_stage_idx += 1
            self.steps_in_stage = 0
            logger.info(f"Advancing to curriculum stage {self.current_stage_idx}")
            return True

        return False

    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_stage_idx >= len(self.config.stages)

    def state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            "current_stage_idx": self.current_stage_idx,
            "steps_in_stage": self.steps_in_stage,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.current_stage_idx = state["current_stage_idx"]
        self.steps_in_stage = state["steps_in_stage"]
        self.total_steps = state["total_steps"]


def create_default_curriculum() -> CurriculumConfig:
    """Create default curriculum for Pragnosia training."""
    return CurriculumConfig(
        stages=[
            # Stage 1: Simple text, short sequences
            CurriculumStage(
                name="warmup",
                datasets=["text_simple"],
                weights=[1.0],
                steps=10000,
                seq_length=256,
                energy_budget=0.5,
            ),
            # Stage 2: Mixed text
            CurriculumStage(
                name="pretrain_text",
                datasets=["text_simple", "text_complex"],
                weights=[0.7, 0.3],
                steps=50000,
                seq_length=512,
                energy_budget=0.8,
            ),
            # Stage 3: Add code and math
            CurriculumStage(
                name="pretrain_mixed",
                datasets=["text_simple", "text_complex", "code", "math"],
                weights=[0.4, 0.3, 0.2, 0.1],
                steps=100000,
                seq_length=1024,
                energy_budget=1.0,
            ),
            # Stage 4: Full training
            CurriculumStage(
                name="full_training",
                datasets=["text_simple", "text_complex", "code", "math", "instructions"],
                weights=[0.3, 0.25, 0.2, 0.15, 0.1],
                steps=200000,
                seq_length=2048,
                energy_budget=1.0,
            ),
        ],
        progressive_seq_length=True,
        min_seq_length=128,
        max_seq_length=2048,
        seq_length_warmup_steps=50000,
    )


def create_instruction_curriculum() -> CurriculumConfig:
    """Create curriculum for instruction fine-tuning."""
    return CurriculumConfig(
        stages=[
            # Stage 1: Simple instructions
            CurriculumStage(
                name="simple_instructions",
                datasets=["alpaca"],
                weights=[1.0],
                steps=5000,
                seq_length=512,
            ),
            # Stage 2: Mixed instructions
            CurriculumStage(
                name="mixed_instructions",
                datasets=["alpaca", "dolly"],
                weights=[0.5, 0.5],
                steps=10000,
                seq_length=1024,
            ),
            # Stage 3: Complex reasoning
            CurriculumStage(
                name="complex_reasoning",
                datasets=["alpaca", "dolly", "reasoning"],
                weights=[0.3, 0.3, 0.4],
                steps=20000,
                seq_length=2048,
            ),
        ],
    )
