"""Hippocampus implementation for fast, episodic memory storage."""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from collections import deque
import random


class Hippocampus(nn.Module):
    """
    Fast episodic memory system inspired by hippocampus.

    Stores recent experiences for quick retrieval and replay during consolidation.
    Capacity: ~50MB, implements FIFO with priority-based sampling.
    """

    def __init__(
        self,
        capacity: int = 10000,
        batch_size: int = 32,
        consolidation_threshold: float = 0.7,
    ):
        super().__init__()
        self.capacity = capacity
        self.batch_size = batch_size
        self.consolidation_threshold = consolidation_threshold

        # Memory buffers
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

        # Statistics
        self.register_buffer("total_stored", torch.zeros(1))
        self.register_buffer("total_consolidated", torch.zeros(1))
        self.register_buffer("average_priority", torch.zeros(1))

    def store(
        self,
        hidden_states: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        surprise: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Store new experience in hippocampus.

        Args:
            hidden_states: Hidden representations (batch, seq_len, hidden_size)
            targets: Target predictions
            loss: Loss value for prioritization
            surprise: Surprise signal for prioritization
            metadata: Additional context
        """
        batch_size = hidden_states.shape[0]

        # Compute priority (higher = more important to retain)
        priority = self._compute_priority(loss, surprise)

        for i in range(batch_size):
            experience = {
                "hidden_states": hidden_states[i].detach().cpu(),
                "targets": targets[i].detach().cpu() if targets is not None else None,
                "loss": loss[i].detach().cpu() if loss is not None else None,
                "surprise": surprise[i].detach().cpu() if surprise is not None else None,
                "metadata": metadata,
                "replay_count": 0,
            }

            self.memory.append(experience)
            self.priorities.append(priority[i].item() if priority is not None else 1.0)

        self.total_stored += batch_size

        # Update average priority
        if len(self.priorities) > 0:
            self.average_priority = torch.tensor(sum(self.priorities) / len(self.priorities))

    def sample(
        self,
        batch_size: Optional[int] = None,
        prioritized: bool = True,
    ) -> Optional[dict]:
        """
        Sample experiences from hippocampus for replay.

        Args:
            batch_size: Number of experiences to sample
            prioritized: Whether to use priority-based sampling

        Returns:
            batch: Dictionary containing batched experiences
        """
        if len(self.memory) == 0:
            return None

        batch_size = batch_size or self.batch_size
        batch_size = min(batch_size, len(self.memory))

        # Sample experiences
        if prioritized and len(self.priorities) > 0:
            # Priority-based sampling
            priorities = torch.tensor(list(self.priorities))
            probs = priorities / priorities.sum()
            indices = torch.multinomial(probs, batch_size, replacement=False)
        else:
            # Uniform sampling
            indices = torch.randperm(len(self.memory))[:batch_size]

        # Gather experiences
        experiences = [self.memory[i] for i in indices]

        # Increment replay counts
        for i in indices:
            self.memory[i]["replay_count"] += 1

        # Batch experiences
        batch = self._batch_experiences(experiences)

        return batch

    def _batch_experiences(self, experiences: List[dict]) -> dict:
        """Batch list of experiences into tensors."""
        batch = {
            "hidden_states": torch.stack([exp["hidden_states"] for exp in experiences]),
            "targets": torch.stack([exp["targets"] for exp in experiences])
            if experiences[0]["targets"] is not None else None,
            "loss": torch.stack([exp["loss"] for exp in experiences])
            if experiences[0]["loss"] is not None else None,
            "surprise": torch.stack([exp["surprise"] for exp in experiences])
            if experiences[0]["surprise"] is not None else None,
            "replay_count": torch.tensor([exp["replay_count"] for exp in experiences]),
        }
        return batch

    def _compute_priority(
        self,
        loss: Optional[torch.Tensor],
        surprise: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Compute storage priority based on loss and surprise.

        Higher priority = more important to retain and replay.
        """
        if loss is None and surprise is None:
            return None

        # Handle scalar or batch tensors
        if loss is not None and loss.dim() == 0:
            # Scalar loss - use surprise shape or default to 1
            batch_size = surprise.shape[0] if surprise is not None else 1
        elif surprise is not None and surprise.dim() > 0:
            batch_size = surprise.shape[0]
        elif loss is not None and loss.dim() > 0:
            batch_size = loss.shape[0]
        else:
            batch_size = 1

        # Determine device from inputs
        device = loss.device if loss is not None else (surprise.device if surprise is not None else torch.device("cpu"))
        priority = torch.ones(batch_size, device=device)

        if loss is not None:
            # Higher loss = higher priority
            loss_val = loss.detach()
            if loss_val.dim() == 0:
                priority = priority + loss_val.item()
            else:
                priority = priority + loss_val

        if surprise is not None:
            # Higher surprise = higher priority
            surprise_val = surprise.detach()
            if surprise_val.dim() > 0:
                priority = priority + surprise_val
            else:
                priority = priority + surprise_val.item()

        return priority

    def get_consolidation_candidates(
        self,
        threshold: Optional[float] = None,
    ) -> Optional[List[dict]]:
        """
        Get experiences ready for consolidation to neocortex.

        Experiences are candidates if they have:
        - High replay count (well-rehearsed)
        - Low error (successfully learned)
        """
        threshold = threshold or self.consolidation_threshold

        if len(self.memory) == 0:
            return None

        candidates = []

        for exp in self.memory:
            # Check if experience is well-learned
            is_rehearsed = exp["replay_count"] >= 3
            is_learned = (exp["loss"] < threshold) if exp["loss"] is not None else True

            if is_rehearsed and is_learned:
                candidates.append(exp)

        return candidates if len(candidates) > 0 else None

    def clear_consolidated(self, experiences: List[dict]):
        """Remove consolidated experiences from hippocampus."""
        # Mark as consolidated
        self.total_consolidated += len(experiences)

        # Remove from memory (in practice, they'll be replaced by new experiences)
        # FIFO queue handles this automatically

    def get_size(self) -> int:
        """Get current number of stored experiences."""
        return len(self.memory)

    def get_statistics(self) -> dict:
        """Get memory statistics."""
        return {
            "size": len(self.memory),
            "capacity": self.capacity,
            "utilization": len(self.memory) / self.capacity,
            "total_stored": self.total_stored.item(),
            "total_consolidated": self.total_consolidated.item(),
            "average_priority": self.average_priority.item(),
        }

    def get_memory_size_mb(self) -> float:
        """Estimate memory footprint in MB."""
        if len(self.memory) == 0:
            return 0.0

        # Estimate based on first experience
        exp = self.memory[0]
        exp_size = (
            exp["hidden_states"].numel() * exp["hidden_states"].element_size()
        )

        if exp["targets"] is not None:
            exp_size += exp["targets"].numel() * exp["targets"].element_size()

        total_size = exp_size * len(self.memory)
        return total_size / (1024 ** 2)
