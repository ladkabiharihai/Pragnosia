"""Neocortex implementation for slow, structured memory consolidation."""
import torch
import torch.nn as nn
from typing import List, Optional
import random


class Neocortex(nn.Module):
    """
    Slow, structured memory system inspired by neocortex.

    Stores consolidated knowledge from hippocampus with structured organization.
    Capacity: ~100MB, implements semantic clustering and interference management.
    """

    def __init__(
        self,
        capacity: int = 50000,
        hidden_size: int = 768,
        num_clusters: int = 32,
    ):
        super().__init__()
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.num_clusters = num_clusters

        # Structured memory with semantic clustering
        self.clusters = [[] for _ in range(num_clusters)]
        self.cluster_centroids = nn.Parameter(
            torch.randn(num_clusters, hidden_size),
            requires_grad=False,
        )

        # Statistics
        self.register_buffer("total_consolidated", torch.zeros(1))
        self.register_buffer("total_retrievals", torch.zeros(1))
        self.register_buffer("cluster_sizes", torch.zeros(num_clusters))

    def consolidate(
        self,
        experiences: List[dict],
        replay_model: Optional[nn.Module] = None,
    ):
        """
        Consolidate experiences from hippocampus into structured memory.

        Args:
            experiences: List of experiences to consolidate
            replay_model: Optional model for generative replay
        """
        for exp in experiences:
            # Assign to cluster based on hidden state similarity
            hidden_state = exp["hidden_states"].mean(dim=0)  # Pool over sequence
            cluster_idx = self._assign_cluster(hidden_state)

            # Store in appropriate cluster
            if len(self.clusters[cluster_idx]) < self.capacity // self.num_clusters:
                self.clusters[cluster_idx].append(exp)
                self.cluster_sizes[cluster_idx] += 1
            else:
                # Replace oldest experience if cluster is full
                self.clusters[cluster_idx].pop(0)
                self.clusters[cluster_idx].append(exp)

            self.total_consolidated += 1

        # Update cluster centroids
        self._update_centroids()

    def _assign_cluster(self, hidden_state: torch.Tensor) -> int:
        """Assign experience to nearest cluster based on hidden state."""
        hidden_state = hidden_state.to(self.cluster_centroids.device)

        # Compute distances to all centroids
        distances = torch.cdist(
            hidden_state.unsqueeze(0),
            self.cluster_centroids,
        ).squeeze(0)

        # Assign to nearest cluster
        cluster_idx = distances.argmin().item()
        return cluster_idx

    def _update_centroids(self):
        """Update cluster centroids based on stored experiences."""
        with torch.no_grad():
            for cluster_idx in range(self.num_clusters):
                if len(self.clusters[cluster_idx]) > 0:
                    # Compute mean of all experiences in cluster
                    hidden_states = [
                        exp["hidden_states"].mean(dim=0)
                        for exp in self.clusters[cluster_idx]
                    ]
                    cluster_mean = torch.stack(hidden_states).mean(dim=0)
                    self.cluster_centroids[cluster_idx] = cluster_mean

    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 32,
    ) -> Optional[dict]:
        """
        Retrieve relevant experiences based on query.

        Args:
            query: Query hidden state (hidden_size,)
            k: Number of experiences to retrieve

        Returns:
            batch: Dictionary containing retrieved experiences
        """
        # Find relevant clusters
        query = query.to(self.cluster_centroids.device)
        distances = torch.cdist(
            query.unsqueeze(0),
            self.cluster_centroids,
        ).squeeze(0)

        # Get top-k nearest clusters
        num_clusters_to_search = min(3, self.num_clusters)
        _, cluster_indices = torch.topk(
            distances,
            num_clusters_to_search,
            largest=False,
        )

        # Gather experiences from relevant clusters
        experiences = []
        for cluster_idx in cluster_indices:
            experiences.extend(self.clusters[cluster_idx.item()])

        if len(experiences) == 0:
            return None

        # Sample k experiences
        k = min(k, len(experiences))
        sampled_experiences = random.sample(experiences, k)

        self.total_retrievals += k

        # Batch experiences
        batch = self._batch_experiences(sampled_experiences)

        return batch

    def _batch_experiences(self, experiences: List[dict]) -> dict:
        """Batch list of experiences into tensors."""
        batch = {
            "hidden_states": torch.stack([exp["hidden_states"] for exp in experiences]),
            "targets": torch.stack([exp["targets"] for exp in experiences])
            if experiences[0]["targets"] is not None else None,
        }
        return batch

    def get_size(self) -> int:
        """Get total number of stored experiences."""
        return sum(len(cluster) for cluster in self.clusters)

    def get_statistics(self) -> dict:
        """Get memory statistics."""
        sizes = [len(cluster) for cluster in self.clusters]
        return {
            "size": self.get_size(),
            "capacity": self.capacity,
            "utilization": self.get_size() / self.capacity,
            "num_clusters": self.num_clusters,
            "min_cluster_size": min(sizes) if sizes else 0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "mean_cluster_size": sum(sizes) / len(sizes) if sizes else 0,
            "total_consolidated": self.total_consolidated.item(),
            "total_retrievals": self.total_retrievals.item(),
        }

    def get_memory_size_mb(self) -> float:
        """Estimate memory footprint in MB."""
        total_size = 0

        for cluster in self.clusters:
            for exp in cluster:
                total_size += (
                    exp["hidden_states"].numel() * exp["hidden_states"].element_size()
                )
                if exp.get("targets") is not None:
                    total_size += exp["targets"].numel() * exp["targets"].element_size()

        # Add centroid size
        total_size += (
            self.cluster_centroids.numel() * self.cluster_centroids.element_size()
        )

        return total_size / (1024 ** 2)
