"""Configuration class for Pragnosia model."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PragnosiaConfig:
    """Configuration for Pragnosia model architecture."""

    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_experts: int = 8
    num_active_experts: int = 2
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048

    # Router configuration
    router_size: int = 256
    router_learning_rate: float = 0.001
    hebbian_learning_rate: float = 0.01
    lateral_inhibition_strength: float = 0.1
    routing_threshold: float = 0.1

    # Expert configuration
    expert_size: int = 768
    expert_dropout: float = 0.1

    # Memory systems
    hippocampus_capacity: int = 10000
    hippocampus_batch_size: int = 32
    neocortex_capacity: int = 50000
    replay_ratio: float = 0.1
    consolidation_threshold: float = 0.7

    # Intrinsic learning weights
    alpha_surprise: float = 0.3
    beta_temporal: float = 0.25
    gamma_disagreement: float = 0.25
    delta_compression: float = 0.2
    epsilon_entropy: float = 0.3  # NEW: Sequence entropy (critical for local learning)
    entropy_window: int = 20  # Rolling window for entropy computation
    entropy_min_threshold: float = 2.0  # Minimum entropy target (bits)
    learn_intrinsic_weights: bool = True
    use_neuromodulation: bool = True  # Adaptive intrinsic learning strength

    # Homeostatic regulation (reduced to prevent dominance over intrinsic learning)
    lambda_uncertainty: float = 0.05
    lambda_surprise: float = 0.05
    lambda_churn: float = 0.02
    target_entropy: float = 2.0

    # Neuroplasticity phases (as fraction of total training)
    exploration_end: float = 0.3
    stabilization_end: float = 0.7

    # Safety bounds
    min_active_params: float = 0.3
    max_active_params: float = 1.0  # Allow full activation at start (exploration phase)
    min_expert_entropy: float = 2.0
    max_growth_rate: float = 0.01
    max_pruning_rate: float = 0.01

    # Training configuration
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    warmup_steps: int = 1000

    # Micro-sleep consolidation
    microsleep_samples: int = 2
    consolidation_stream_priority: int = -1
    intensive_consolidation_interval: int = 1000

    # GPU memory management
    max_gpu_memory_gb: float = 4.0
    offload_to_cpu: bool = True
    expert_load_async: bool = True

    # Perceptual distillation
    use_perceptual_distillation: bool = True
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    vision_encoder: str = "openai/clip-vit-base-patch32"
    audio_encoder: str = "openai/whisper-tiny"

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.exploration_end < 1, "exploration_end must be in (0, 1)"
        assert self.exploration_end < self.stabilization_end < 1, \
            "stabilization_end must be after exploration_end and before 1"
        assert self.num_active_experts <= self.num_experts, \
            "num_active_experts must be <= num_experts"
        assert 0 < self.min_active_params < self.max_active_params <= 1, \
            "active params bounds must satisfy 0 < min < max <= 1"
