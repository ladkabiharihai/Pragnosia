"""
Configuration system for Pragnosia model.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import yaml
from pathlib import Path


@dataclass
class PragnosiaConfig:
    """Configuration for Pragnosia model."""

    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA
    max_position_embeddings: int = 4096

    # MoE configuration
    num_experts: int = 8
    num_experts_per_token: int = 2  # Top-K routing
    expert_capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001

    # Expert types for cognitive cortex
    expert_types: list = field(default_factory=lambda: [
        "language", "reasoning", "memory", "planning",
        "language", "reasoning", "memory", "planning"  # duplicates for capacity
    ])

    # Attention configuration
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_flash_attention: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    # Normalization
    rms_norm_eps: float = 1e-6

    # Activation
    hidden_act: str = "silu"

    # Energy-aware gating
    energy_budget_default: float = 1.0  # 1.0 = full compute, 0.5 = half
    min_energy_threshold: float = 0.1
    enable_energy_gating: bool = True

    # Plasticity configuration
    enable_plasticity: bool = True
    growth_threshold: float = 0.8  # routing entropy threshold for growth
    prune_threshold: float = 0.05  # activation frequency threshold for pruning
    max_experts: int = 16
    min_experts: int = 4

    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # Multimodal
    vision_encoder: str = "openai/clip-vit-base-patch32"
    vision_hidden_size: int = 768
    vision_projection_dim: int = 2048
    image_size: int = 224
    patch_size: int = 32
    num_image_tokens: int = 49  # (224/32)^2

    # Tokenizer
    tokenizer_path: Optional[str] = None
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Training
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    @classmethod
    def from_yaml(cls, path: str) -> "PragnosiaConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.__dict__.copy()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def from_pretrained(cls, path: str) -> "PragnosiaConfig":
        """Load configuration from pretrained model directory."""
        config_path = Path(path) / "config.yaml"
        if config_path.exists():
            return cls.from_yaml(str(config_path))
        raise FileNotFoundError(f"Config not found at {config_path}")

    def save_pretrained(self, path: str) -> None:
        """Save configuration to model directory."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.to_yaml(str(Path(path) / "config.yaml"))

    # Preset configurations
    @classmethod
    def tiny(cls) -> "PragnosiaConfig":
        """Tiny config for testing (~125M params)."""
        return cls(
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_token=1,
        )

    @classmethod
    def small(cls) -> "PragnosiaConfig":
        """Small config (~350M active, ~1B total with MoE)."""
        return cls(
            hidden_size=1024,
            intermediate_size=2752,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_token=2,
        )

    @classmethod
    def base(cls) -> "PragnosiaConfig":
        """Base config (~1B active, ~3B total with MoE)."""
        return cls(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_token=2,
        )

    @classmethod
    def large(cls) -> "PragnosiaConfig":
        """Large config (~3B active, ~7B total with MoE)."""
        return cls(
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=32,
            num_attention_heads=24,
            num_key_value_heads=8,
            num_experts=16,
            num_experts_per_token=2,
        )

    @classmethod
    def xl(cls) -> "PragnosiaConfig":
        """XL config (~7B active, ~14B total with MoE)."""
        return cls(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=16,
            num_experts_per_token=2,
            max_position_embeddings=8192,
        )

    @classmethod
    def config_3b(cls) -> "PragnosiaConfig":
        """3B total parameters config (optimized for 4GB GPU with offload)."""
        return cls(
            hidden_size=2560,
            intermediate_size=6912,
            num_hidden_layers=28,
            num_attention_heads=20,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_token=2,
            use_gradient_checkpointing=True,
            use_cpu_offload=True,
        )

    @classmethod
    def config_7b(cls) -> "PragnosiaConfig":
        """7B total parameters config (requires offload or quantization)."""
        return cls(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=8,
            num_experts_per_token=2,
            use_gradient_checkpointing=True,
            use_cpu_offload=True,
            load_in_4bit=True,
        )

    @classmethod
    def efficient_4gb(cls) -> "PragnosiaConfig":
        """Config optimized for inference on 4GB GPU."""
        return cls(
            hidden_size=1536,
            intermediate_size=4096,
            num_hidden_layers=20,
            num_attention_heads=12,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_token=1,  # Single expert for efficiency
            use_gradient_checkpointing=True,
            use_flash_attention=True,
            enable_energy_gating=True,
            energy_budget_default=0.7,  # Reduced compute
        )
