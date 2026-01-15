"""
Pragnosia neural network modules.

Brain-inspired components:
- Input Router: Modality detection and energy budget assignment
- Text Cortex: Text encoding (tokenizer + embeddings)
- Vision Cortex: Visual encoding (ViT/CNN)
- Thalamus Router: Token-to-expert routing with energy gating
- Cognitive Cortex: MoE experts (Language, Reasoning, Memory, Planning)
- Plasticity Engine: Dynamic expert growth/pruning
- Output Cortex: Output projection and decoding
"""

from pragnosia.modules.attention import PragnosiaAttention
from pragnosia.modules.mlp import PragnosiaMLP
from pragnosia.modules.experts import Expert, CognitiveCortex
from pragnosia.modules.thalamus import ThalamusRouter
from pragnosia.modules.normalization import RMSNorm
from pragnosia.modules.embeddings import RotaryEmbedding
from pragnosia.modules.transformer import PragnosiaBlock, PragnosiaTransformer
from pragnosia.modules.cortex import InputRouter, TextCortex, VisionCortex, OutputCortex
from pragnosia.modules.plasticity import PlasticityEngine
from pragnosia.modules.lora import (
    LoRALinear,
    LoRAConfig,
    apply_lora,
    get_lora_state_dict,
    load_lora_state_dict,
    merge_lora_weights,
    unmerge_lora_weights,
    count_lora_parameters,
)
from pragnosia.modules.memory_expert import (
    MemoryExpert,
    MemoryState,
    MemoryAugmentedTransformerBlock,
)

__all__ = [
    "PragnosiaAttention",
    "PragnosiaMLP",
    "Expert",
    "CognitiveCortex",
    "ThalamusRouter",
    "RMSNorm",
    "RotaryEmbedding",
    "PragnosiaBlock",
    "PragnosiaTransformer",
    "InputRouter",
    "TextCortex",
    "VisionCortex",
    "OutputCortex",
    "PlasticityEngine",
    "LoRALinear",
    "LoRAConfig",
    "apply_lora",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "count_lora_parameters",
    "MemoryExpert",
    "MemoryState",
    "MemoryAugmentedTransformerBlock",
]
