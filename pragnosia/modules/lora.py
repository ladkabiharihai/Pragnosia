"""
LoRA (Low-Rank Adaptation) for Pragnosia.

Enables parameter-efficient fine-tuning by adding low-rank
decomposition matrices to linear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple
import math


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    During training, the forward pass computes:
        y = Wx + (alpha/r) * BAx

    Where:
        W is the frozen pretrained weight
        B is the low-rank down-projection (r x in_features)
        A is the low-rank up-projection (out_features x r)
        r is the rank
        alpha is the scaling factor
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merge_weights = merge_weights
        self.merged = False

        # Pretrained weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = None

        # LoRA weights
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

            # Initialize LoRA weights
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # Freeze pretrained weight
        self.weight.requires_grad = False

    def reset_lora_parameters(self):
        """Reset LoRA parameters."""
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merge LoRA weights into pretrained weights."""
        if self.r > 0 and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from pretrained weights."""
        if self.r > 0 and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            # Apply LoRA
            # lora_A: [r, in_features], lora_B: [out_features, r]
            # x @ lora_A.T -> [..., r], then @ lora_B.T -> [..., out_features]
            result = F.linear(x, self.weight, self.bias)
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)  # [..., r]
            lora_out = F.linear(lora_out, self.lora_B) * self.scaling  # [..., out_features]
            return result + lora_out
        else:
            return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> "LoRALinear":
        """Create LoRALinear from existing nn.Linear."""
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        lora_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora_linear.bias = nn.Parameter(linear.bias.data.clone())
        return lora_linear


class LoRAConfig:
    """Configuration for LoRA adaptation."""

    def __init__(
        self,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        modules_to_save: Optional[List[str]] = None,
        merge_weights: bool = False,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Default target modules for attention
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.modules_to_save = modules_to_save or []
        self.merge_weights = merge_weights


def find_linear_modules(
    model: nn.Module,
    target_modules: List[str],
) -> Dict[str, nn.Linear]:
    """Find all linear modules matching target names."""
    linear_modules = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if any target module name is in the full module name
            for target in target_modules:
                if target in name:
                    linear_modules[name] = module
                    break

    return linear_modules


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """
    Apply LoRA to a model.

    Args:
        model: The model to apply LoRA to
        config: LoRA configuration

    Returns:
        Model with LoRA layers
    """
    # Find target modules
    linear_modules = find_linear_modules(model, config.target_modules)

    # Replace linear modules with LoRA versions
    for name, linear in linear_modules.items():
        # Split name to get parent and child
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name

        # Create LoRA layer
        lora_linear = LoRALinear.from_linear(
            linear,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        # Replace module
        setattr(parent, child_name, lora_linear)

    # Freeze all parameters except LoRA and specified modules
    for name, param in model.named_parameters():
        # Check if this is a LoRA parameter
        is_lora = "lora_A" in name or "lora_B" in name

        # Check if in modules_to_save
        in_save_modules = any(m in name for m in config.modules_to_save)

        if not is_lora and not in_save_modules:
            param.requires_grad = False

    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get state dict containing only LoRA parameters."""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param.data
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Load LoRA parameters from state dict."""
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into base weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights from base weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total_params = 0
    trainable_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if "lora_A" in name or "lora_B" in name:
            lora_params += param.numel()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora_params": lora_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }
