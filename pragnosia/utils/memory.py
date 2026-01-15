"""
Memory optimization utilities for Pragnosia.

Enables training and inference on 4GB GPUs through:
- 4-bit/8-bit quantization
- Gradient checkpointing
- CPU offloading
- Layer-wise streaming
"""

import gc
import torch
import torch.nn as nn
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get current GPU memory statistics.

    Returns:
        Dict with allocated, reserved, and free memory in GB
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3

    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free = total - reserved

    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
    }


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Returns:
        Dict with total, trainable, and frozen parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_millions": round(total / 1e6, 2),
        "trainable_millions": round(trainable / 1e6, 2),
    }


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing for memory savings.

    Trades compute for memory by recomputing activations during backward pass.
    ~50% memory savings at ~30% compute overhead.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing via model method")
        return

    # Manual enablement for transformer layers
    for module in model.modules():
        if hasattr(module, "use_gradient_checkpointing"):
            module.use_gradient_checkpointing = True

    # For PragnosiaTransformer
    if hasattr(model, "transformer"):
        model.transformer.use_gradient_checkpointing = True

    logger.info("Enabled gradient checkpointing")


def load_model_in_4bit(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.float16,
    quant_type: str = "nf4",
) -> nn.Module:
    """
    Quantize model to 4-bit using bitsandbytes.

    Args:
        model: Model to quantize
        compute_dtype: Dtype for computations (float16 or bfloat16)
        quant_type: Quantization type ("nf4" or "fp4")

    Returns:
        Quantized model
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        logger.warning("bitsandbytes not installed. Skipping 4-bit quantization.")
        return model

    def replace_linear_with_4bit(module: nn.Module, name: str = ""):
        """Recursively replace Linear layers with 4-bit versions."""
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear):
                # Create 4-bit linear layer
                new_layer = Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                )

                # Copy weights (will be quantized)
                new_layer.weight = bnb.nn.Params4bit(
                    child.weight.data,
                    requires_grad=False,
                    quant_type=quant_type,
                )
                if child.bias is not None:
                    new_layer.bias = nn.Parameter(child.bias.data)

                setattr(module, child_name, new_layer)
                logger.debug(f"Quantized {full_name} to 4-bit")
            else:
                replace_linear_with_4bit(child, full_name)

    replace_linear_with_4bit(model)
    logger.info(f"Model quantized to 4-bit ({quant_type})")
    return model


def setup_cpu_offload(
    model: nn.Module,
    device: torch.device = torch.device("cuda:0"),
) -> nn.Module:
    """
    Set up CPU offloading for large models.

    Keeps only current layer on GPU, offloads rest to CPU.

    Args:
        model: Model to set up offloading for
        device: Target GPU device

    Returns:
        Model with offloading hooks
    """
    try:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
    except ImportError:
        logger.warning("accelerate not installed. Skipping CPU offload setup.")
        return model.to(device)

    # Infer device map
    max_memory = get_balanced_memory(
        model,
        max_memory={0: "3GB", "cpu": "16GB"},
        no_split_module_classes=["PragnosiaBlock"],
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["PragnosiaBlock"],
    )

    model = dispatch_model(model, device_map=device_map)
    logger.info(f"Model distributed across devices: {set(device_map.values())}")

    return model


def optimize_model_memory(
    model: nn.Module,
    use_4bit: bool = True,
    use_gradient_checkpointing: bool = True,
    use_cpu_offload: bool = False,
    device: torch.device = torch.device("cuda:0"),
) -> nn.Module:
    """
    Apply all memory optimizations to model.

    Args:
        model: Model to optimize
        use_4bit: Enable 4-bit quantization
        use_gradient_checkpointing: Enable gradient checkpointing
        use_cpu_offload: Enable CPU offloading
        device: Target device

    Returns:
        Optimized model
    """
    initial_mem = get_memory_stats(device)
    logger.info(f"Initial memory: {initial_mem}")

    if use_gradient_checkpointing:
        enable_gradient_checkpointing(model)

    if use_4bit:
        model = load_model_in_4bit(model)

    if use_cpu_offload:
        model = setup_cpu_offload(model, device)
    else:
        model = model.to(device)

    clear_memory()
    final_mem = get_memory_stats(device)
    logger.info(f"Final memory: {final_mem}")

    params = count_parameters(model)
    logger.info(f"Model parameters: {params['total_millions']}M total, "
                f"{params['trainable_millions']}M trainable")

    return model


class LayerWiseInference:
    """
    Layer-wise inference for extremely large models.

    Loads one layer at a time from disk, enabling inference
    on models that don't fit in GPU memory.

    Based on AirLLM approach.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.float16,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.config = None
        self.embeddings = None
        self.norm = None
        self.lm_head = None

    def load_config(self):
        """Load model configuration."""
        from pragnosia.config import PragnosiaConfig
        self.config = PragnosiaConfig.from_pretrained(self.model_path)
        return self.config

    def load_embeddings(self):
        """Load and keep embeddings in memory."""
        # Implementation depends on checkpoint format
        pass

    def stream_layer(self, layer_idx: int) -> nn.Module:
        """
        Load a single layer from disk to GPU.

        Args:
            layer_idx: Index of layer to load

        Returns:
            Layer module on GPU
        """
        # Implementation depends on checkpoint format
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with layer streaming.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Output logits
        """
        if self.config is None:
            self.load_config()

        # Get embeddings
        hidden_states = self.embeddings(input_ids)

        # Stream through layers
        for layer_idx in range(self.config.num_hidden_layers):
            # Load layer to GPU
            layer = self.stream_layer(layer_idx)
            layer = layer.to(self.device, dtype=self.dtype)

            # Forward pass
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

            # Offload layer back to CPU
            layer = layer.to("cpu")
            clear_memory()

        # Final norm and head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
