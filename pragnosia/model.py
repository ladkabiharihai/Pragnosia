"""
Main Pragnosia model implementation.

A brain-inspired, energy-efficient multimodal LLM with:
- Sparse Mixture-of-Experts (MoE)
- Dynamic plasticity (expert growth/pruning)
- Energy-aware computation
- Multimodal support (text + vision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from pragnosia.config import PragnosiaConfig
from pragnosia.modules.transformer import PragnosiaTransformer
from pragnosia.modules.cortex import InputRouter, TextCortex, VisionCortex, OutputCortex
from pragnosia.modules.plasticity import PlasticityEngine

logger = logging.getLogger(__name__)


class Pragnosia(nn.Module):
    """
    Pragnosia: A Brain-Inspired, Energy-Efficient Multimodal LLM.

    Architecture:
    - Input Router: Modality detection and energy budget assignment
    - Text Cortex: Text tokenization and embedding
    - Vision Cortex: Image encoding (optional)
    - Transformer: Multi-layer transformer with MoE (Cognitive Cortex)
    - Output Cortex: Language modeling head

    Key features:
    - Sparse MoE: Only 1-2 experts active per token
    - Dynamic plasticity: Expert growth and pruning
    - Energy-aware gating: Adjustable compute budget
    - 4GB GPU compatible: Quantization + offloading
    """

    def __init__(self, config: PragnosiaConfig):
        super().__init__()
        self.config = config

        # Text Cortex (embeddings)
        self.text_cortex = TextCortex(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
        )

        # Input Router
        self.input_router = InputRouter(
            hidden_size=config.hidden_size,
            default_energy_budget=config.energy_budget_default,
        )

        # Main Transformer with MoE
        self.transformer = PragnosiaTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
            expert_types=config.expert_types,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            rms_norm_eps=config.rms_norm_eps,
            use_flash_attention=config.use_flash_attention,
            hidden_act=config.hidden_act,
            pad_token_id=config.pad_token_id,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

        # Output Cortex (LM head)
        self.output_cortex = OutputCortex(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            tie_weights=config.tie_word_embeddings,
            embed_tokens=self.text_cortex.embed_tokens if config.tie_word_embeddings else None,
        )

        # Vision Cortex (optional, loaded on demand)
        self._vision_cortex = None

        # Plasticity Engine
        if config.enable_plasticity:
            self.plasticity_engine = PlasticityEngine(
                growth_threshold=config.growth_threshold,
                prune_threshold=config.prune_threshold,
                max_experts=config.max_experts,
                min_experts=config.min_experts,
            )
        else:
            self.plasticity_engine = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def vision_cortex(self) -> VisionCortex:
        """Lazy-load vision cortex."""
        if self._vision_cortex is None:
            self._vision_cortex = VisionCortex(
                hidden_size=self.config.hidden_size,
                vision_hidden_size=self.config.vision_hidden_size,
                image_size=self.config.image_size,
                patch_size=self.config.patch_size,
            )
            # Move to same device as model
            device = next(self.parameters()).device
            self._vision_cortex = self._vision_cortex.to(device)
        return self._vision_cortex

    def get_input_embeddings(self) -> nn.Embedding:
        return self.text_cortex.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.text_cortex.embed_tokens = value
        self.transformer.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        energy_budget: Optional[float] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Pragnosia.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            images: Optional images [batch, channels, height, width]
            past_key_values: KV cache for generation
            labels: Target token IDs for loss computation
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            energy_budget: Override energy budget (0.0 to 1.0)
            return_dict: Whether to return dict (always True)

        Returns:
            Dict with loss, logits, past_key_values, hidden_states,
            attentions, and moe_aux_loss
        """
        # Get text embeddings
        if input_ids is not None:
            embeddings, attention_mask = self.text_cortex(input_ids, attention_mask)
        else:
            raise ValueError("input_ids must be provided")

        # Handle multimodal input (prepend image tokens)
        if images is not None:
            image_embeds = self.vision_cortex(images)
            # Prepend image embeddings to text embeddings
            embeddings = torch.cat([image_embeds, embeddings], dim=1)
            # Extend attention mask for image tokens
            batch_size = images.shape[0]
            image_mask = torch.ones(
                batch_size, image_embeds.shape[1],
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # Route through input router (determines energy budget)
        routed = self.input_router(
            embeddings,
            modality="multimodal" if images is not None else "text",
            max_energy=energy_budget,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Forward through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=routed.attention_mask,
            position_ids=routed.position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            energy_budget=routed.energy_budget,
        )

        hidden_states = transformer_outputs["last_hidden_state"]

        # Get logits and loss from output cortex
        output = self.output_cortex(hidden_states, labels)

        # Add MoE auxiliary losses to main loss
        total_loss = output.get("loss")
        if total_loss is not None:
            moe_loss = transformer_outputs["moe_aux_loss"] + transformer_outputs["moe_z_loss"]
            total_loss = total_loss + moe_loss

        return {
            "loss": total_loss,
            "logits": output["logits"],
            "past_key_values": transformer_outputs["past_key_values"],
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
            "moe_aux_loss": transformer_outputs["moe_aux_loss"],
            "moe_z_loss": transformer_outputs["moe_z_loss"],
            "energy_budget_used": routed.energy_budget,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        energy_budget: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            energy_budget: Energy budget for generation

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                # Only process last token with cache
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            outputs = self.forward(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
                energy_budget=energy_budget,
            )

            past_key_values = outputs["past_key_values"]
            next_token_logits = outputs["logits"][:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def save_pretrained(self, path: str):
        """Save model and config to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_pretrained(str(path))

        # Save model weights
        torch.save(self.state_dict(), path / "model.pt")

        logger.info(f"Model saved to {path}")

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: Optional[torch.device] = None,
    ) -> "Pragnosia":
        """
        Load model from pretrained weights.

        Args:
            path: Path to model directory
            load_in_4bit: Load with 4-bit quantization
            load_in_8bit: Load with 8-bit quantization
            device: Target device

        Returns:
            Loaded Pragnosia model
        """
        path = Path(path)

        # Load config
        config = PragnosiaConfig.from_pretrained(str(path))
        config.load_in_4bit = load_in_4bit
        config.load_in_8bit = load_in_8bit

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)

        # Apply quantization if requested
        if load_in_4bit:
            from pragnosia.utils.memory import load_model_in_4bit
            model = load_model_in_4bit(model)
        elif load_in_8bit:
            # 8-bit quantization implementation
            pass

        # Move to device
        if device is not None:
            model = model.to(device)

        logger.info(f"Model loaded from {path}")
        return model

    @classmethod
    def from_config(cls, config: PragnosiaConfig) -> "Pragnosia":
        """Create model from config."""
        return cls(config)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.text_cortex.embed_tokens.weight.numel()
        return n_params

    def get_memory_footprint(self) -> Dict[str, float]:
        """Get model memory footprint."""
        from pragnosia.utils.memory import count_parameters
        params = count_parameters(self)

        # Estimate memory (assuming float32)
        memory_gb = params["total"] * 4 / 1024**3

        return {
            "params_millions": params["total_millions"],
            "memory_gb_fp32": round(memory_gb, 2),
            "memory_gb_fp16": round(memory_gb / 2, 2),
            "memory_gb_4bit": round(memory_gb / 8, 2),
        }
