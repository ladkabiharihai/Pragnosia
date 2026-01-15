"""
Input and Output Cortex modules for Pragnosia.

Input Cortex: Handles modality detection, tokenization, and energy budget assignment
Output Cortex: Handles output projection and multimodal decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Literal, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class RoutedInput:
    """Output from Input Router."""
    tokens: torch.Tensor  # [batch, seq_len, hidden_size]
    modality: str
    energy_budget: float
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


class InputRouter(nn.Module):
    """
    Input Router for Pragnosia.

    Responsibilities:
    - Detect input modality (text, image, audio)
    - Estimate input complexity
    - Assign energy budget based on complexity
    """

    def __init__(
        self,
        hidden_size: int,
        default_energy_budget: float = 1.0,
        complexity_estimator: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.default_energy_budget = default_energy_budget

        # Complexity estimation network
        if complexity_estimator:
            self.complexity_net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid(),
            )
        else:
            self.complexity_net = None

    def estimate_complexity(self, embeddings: torch.Tensor) -> float:
        """
        Estimate input complexity to determine energy budget.

        Higher complexity -> higher energy budget (more experts active)
        """
        if self.complexity_net is None:
            return self.default_energy_budget

        # Pool embeddings
        pooled = embeddings.mean(dim=1)  # [batch, hidden]
        complexity = self.complexity_net(pooled).mean().item()

        # Scale to energy budget (0.5 to 1.0 range)
        energy_budget = 0.5 + 0.5 * complexity
        return energy_budget

    def forward(
        self,
        embeddings: torch.Tensor,
        modality: str = "text",
        max_energy: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> RoutedInput:
        """
        Route input embeddings with energy budget.

        Args:
            embeddings: Input embeddings [batch, seq_len, hidden_size]
            modality: Input modality type
            max_energy: Maximum energy budget override
            attention_mask: Attention mask
            position_ids: Position indices

        Returns:
            RoutedInput with tokens and routing information
        """
        # Estimate complexity and energy budget
        estimated_energy = self.estimate_complexity(embeddings)

        # Apply max energy constraint if provided
        if max_energy is not None:
            energy_budget = min(estimated_energy, max_energy)
        else:
            energy_budget = estimated_energy

        return RoutedInput(
            tokens=embeddings,
            modality=modality,
            energy_budget=energy_budget,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


class TextCortex(nn.Module):
    """
    Text Cortex for Pragnosia.

    Handles text tokenization and embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 4096,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Embed text tokens.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Tuple of (embeddings, attention_mask)
        """
        embeddings = self.embed_tokens(input_ids)

        # Create attention mask from padding if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        return embeddings, attention_mask


class VisionCortex(nn.Module):
    """
    Vision Cortex for Pragnosia.

    Handles image encoding using pretrained vision encoders.
    Projects vision features to LLM embedding space.
    """

    def __init__(
        self,
        hidden_size: int,
        vision_hidden_size: int = 768,
        image_size: int = 224,
        patch_size: int = 32,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vision_hidden_size = vision_hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Simple patch embedding (can be replaced with pretrained ViT)
        self.patch_embed = nn.Conv2d(
            3, vision_hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Position embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, vision_hidden_size)
        )

        # Simple transformer encoder for vision
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_hidden_size,
            nhead=8,
            dim_feedforward=vision_hidden_size * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Projection to LLM space
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self._freeze_encoder()

        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)

    def _freeze_encoder(self):
        """Freeze vision encoder weights."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.pos_embed.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to token embeddings.

        Args:
            images: Image tensor [batch, channels, height, width]

        Returns:
            Image embeddings [batch, num_patches, hidden_size]
        """
        batch_size = images.shape[0]

        # Patch embedding: [batch, vision_hidden, h/patch, w/patch]
        x = self.patch_embed(images)

        # Flatten patches: [batch, num_patches, vision_hidden]
        x = x.flatten(2).transpose(1, 2)

        # Add position embeddings
        x = x + self.pos_embed

        # Encode with transformer
        x = self.encoder(x)

        # Project to LLM space
        x = self.proj(x)

        return x


class OutputCortex(nn.Module):
    """
    Output Cortex for Pragnosia.

    Handles output projection for language modeling and
    optional multimodal decoding.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tie_weights: bool = False,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        if tie_weights and embed_tokens is not None:
            # Tie output weights with input embeddings
            self.lm_head = None
            self.embed_tokens = embed_tokens
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.embed_tokens = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute output logits and optional loss.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_size]
            labels: Target token IDs for loss computation

        Returns:
            Dict with logits and optional loss
        """
        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied weights
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten and compute cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output
