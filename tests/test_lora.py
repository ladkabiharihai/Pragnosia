"""
Tests for LoRA module.
"""

import pytest
import torch
import torch.nn as nn

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
from pragnosia import Pragnosia, PragnosiaConfig


class TestLoRALinear:
    """Tests for LoRALinear layer."""

    def test_lora_linear_forward(self):
        """Test LoRALinear forward pass."""
        lora_layer = LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
            lora_alpha=16.0,
        )

        x = torch.randn(2, 10, 64)
        y = lora_layer(x)

        assert y.shape == (2, 10, 128)

    def test_lora_linear_from_linear(self):
        """Test creating LoRALinear from nn.Linear."""
        linear = nn.Linear(64, 128)
        lora_layer = LoRALinear.from_linear(linear, r=8)

        # Check weight was copied
        assert torch.allclose(lora_layer.weight, linear.weight)

        # Check forward pass works
        x = torch.randn(2, 64)
        _ = lora_layer(x)

    def test_lora_merge_unmerge(self):
        """Test LoRA weight merging and unmerging."""
        lora_layer = LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
            lora_alpha=16.0,
        )

        x = torch.randn(2, 64)

        # Output before merge
        y_before = lora_layer(x)

        # Merge weights
        lora_layer.merge()
        assert lora_layer.merged

        # Output after merge should be same
        y_merged = lora_layer(x)
        assert torch.allclose(y_before, y_merged, atol=1e-5)

        # Unmerge weights
        lora_layer.unmerge()
        assert not lora_layer.merged

        # Output after unmerge should still be same
        y_unmerged = lora_layer(x)
        assert torch.allclose(y_before, y_unmerged, atol=1e-5)

    def test_lora_parameters_frozen(self):
        """Test that pretrained weight is frozen."""
        lora_layer = LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
        )

        # Pretrained weight should be frozen
        assert not lora_layer.weight.requires_grad

        # LoRA weights should be trainable
        assert lora_layer.lora_A.requires_grad
        assert lora_layer.lora_B.requires_grad

    def test_lora_initialization(self):
        """Test LoRA weight initialization."""
        lora_layer = LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
        )

        # lora_B should be initialized to zeros
        assert torch.allclose(lora_layer.lora_B, torch.zeros_like(lora_layer.lora_B))

        # lora_A should not be all zeros
        assert not torch.allclose(lora_layer.lora_A, torch.zeros_like(lora_layer.lora_A))


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_config(self):
        """Test default LoRA config."""
        config = LoRAConfig()

        assert config.r == 8
        assert config.lora_alpha == 16.0
        assert config.lora_dropout == 0.0
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_custom_config(self):
        """Test custom LoRA config."""
        config = LoRAConfig(
            r=16,
            lora_alpha=32.0,
            target_modules=["gate"],
        )

        assert config.r == 16
        assert config.lora_alpha == 32.0
        assert config.target_modules == ["gate"]


class TestApplyLoRA:
    """Tests for applying LoRA to models."""

    def test_apply_lora_to_simple_model(self):
        """Test applying LoRA to a simple model."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Apply LoRA
        config = LoRAConfig(r=4, target_modules=["0", "2"])
        model = apply_lora(model, config)

        # Check that layers were replaced
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], LoRALinear)

        # Check forward pass
        x = torch.randn(2, 64)
        y = model(x)
        assert y.shape == (2, 64)

    def test_apply_lora_to_pragnosia(self):
        """Test applying LoRA to Pragnosia model."""
        config = PragnosiaConfig.tiny()
        model = Pragnosia(config)

        # Count params before
        total_before = sum(p.numel() for p in model.parameters())

        # Apply LoRA
        lora_config = LoRAConfig(
            r=4,
            target_modules=["q_proj", "v_proj"],
        )
        model = apply_lora(model, lora_config)

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Trainable params should be much less than total
        assert trainable < total_before * 0.1  # Less than 10%

        # Forward pass should work
        x = torch.randint(0, config.vocab_size, (1, 32))
        outputs = model(x)
        assert "logits" in outputs

    def test_count_lora_parameters(self):
        """Test counting LoRA parameters."""
        config = PragnosiaConfig.tiny()
        model = Pragnosia(config)

        # Apply LoRA
        lora_config = LoRAConfig(r=8)
        model = apply_lora(model, lora_config)

        # Count parameters
        counts = count_lora_parameters(model)

        assert counts["total_params"] > 0
        assert counts["trainable_params"] > 0
        assert counts["lora_params"] > 0
        assert counts["trainable_percentage"] < 10  # Should be small


class TestLoRAStateDicts:
    """Tests for LoRA state dict operations."""

    def test_get_lora_state_dict(self):
        """Test getting LoRA state dict."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        config = LoRAConfig(r=4, target_modules=["0", "2"])
        model = apply_lora(model, config)

        # Get LoRA state dict
        lora_state = get_lora_state_dict(model)

        # Should contain lora_A and lora_B for each layer
        assert len(lora_state) == 4  # 2 layers * 2 params (A and B)
        assert all("lora_A" in k or "lora_B" in k for k in lora_state.keys())

    def test_load_lora_state_dict(self):
        """Test loading LoRA state dict."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        config = LoRAConfig(r=4, target_modules=["0", "2"])
        model = apply_lora(model, config)

        # Modify LoRA weights
        model[0].lora_A.data.fill_(1.0)

        # Get state dict
        lora_state = get_lora_state_dict(model)

        # Create new model
        model2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        model2 = apply_lora(model2, config)

        # Load state dict
        load_lora_state_dict(model2, lora_state)

        # Check weights match
        assert torch.allclose(model[0].lora_A, model2[0].lora_A)


class TestLoRATraining:
    """Tests for LoRA training scenarios."""

    def test_lora_gradient_flow(self):
        """Test that gradients flow through LoRA layers."""
        model = nn.Sequential(
            nn.Linear(64, 64),
        )

        config = LoRAConfig(r=4, target_modules=["0"])
        model = apply_lora(model, config)

        x = torch.randn(2, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # LoRA weights should have gradients
        assert model[0].lora_A.grad is not None
        assert model[0].lora_B.grad is not None

        # Pretrained weight should not have gradients
        assert model[0].weight.grad is None

    def test_lora_training_step(self):
        """Test a LoRA training step."""
        config = PragnosiaConfig.tiny()
        model = Pragnosia(config)

        lora_config = LoRAConfig(r=4)
        model = apply_lora(model, lora_config)

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        # Training step
        x = torch.randint(0, config.vocab_size, (1, 32))
        outputs = model(x, labels=x)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Should complete without error
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
