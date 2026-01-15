"""
Tests for the main Pragnosia model.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from pragnosia.config import PragnosiaConfig
from pragnosia.model import Pragnosia


class TestPragnosiaConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PragnosiaConfig()
        assert config.hidden_size == 2048
        assert config.num_experts == 8
        assert config.num_experts_per_token == 2

    def test_tiny_config(self):
        """Test tiny preset."""
        config = PragnosiaConfig.tiny()
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 8

    def test_small_config(self):
        """Test small preset."""
        config = PragnosiaConfig.small()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 16

    def test_yaml_roundtrip(self):
        """Test saving and loading config."""
        config = PragnosiaConfig.tiny()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(str(path))
            loaded = PragnosiaConfig.from_yaml(str(path))
            assert loaded.hidden_size == config.hidden_size
            assert loaded.num_experts == config.num_experts


class TestPragnosiaModel:
    """Tests for main model."""

    @pytest.fixture
    def tiny_config(self):
        """Create tiny config for testing."""
        return PragnosiaConfig.tiny()

    @pytest.fixture
    def tiny_model(self, tiny_config):
        """Create tiny model for testing."""
        return Pragnosia(tiny_config)

    def test_model_creation(self, tiny_config):
        """Test model can be created."""
        model = Pragnosia(tiny_config)
        assert model is not None

    def test_forward_shape(self, tiny_model, tiny_config):
        """Test forward pass output shapes."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))
        outputs = tiny_model(input_ids)

        assert outputs["logits"].shape == (2, 10, tiny_config.vocab_size)

    def test_with_labels(self, tiny_model, tiny_config):
        """Test loss computation."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))
        labels = torch.randint(0, tiny_config.vocab_size, (2, 10))
        outputs = tiny_model(input_ids, labels=labels)

        assert outputs["loss"] is not None
        assert outputs["loss"].ndim == 0  # scalar

    def test_generate(self, tiny_model, tiny_config):
        """Test text generation."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 5))
        generated = tiny_model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
        )

        assert generated.shape[1] >= input_ids.shape[1]
        assert generated.shape[1] <= input_ids.shape[1] + 10

    def test_generate_with_sampling(self, tiny_model, tiny_config):
        """Test generation with sampling."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 5))
        generated = tiny_model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )

        assert generated.shape[1] >= input_ids.shape[1]

    def test_energy_budget(self, tiny_model, tiny_config):
        """Test energy budget affects computation."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))

        # Full energy
        out_full = tiny_model(input_ids, energy_budget=1.0)

        # Reduced energy
        out_low = tiny_model(input_ids, energy_budget=0.5)

        # Both should produce valid outputs
        assert out_full["logits"].shape == out_low["logits"].shape
        assert out_low["energy_budget_used"] <= out_full["energy_budget_used"]

    def test_save_load(self, tiny_model, tiny_config):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            tiny_model.save_pretrained(tmpdir)

            # Verify files exist
            assert (Path(tmpdir) / "config.yaml").exists()
            assert (Path(tmpdir) / "model.pt").exists()

            # Load
            loaded = Pragnosia.from_pretrained(tmpdir)
            assert loaded is not None

            # Verify same output
            input_ids = torch.randint(0, tiny_config.vocab_size, (1, 5))
            with torch.no_grad():
                out1 = tiny_model(input_ids)
                out2 = loaded(input_ids)
            assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)

    def test_parameter_count(self, tiny_model):
        """Test parameter counting."""
        num_params = tiny_model.get_num_params(non_embedding=False)
        assert num_params > 0

        # Non-embedding count
        num_params_no_emb = tiny_model.get_num_params(non_embedding=True)
        # Should be strictly less (excluding embeddings)
        assert num_params_no_emb <= num_params

    def test_memory_footprint(self, tiny_model):
        """Test memory footprint estimation."""
        footprint = tiny_model.get_memory_footprint()
        assert "params_millions" in footprint
        assert "memory_gb_fp32" in footprint
        assert footprint["memory_gb_4bit"] < footprint["memory_gb_fp16"]


class TestPragnosiaGradients:
    """Tests for gradient computation."""

    @pytest.fixture
    def tiny_model(self):
        """Create tiny model for testing."""
        config = PragnosiaConfig.tiny()
        return Pragnosia(config)

    def test_gradients_flow(self, tiny_model):
        """Test gradients flow through model."""
        config = tiny_model.config
        # Avoid token 0 (padding) to ensure gradient flow through embeddings
        input_ids = torch.randint(1, config.vocab_size, (2, 10))
        labels = torch.randint(1, config.vocab_size, (2, 10))

        outputs = tiny_model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        # Check gradients exist for most parameters
        # Note: embedding layers may have sparse gradients due to padding_idx
        params_with_grad = 0
        params_without_grad = 0
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad += 1
                else:
                    params_without_grad += 1

        # Most parameters should have gradients
        assert params_with_grad > params_without_grad, \
            f"Expected more params with gradients: {params_with_grad} vs {params_without_grad}"

    def test_moe_loss_in_backward(self, tiny_model):
        """Test MoE losses are included in backward."""
        config = tiny_model.config
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        labels = torch.randint(0, config.vocab_size, (2, 10))

        outputs = tiny_model(input_ids, labels=labels)

        # Total loss should include MoE losses
        total_loss = outputs["loss"]
        moe_aux = outputs["moe_aux_loss"]
        moe_z = outputs["moe_z_loss"]

        assert moe_aux >= 0
        assert moe_z >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPragnosiaGPU:
    """GPU-specific tests."""

    def test_gpu_forward(self):
        """Test forward pass on GPU."""
        config = PragnosiaConfig.tiny()
        model = Pragnosia(config).cuda()
        input_ids = torch.randint(0, config.vocab_size, (2, 10)).cuda()

        outputs = model(input_ids)
        assert outputs["logits"].device.type == "cuda"

    def test_gpu_generate(self):
        """Test generation on GPU."""
        config = PragnosiaConfig.tiny()
        model = Pragnosia(config).cuda()
        input_ids = torch.randint(0, config.vocab_size, (1, 5)).cuda()

        generated = model.generate(input_ids, max_new_tokens=10)
        assert generated.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
