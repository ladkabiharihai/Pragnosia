"""
Tests for Pragnosia modules.
"""

import pytest
import torch
import torch.nn as nn

from pragnosia.config import PragnosiaConfig
from pragnosia.modules.normalization import RMSNorm
from pragnosia.modules.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from pragnosia.modules.attention import PragnosiaAttention
from pragnosia.modules.mlp import PragnosiaMLP
from pragnosia.modules.thalamus import ThalamusRouter
from pragnosia.modules.experts import Expert, CognitiveCortex
from pragnosia.modules.cortex import InputRouter, TextCortex, VisionCortex, OutputCortex
from pragnosia.modules.transformer import PragnosiaBlock, PragnosiaTransformer


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        norm = RMSNorm(hidden_size=512)
        x = torch.randn(2, 10, 512)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        """Test that output is normalized."""
        norm = RMSNorm(hidden_size=512)
        x = torch.randn(2, 10, 512) * 100  # Large values
        out = norm(x)
        # RMS should be close to 1
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestRotaryEmbedding:
    """Tests for Rotary Position Embedding."""

    def test_forward_shape(self):
        """Test cos/sin output shapes."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        x = torch.randn(2, 10, 8, 64)  # [batch, seq, heads, dim]
        cos, sin = rope(x)
        assert cos.shape[-1] == 64
        assert sin.shape[-1] == 64

    def test_apply_rotary(self):
        """Test applying rotary embeddings."""
        rope = RotaryEmbedding(dim=64)
        q = torch.randn(2, 8, 10, 64)  # [batch, heads, seq, dim]
        k = torch.randn(2, 8, 10, 64)
        cos, sin = rope(q.transpose(1, 2))  # expects [batch, seq, heads, dim]
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestPragnosiaAttention:
    """Tests for attention mechanism."""

    def test_forward_shape(self):
        """Test output shape."""
        attn = PragnosiaAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        x = torch.randn(2, 10, 512)
        out, weights, past_kv = attn(x)
        assert out.shape == x.shape

    def test_gqa_kv_heads(self):
        """Test GQA with fewer KV heads."""
        attn = PragnosiaAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,  # 4x fewer KV heads
        )
        x = torch.randn(2, 10, 512)
        out, _, _ = attn(x)
        assert out.shape == x.shape

    def test_with_cache(self):
        """Test KV caching for generation."""
        attn = PragnosiaAttention(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        x = torch.randn(2, 10, 512)
        out1, _, past_kv = attn(x, use_cache=True)
        assert past_kv is not None
        assert len(past_kv) == 2  # key, value

        # Incremental decode
        x2 = torch.randn(2, 1, 512)
        out2, _, past_kv2 = attn(x2, past_key_value=past_kv, use_cache=True)
        assert out2.shape == (2, 1, 512)
        assert past_kv2[0].shape[2] == 11  # 10 + 1


class TestPragnosiaMLP:
    """Tests for feed-forward network."""

    def test_forward_shape(self):
        """Test output shape."""
        mlp = PragnosiaMLP(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 10, 512)
        out = mlp(x)
        assert out.shape == x.shape

    def test_swiglu_activation(self):
        """Test SwiGLU activation."""
        mlp = PragnosiaMLP(hidden_size=512, intermediate_size=2048, hidden_act="silu")
        x = torch.randn(2, 10, 512)
        out = mlp(x)
        assert not torch.isnan(out).any()


class TestThalamusRouter:
    """Tests for MoE router."""

    def test_forward_shape(self):
        """Test router output shapes."""
        router = ThalamusRouter(
            hidden_size=512,
            num_experts=8,
            num_experts_per_token=2,
        )
        x = torch.randn(2, 10, 512)
        output = router(x)

        assert output.dispatch_mask.shape == (2, 10, 8)
        assert output.combine_weights.shape == (2, 10, 8)
        assert output.expert_indices.shape == (2, 10, 2)

    def test_top_k_selection(self):
        """Test that only top-K experts are selected."""
        router = ThalamusRouter(
            hidden_size=512,
            num_experts=8,
            num_experts_per_token=2,
        )
        x = torch.randn(2, 10, 512)
        output = router(x)

        # Each token should route to exactly 2 experts
        active_per_token = output.dispatch_mask.sum(dim=-1)
        assert torch.allclose(active_per_token, torch.ones_like(active_per_token) * 2)

    def test_energy_gating(self):
        """Test energy budget affects routing."""
        router = ThalamusRouter(
            hidden_size=512,
            num_experts=8,
            num_experts_per_token=2,
            enable_energy_gating=True,
        )
        x = torch.randn(2, 10, 512)

        # Full energy budget
        out_full = router(x, energy_budget=1.0)

        # Reduced energy budget
        out_low = router(x, energy_budget=0.5)

        # Low energy should route to fewer experts
        assert out_low.expert_indices.shape[-1] <= out_full.expert_indices.shape[-1]


class TestExpert:
    """Tests for Expert module."""

    def test_forward_shape(self):
        """Test output shape."""
        expert = Expert(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 10, 512)
        out = expert(x)
        assert out.shape == x.shape


class TestCognitiveCortex:
    """Tests for MoE layer."""

    def test_forward_shape(self):
        """Test output shape."""
        cortex = CognitiveCortex(
            hidden_size=512,
            intermediate_size=2048,
            num_experts=4,
            num_experts_per_token=2,
        )
        x = torch.randn(2, 10, 512)
        out, aux_info = cortex(x)
        assert out.shape == x.shape

    def test_sparse_activation(self):
        """Test that MoE uses sparse activation."""
        cortex = CognitiveCortex(
            hidden_size=512,
            intermediate_size=2048,
            num_experts=8,
            num_experts_per_token=2,
        )
        x = torch.randn(2, 10, 512)
        out, aux_info = cortex(x)

        # Check aux losses are computed
        assert "aux_loss" in aux_info
        assert "z_loss" in aux_info

    def test_expert_utilization(self):
        """Test expert utilization tracking."""
        cortex = CognitiveCortex(
            hidden_size=512,
            intermediate_size=2048,
            num_experts=4,
            num_experts_per_token=2,
        )
        cortex.train()
        x = torch.randn(4, 20, 512)
        _ = cortex(x)

        utilization = cortex.get_expert_utilization()
        assert utilization.shape == (4,)
        assert utilization.sum() > 0


class TestCortexModules:
    """Tests for Input/Output Cortex."""

    def test_text_cortex(self):
        """Test text embedding."""
        cortex = TextCortex(vocab_size=32000, hidden_size=512)
        input_ids = torch.randint(0, 32000, (2, 10))
        embeddings, mask = cortex(input_ids)
        assert embeddings.shape == (2, 10, 512)

    def test_input_router(self):
        """Test input routing."""
        router = InputRouter(hidden_size=512)
        embeddings = torch.randn(2, 10, 512)
        routed = router(embeddings)
        assert routed.tokens.shape == embeddings.shape
        assert 0 <= routed.energy_budget <= 1

    def test_output_cortex(self):
        """Test output projection."""
        cortex = OutputCortex(hidden_size=512, vocab_size=32000)
        hidden = torch.randn(2, 10, 512)
        output = cortex(hidden)
        assert output["logits"].shape == (2, 10, 32000)

    def test_output_with_loss(self):
        """Test loss computation."""
        cortex = OutputCortex(hidden_size=512, vocab_size=32000)
        hidden = torch.randn(2, 10, 512)
        labels = torch.randint(0, 32000, (2, 10))
        output = cortex(hidden, labels)
        assert "loss" in output
        assert output["loss"].ndim == 0  # scalar


class TestPragnosiaBlock:
    """Tests for transformer block."""

    def test_forward_shape(self):
        """Test output shape."""
        block = PragnosiaBlock(
            hidden_size=512,
            intermediate_size=2048,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_token=2,
        )
        x = torch.randn(2, 10, 512)
        out, attn, past_kv, moe_aux = block(x)
        assert out.shape == x.shape

    def test_with_moe(self):
        """Test block with MoE enabled."""
        block = PragnosiaBlock(
            hidden_size=512,
            intermediate_size=2048,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            use_moe=True,
        )
        x = torch.randn(2, 10, 512)
        out, attn, past_kv, moe_aux = block(x)
        assert moe_aux is not None
        assert "aux_loss" in moe_aux

    def test_without_moe(self):
        """Test block without MoE (standard MLP)."""
        block = PragnosiaBlock(
            hidden_size=512,
            intermediate_size=2048,
            num_attention_heads=8,
            num_key_value_heads=2,
            use_moe=False,
        )
        x = torch.randn(2, 10, 512)
        out, attn, past_kv, moe_aux = block(x)
        assert moe_aux is None


class TestPragnosiaTransformer:
    """Tests for full transformer."""

    def test_forward_shape(self):
        """Test output shapes."""
        transformer = PragnosiaTransformer(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
        )
        input_ids = torch.randint(0, 32000, (2, 10))
        outputs = transformer(input_ids)

        assert outputs["last_hidden_state"].shape == (2, 10, 512)

    def test_with_cache(self):
        """Test KV caching."""
        transformer = PragnosiaTransformer(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        input_ids = torch.randint(0, 32000, (2, 10))
        outputs = transformer(input_ids, use_cache=True)

        assert outputs["past_key_values"] is not None
        assert len(outputs["past_key_values"]) == 4  # num_layers

    def test_moe_losses(self):
        """Test MoE auxiliary losses."""
        transformer = PragnosiaTransformer(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
        )
        input_ids = torch.randint(0, 32000, (2, 10))
        outputs = transformer(input_ids)

        assert outputs["moe_aux_loss"] is not None
        assert outputs["moe_z_loss"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
