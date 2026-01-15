"""
Tests for memory optimization utilities.
"""

import pytest
import torch
import torch.nn as nn

from pragnosia.config import PragnosiaConfig
from pragnosia.model import Pragnosia
from pragnosia.utils.memory import (
    get_memory_stats,
    count_parameters,
    enable_gradient_checkpointing,
    clear_memory,
)


class TestMemoryUtils:
    """Tests for memory utilities."""

    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Linear(100, 100)
        params = count_parameters(model)

        assert params["total"] == 100 * 100 + 100  # weights + bias
        assert params["trainable"] == params["total"]

    def test_count_with_frozen(self):
        """Test counting with frozen parameters."""
        model = nn.Linear(100, 100)
        model.weight.requires_grad = False

        params = count_parameters(model)
        assert params["frozen"] == 100 * 100
        assert params["trainable"] == 100  # only bias

    def test_get_memory_stats_cpu(self):
        """Test memory stats on CPU."""
        stats = get_memory_stats(torch.device("cpu"))
        assert stats["allocated_gb"] == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_stats_gpu(self):
        """Test memory stats on GPU."""
        # Allocate some memory - need to keep reference alive
        x = torch.randn(1000, 1000, device="cuda")
        torch.cuda.synchronize()  # Ensure allocation is complete

        stats = get_memory_stats()
        assert stats["total_gb"] > 0
        # allocated_gb may be 0 due to caching, but reserved should be > 0
        assert stats["reserved_gb"] > 0 or stats["allocated_gb"] > 0

        del x
        clear_memory()

    def test_enable_gradient_checkpointing(self):
        """Test enabling gradient checkpointing."""
        config = PragnosiaConfig.tiny()
        model = Pragnosia(config)

        enable_gradient_checkpointing(model)

        assert model.transformer.use_gradient_checkpointing


class TestPlasticity:
    """Tests for plasticity engine."""

    def test_plasticity_engine_creation(self):
        """Test plasticity engine initialization."""
        from pragnosia.modules.plasticity import PlasticityEngine

        engine = PlasticityEngine(
            growth_threshold=0.8,
            prune_threshold=0.05,
            max_experts=16,
            min_experts=4,
        )
        assert engine is not None

    def test_record_step(self):
        """Test recording training steps."""
        from pragnosia.modules.plasticity import PlasticityEngine

        engine = PlasticityEngine()
        engine.record_step(loss=1.0, routing_entropy=0.5)
        engine.record_step(loss=0.9, routing_entropy=0.5)

        assert engine.step_count == 2
        assert len(engine.loss_history) == 2

    def test_should_check(self):
        """Test check frequency."""
        from pragnosia.modules.plasticity import PlasticityEngine

        engine = PlasticityEngine(warmup_steps=10, check_frequency=5)

        # During warmup (steps 1-9)
        for i in range(9):
            engine.record_step(1.0, 0.5)
            assert not engine.should_check(), f"Should not check at step {engine.step_count}"

        # Step 10 - warmup complete, and 10 % 5 == 0, so should check
        engine.record_step(1.0, 0.5)
        assert engine.should_check(), "Should check at step 10 (warmup done, multiple of 5)"

        # Steps 11-14 should not check (not multiple of 5)
        for i in range(4):
            engine.record_step(1.0, 0.5)
            assert not engine.should_check()

        # Step 15 should check (multiple of 5)
        engine.record_step(1.0, 0.5)
        assert engine.should_check()

    def test_analyze_growth(self):
        """Test growth decision."""
        from pragnosia.modules.plasticity import PlasticityEngine

        engine = PlasticityEngine(growth_threshold=0.5, max_experts=16)

        # High entropy should trigger growth
        utilization = torch.ones(4) * 0.25  # uniform utilization
        action = engine.analyze(None, routing_entropy=0.9, expert_utilization=utilization)

        assert action.action_type == "grow"

    def test_analyze_prune(self):
        """Test pruning decision."""
        from pragnosia.modules.plasticity import PlasticityEngine

        engine = PlasticityEngine(prune_threshold=0.1, min_experts=2)

        # One expert barely used
        utilization = torch.tensor([0.4, 0.4, 0.15, 0.04])  # last one underutilized
        action = engine.analyze(None, routing_entropy=0.3, expert_utilization=utilization)

        assert action.action_type == "prune"
        assert action.expert_id == 3

    def test_get_summary(self):
        """Test getting summary."""
        from pragnosia.modules.plasticity import PlasticityEngine

        engine = PlasticityEngine()
        for i in range(10):
            engine.record_step(1.0 - i * 0.1, 0.5)

        summary = engine.get_summary()
        assert "total_steps" in summary
        assert summary["total_steps"] == 10


class TestMemoryOptimizedModel:
    """Tests for memory-optimized model."""

    def test_tiny_model_memory(self):
        """Test tiny model fits in memory."""
        config = PragnosiaConfig.tiny()
        model = Pragnosia(config)

        footprint = model.get_memory_footprint()
        # Tiny model should be < 1GB
        assert footprint["memory_gb_fp16"] < 1.0

    def test_gradient_checkpointing_reduces_memory(self):
        """Test that gradient checkpointing is enabled."""
        config = PragnosiaConfig.tiny()
        config.use_gradient_checkpointing = True
        model = Pragnosia(config)

        # Verify checkpointing is enabled
        assert model.transformer.use_gradient_checkpointing

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_fits_in_4gb(self):
        """Test that tiny model fits in 4GB GPU."""
        config = PragnosiaConfig.tiny()
        model = Pragnosia(config).cuda()

        input_ids = torch.randint(0, config.vocab_size, (1, 128)).cuda()
        labels = torch.randint(0, config.vocab_size, (1, 128)).cuda()

        # Forward + backward
        outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()

        # Check memory usage
        stats = get_memory_stats()
        assert stats["allocated_gb"] < 4.0, f"Model uses {stats['allocated_gb']}GB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
