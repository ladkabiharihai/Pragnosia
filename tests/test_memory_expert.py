"""
Tests for Memory Expert module.
"""

import pytest
import torch

from pragnosia.modules.memory_expert import (
    MemoryExpert,
    MemoryState,
    MemoryAugmentedTransformerBlock,
)


class TestMemoryExpert:
    """Tests for MemoryExpert."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
        )

        x = torch.randn(2, 32, 256)
        output, memory_state = expert(x)

        assert output.shape == x.shape
        assert isinstance(memory_state, MemoryState)

    def test_memory_state_initialization(self):
        """Test memory state is properly initialized."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
        )

        x = torch.randn(2, 32, 256)
        _, memory_state = expert(x)

        assert memory_state.memory_bank.shape == (2, 64, 256)
        assert memory_state.memory_mask.shape == (2, 64)
        assert memory_state.write_pointer > 0  # Should have written something

    def test_memory_persistence(self):
        """Test that memory persists across calls."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
            compression_ratio=4,
        )

        # First pass
        x1 = torch.randn(2, 32, 256)
        _, memory_state = expert(x1)
        pointer_after_first = memory_state.write_pointer

        # Second pass with same memory state
        x2 = torch.randn(2, 32, 256)
        _, memory_state = expert(x2, memory_state)
        pointer_after_second = memory_state.write_pointer

        # Write pointer should have advanced
        assert pointer_after_second != pointer_after_first

    def test_memory_no_update(self):
        """Test memory not updated when update_memory=False."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
        )

        # First pass with update
        x1 = torch.randn(2, 32, 256)
        _, memory_state = expert(x1, update_memory=True)
        pointer_after_first = memory_state.write_pointer

        # Second pass without update
        x2 = torch.randn(2, 32, 256)
        _, memory_state_no_update = expert(x2, memory_state, update_memory=False)

        # Write pointer should not have changed
        assert memory_state_no_update.write_pointer == pointer_after_first

    def test_memory_reset(self):
        """Test memory reset."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
        )

        # Fill memory
        x = torch.randn(2, 32, 256)
        _, memory_state = expert(x)

        # Reset memory
        new_state = expert.reset_memory(
            batch_size=2,
            device=x.device,
            dtype=x.dtype,
        )

        assert new_state.write_pointer == 0
        assert not new_state.memory_mask.any()
        assert (new_state.memory_bank == 0).all()

    def test_compression(self):
        """Test token compression."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
            compression_ratio=4,
        )

        x = torch.randn(2, 32, 256)
        compressed = expert._compress_tokens(x)

        # Should be compressed by factor of 4
        assert compressed.shape == (2, 8, 256)

    def test_gradients_flow(self):
        """Test that gradients flow through memory expert."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
        )

        x = torch.randn(2, 32, 256, requires_grad=True)
        output, _ = expert(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
        )

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 16, 256)
            output, memory_state = expert(x)
            assert output.shape == x.shape
            assert memory_state.memory_bank.shape[0] == batch_size


class TestMemoryAugmentedTransformerBlock:
    """Tests for MemoryAugmentedTransformerBlock."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        block = MemoryAugmentedTransformerBlock(
            hidden_size=256,
            num_heads=8,
            memory_size=64,
        )

        x = torch.randn(2, 32, 256)
        output, memory_state = block(x)

        assert output.shape == x.shape
        assert isinstance(memory_state, MemoryState)

    def test_memory_persistence_in_block(self):
        """Test memory persists in transformer block."""
        block = MemoryAugmentedTransformerBlock(
            hidden_size=256,
            num_heads=8,
            memory_size=64,
        )

        # First forward
        x1 = torch.randn(2, 32, 256)
        _, memory_state = block(x1)

        # Second forward with memory
        x2 = torch.randn(2, 32, 256)
        output2, _ = block(x2, memory_state)

        # Output should be valid
        assert output2.shape == x2.shape

    def test_with_attention_mask(self):
        """Test with attention mask."""
        block = MemoryAugmentedTransformerBlock(
            hidden_size=256,
            num_heads=8,
            memory_size=64,
        )

        x = torch.randn(2, 32, 256)
        mask = torch.zeros(2, 32, dtype=torch.bool)
        mask[:, 16:] = True  # Mask second half

        output, _ = block(x, attention_mask=mask)
        assert output.shape == x.shape


class TestMemoryExpertEdgeCases:
    """Tests for edge cases in memory expert."""

    def test_empty_memory_retrieval(self):
        """Test retrieval from empty memory."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
        )

        # Create empty memory state
        memory_state = expert.reset_memory(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        x = torch.randn(2, 32, 256)
        output, _ = expert(x, memory_state, update_memory=False)

        # Should still produce valid output
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_memory_overflow(self):
        """Test memory circular buffer overflow."""
        expert = MemoryExpert(
            hidden_size=128,
            memory_size=16,  # Small memory
            num_heads=4,
            compression_ratio=2,
        )

        memory_state = None

        # Write more than memory can hold
        for _ in range(10):
            x = torch.randn(2, 64, 128)  # Will compress to 32 tokens
            _, memory_state = expert(x, memory_state)

        # Should still work (circular buffer)
        assert memory_state.memory_bank.shape == (2, 16, 128)

    def test_single_token(self):
        """Test with single token sequence."""
        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
            compression_ratio=1,  # No compression for single token
        )

        x = torch.randn(2, 1, 256)
        output, memory_state = expert(x)

        assert output.shape == x.shape

    def test_gpu_if_available(self):
        """Test on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        expert = MemoryExpert(
            hidden_size=256,
            memory_size=64,
            num_heads=8,
        ).cuda()

        x = torch.randn(2, 32, 256).cuda()
        output, memory_state = expert(x)

        assert output.device.type == "cuda"
        assert memory_state.memory_bank.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
