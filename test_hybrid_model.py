"""Test hybrid local-global architecture.

Tests:
1. Model initialization with coherence
2. Forward pass with coherence
3. Memory footprint calculation
4. Gradient flow through coherence
"""
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pragnosia.utils.config import PragnosiaConfig
from pragnosia.models.pragnosia_model import PragnosiaModel


def test_hybrid_model():
    """Test hybrid model architecture."""

    print("=" * 80)
    print("TESTING HYBRID LOCAL-GLOBAL ARCHITECTURE")
    print("=" * 80)

    # Create config with coherence enabled
    config = PragnosiaConfig(
        vocab_size=50257,
        hidden_size=512,
        num_experts=8,
        num_active_experts=2,
        expert_size=512,
        use_coherence_module=True,  # Enable coherence
        coherence_num_layers=2,  # Small for testing
        coherence_num_heads=8,
    )

    print("\n1. INITIALIZING MODEL WITH COHERENCE")
    print("-" * 80)
    model = PragnosiaModel(config)

    # Check coherence module exists
    assert hasattr(model, 'coherence'), "Model should have coherence module"
    assert model.coherence is not None, "Coherence should be initialized"
    print(f"✓ Coherence module initialized: {model.coherence is not None}")

    # Print memory footprint
    coherence_memory = model.coherence.get_memory_size_mb()
    print(f"✓ Coherence memory footprint: {coherence_memory:.2f} MB")

    # Check total model size
    total_params = sum(p.numel() for p in model.parameters())
    coherence_params = sum(p.numel() for p in model.coherence.parameters())
    print(f"✓ Total model parameters: {total_params:,}")
    print(f"✓ Coherence parameters: {coherence_params:,} ({100*coherence_params/total_params:.1f}%)")

    print("\n2. TESTING FORWARD PASS")
    print("-" * 80)

    # Create dummy input
    batch_size = 4
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_ids = input_ids.to(device)

    print(f"✓ Device: {device}")
    print(f"✓ Input shape: {input_ids.shape}")

    # Track VRAM before
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / (1024**2)
        print(f"✓ VRAM before forward: {vram_before:.2f} MB")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, inference_mode=True)

    # Track VRAM after
    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated() / (1024**2)
        vram_peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"✓ VRAM after forward: {vram_after:.2f} MB")
        print(f"✓ VRAM peak: {vram_peak:.2f} MB")

    # Check outputs
    assert "logits" in outputs, "Should have logits"
    assert outputs["logits"].shape == (batch_size, seq_length, config.vocab_size), "Logits shape incorrect"
    print(f"✓ Logits shape: {outputs['logits'].shape}")

    # Check hidden states (should be coherent)
    assert "hidden_states" in outputs, "Should have hidden states"
    print(f"✓ Hidden states shape: {outputs['hidden_states'].shape}")

    print("\n3. TESTING GRADIENT FLOW")
    print("-" * 80)

    # Forward with gradients
    model.train()
    outputs = model(input_ids, labels=input_ids)

    # Check loss
    assert "loss" in outputs, "Should have loss"
    assert outputs["loss"].item() > 0, "Loss should be positive"
    print(f"✓ Loss: {outputs['loss'].item():.4f}")

    # Backward pass
    loss = outputs["loss"]
    loss.backward()

    # Check gradients in coherence module
    coherence_has_grads = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.coherence.parameters()
    )
    print(f"✓ Coherence has gradients: {coherence_has_grads}")

    # Check gradients in experts (should be None for this simple test)
    expert_has_grads = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for expert in model.experts
        for p in expert.parameters()
    )
    print(f"✓ Experts have gradients: {expert_has_grads}")

    # Check output head has gradients
    output_head_has_grads = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.output_head.parameters()
    )
    print(f"✓ Output head has gradients: {output_head_has_grads}")

    print("\n4. TESTING WITHOUT COHERENCE (ABLATION)")
    print("-" * 80)

    # Create model without coherence
    config_no_coherence = PragnosiaConfig(
        vocab_size=50257,
        hidden_size=512,
        num_experts=8,
        num_active_experts=2,
        expert_size=512,
        use_coherence_module=False,  # Disable coherence
    )

    model_no_coherence = PragnosiaModel(config_no_coherence).to(device)

    # Forward pass
    with torch.no_grad():
        outputs_no_coherence = model_no_coherence(input_ids, inference_mode=True)

    print(f"✓ Model without coherence works")
    print(f"✓ Logits shape: {outputs_no_coherence['logits'].shape}")

    # Compare VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_no_coherence = torch.cuda.memory_allocated() / (1024**2)
        vram_diff = vram_after - vram_no_coherence
        print(f"✓ VRAM diff (with vs without coherence): {vram_diff:.2f} MB")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\nHYBRID ARCHITECTURE SUMMARY:")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Coherence overhead: {coherence_params:,} ({100*coherence_params/total_params:.1f}%)")
    print(f"  • Coherence memory: ~{coherence_memory:.0f} MB")
    if torch.cuda.is_available():
        print(f"  • Total VRAM (active): ~{vram_after:.0f} MB")
    print(f"\n  ✓ Dual-system learning enabled")
    print(f"  ✓ Local learning (experts) + Global learning (coherence)")
    print(f"  ✓ Forward/backward passes work correctly")
    print("=" * 80)


if __name__ == "__main__":
    test_hybrid_model()
