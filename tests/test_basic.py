"""Basic tests for Pragnosia components."""
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from pragnosia import PragnosiaModel
from pragnosia.utils.config import PragnosiaConfig


def test_model_initialization():
    """Test basic model initialization."""
    config = PragnosiaConfig(
        vocab_size=1000,
        hidden_size=128,
        num_experts=4,
        num_active_experts=2,
    )

    model = PragnosiaModel(config)
    assert model is not None
    print("✓ Model initialization successful")


def test_forward_pass():
    """Test forward pass through the model."""
    config = PragnosiaConfig(
        vocab_size=1000,
        hidden_size=128,
        num_experts=4,
        num_active_experts=2,
    )

    model = PragnosiaModel(config)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert outputs["loss"] is not None
    print("✓ Forward pass successful")


def test_memory_systems():
    """Test hippocampus and neocortex."""
    from pragnosia.memory.hippocampus import Hippocampus
    from pragnosia.memory.neocortex import Neocortex

    hippo = Hippocampus(capacity=100)
    neo = Neocortex(capacity=500, hidden_size=128)

    # Store experience
    hidden = torch.randn(2, 10, 128)
    labels = torch.randint(0, 1000, (2, 10))
    loss = torch.randn(2)
    surprise = torch.randn(2)

    hippo.store(hidden, labels, loss, surprise)
    assert hippo.get_size() == 2
    print("✓ Memory systems working")


def test_router():
    """Test Hebbian router."""
    from pragnosia.models.router import HebbianRouter

    router = HebbianRouter(
        input_size=128,
        num_experts=4,
        num_active_experts=2,
    )

    hidden = torch.randn(2, 10, 128)
    features, selected, weights = router(hidden, return_routing_weights=True)

    assert len(selected) == 2
    assert weights is not None
    print("✓ Router working")


def test_intrinsic_losses():
    """Test intrinsic learning objectives."""
    from pragnosia.losses.intrinsic import IntrinsicObjective

    intrinsic = IntrinsicObjective()

    current_hidden = torch.randn(2, 10, 768)
    previous_hidden = torch.randn(2, 10, 768)
    logits = torch.randn(2, 10, 1000)
    targets = torch.randint(0, 1000, (2, 10))

    loss, components = intrinsic(
        current_hidden=current_hidden,
        previous_hidden=previous_hidden,
        prediction_logits=logits,
        targets=targets,
    )

    assert loss is not None
    assert "surprise" in components
    assert "temporal" in components
    print("✓ Intrinsic losses working")


def test_plasticity_scheduler():
    """Test plasticity scheduler."""
    from pragnosia.utils.plasticity import PlasticityScheduler, PlasticityPhase

    scheduler = PlasticityScheduler(
        total_steps=1000,
        exploration_end=0.3,
        stabilization_end=0.7,
    )

    # Test exploration phase
    assert scheduler.get_phase() == PlasticityPhase.EXPLORATION
    assert scheduler.can_grow()
    assert not scheduler.can_prune()

    # Advance to stabilization
    for _ in range(300):
        scheduler.step()
    assert scheduler.get_phase() == PlasticityPhase.STABILIZATION
    assert not scheduler.can_grow()
    assert scheduler.can_prune()

    # Advance to exploitation
    for _ in range(400):
        scheduler.step()
    assert scheduler.get_phase() == PlasticityPhase.EXPLOITATION
    assert not scheduler.can_grow()
    assert not scheduler.can_prune()

    print("✓ Plasticity scheduler working")


if __name__ == "__main__":
    print("Running Pragnosia tests...\n")

    test_model_initialization()
    test_forward_pass()
    test_memory_systems()
    test_router()
    test_intrinsic_losses()
    test_plasticity_scheduler()

    print("\n✓ All tests passed!")
