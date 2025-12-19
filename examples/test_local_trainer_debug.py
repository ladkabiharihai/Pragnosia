"""Debug test to check if loss is computed correctly."""
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append("../src")

from pragnosia import PragnosiaModel
from pragnosia.utils.config import PragnosiaConfig

print("Creating test data...")
batch_size = 2
seq_len = 32
vocab_size = 100

# Create simple dataset
input_ids = torch.randint(0, vocab_size, (10, seq_len))
labels = input_ids.clone()

print("Creating model...")
config = PragnosiaConfig(
    vocab_size=vocab_size,
    hidden_size=128,
    num_experts=4,
    num_active_experts=2,
    num_hidden_layers=2,
    max_position_embeddings=64,
    offload_to_cpu=False,
)

model = PragnosiaModel(config)
model.eval()

print("\nTesting forward pass...")
with torch.no_grad():
    batch_input = input_ids[:2]
    batch_labels = labels[:2]

    print(f"Input shape: {batch_input.shape}")
    print(f"Labels shape: {batch_labels.shape}")

    outputs = model(batch_input, labels=batch_labels)

    print(f"\nOutputs keys: {outputs.keys()}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Intrinsic loss: {outputs['intrinsic_loss'].item():.4f}")
    print(f"Homeostatic loss: {outputs['homeostatic_loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")

    # Check if logits are reasonable
    print(f"\nLogits stats:")
    print(f"  Min: {outputs['logits'].min().item():.4f}")
    print(f"  Max: {outputs['logits'].max().item():.4f}")
    print(f"  Mean: {outputs['logits'].mean().item():.4f}")
    print(f"  Std: {outputs['logits'].std().item():.4f}")

    # Compute CE loss manually
    ce_loss = torch.nn.functional.cross_entropy(
        outputs['logits'].view(-1, vocab_size),
        batch_labels.view(-1),
    )
    print(f"\nManual CE loss: {ce_loss.item():.4f}")

print("\n✓ Forward pass works!")
print("\nNow testing local training...")

# Test local expert update
expert = model.experts[0]
optimizer = torch.optim.SGD(expert.parameters(), lr=0.01)

print("\nBefore training:")
token_embeds = model.token_embedding(batch_input)
position_ids = torch.arange(seq_len).unsqueeze(0)
position_embeds = model.position_embedding(position_ids)
hidden_states = token_embeds + position_embeds

output1, _ = expert(hidden_states)
logits1 = model.output_head(output1)
loss1 = torch.nn.functional.cross_entropy(
    logits1.view(-1, vocab_size),
    batch_labels.view(-1),
)
print(f"Expert 0 loss: {loss1.item():.4f}")

# Update
optimizer.zero_grad()
loss1.backward()
optimizer.step()

print("\nAfter 1 update:")
with torch.no_grad():
    output2, _ = expert(hidden_states.detach())
    logits2 = model.output_head(output2)
    loss2 = torch.nn.functional.cross_entropy(
        logits2.view(-1, vocab_size),
        batch_labels.view(-1),
    )
    print(f"Expert 0 loss: {loss2.item():.4f}")
    print(f"Loss change: {loss2.item() - loss1.item():.4f}")

if loss2.item() < loss1.item():
    print("\n✓ Expert is learning (loss decreased)")
else:
    print("\n⚠ Expert not learning well (loss did not decrease)")
