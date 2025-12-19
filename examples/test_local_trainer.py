"""Quick test of local learning trainer."""
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append("../src")

from pragnosia import PragnosiaModel, LocalLearningTrainer
from pragnosia.utils.config import PragnosiaConfig

print("Creating tiny test dataset...")
# Create tiny synthetic dataset
batch_size = 2
seq_len = 32
vocab_size = 100

# Create 10 samples
input_ids = torch.randint(0, vocab_size, (10, seq_len))
labels = input_ids.clone()

dataset = TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Creating model...")
config = PragnosiaConfig(
    vocab_size=vocab_size,
    hidden_size=128,
    num_experts=4,
    num_active_experts=2,
    num_hidden_layers=2,
    max_position_embeddings=64,
    offload_to_cpu=False,  # Keep on GPU for testing
)

model = PragnosiaModel(config)

print("Creating trainer...")
def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    # For language modeling: predict next token
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]  # Shift left
    labels[:, -1] = -100  # Ignore last position
    return {"input_ids": input_ids, "labels": labels}

train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

trainer = LocalLearningTrainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    device="cpu",  # Use CPU for testing
    output_dir="./test_output",
    logging_steps=1,
    local_learning_rate=0.01,
)

print("Starting training (1 epoch, 5 batches)...")
try:
    trainer.train(num_epochs=1)
    print("✓ Training completed successfully!")
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
