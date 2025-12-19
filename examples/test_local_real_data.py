"""Test local trainer with real wikitext data."""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("../src")

from pragnosia import PragnosiaModel, LocalLearningTrainer
from pragnosia.utils.config import PragnosiaConfig

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Loading wikitext dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    texts = [text for text in examples["text"] if text and len(text.strip()) > 0]
    if not texts:
        return {"input_ids": [], "attention_mask": []}
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

print("Tokenizing...")
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Take small subset
train_subset = tokenized["train"].select(range(min(50, len(tokenized["train"]))))

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    # For language modeling, labels should be shifted: predict next token
    # labels[i] = input_ids[i+1], with last position set to -100 (ignore)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]  # Shift left
    labels[:, -1] = -100  # Ignore last position (no next token)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader = DataLoader(train_subset, batch_size=2, collate_fn=collate_fn)

print("Creating model...")
config = PragnosiaConfig(
    vocab_size=50257,
    hidden_size=256,
    num_experts=4,
    num_active_experts=2,
    num_hidden_layers=4,
    max_position_embeddings=256,
    offload_to_cpu=False,
    use_neuromodulation=False,  # Disable for simpler testing
)

model = PragnosiaModel(config)

print("Creating trainer...")
trainer = LocalLearningTrainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    device="cpu",
    output_dir="./test_output",
    logging_steps=5,
    local_learning_rate=0.001,
)

print("\nStarting training...")
print("Watch for loss values - they should be > 0 and decrease")
print("=" * 60)

try:
    trainer.train(num_epochs=1)
    print("\n✓ Training completed!")

    # Check loss history
    if hasattr(trainer, 'global_step') and trainer.global_step > 0:
        print("\n✓ Model trained successfully")
    else:
        print("\n⚠ No training steps completed")

except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
