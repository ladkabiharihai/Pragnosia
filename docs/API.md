# Pragnosia API Reference

## Core Classes

### PragnosiaModel

Main model class integrating all components.

```python
from pragnosia import PragnosiaModel
from pragnosia.utils.config import PragnosiaConfig

config = PragnosiaConfig(
    vocab_size=50257,
    hidden_size=768,
    num_experts=8,
    num_active_experts=2,
)

model = PragnosiaModel(config)
```

#### Methods

##### `forward(input_ids, attention_mask=None, labels=None, return_dict=True)`

Forward pass through the model.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs, shape (batch, seq_len)
- `attention_mask` (torch.Tensor, optional): Attention mask, shape (batch, seq_len)
- `labels` (torch.Tensor, optional): Target labels, shape (batch, seq_len)
- `return_dict` (bool): Whether to return dictionary output

**Returns:**
- Dictionary containing:
  - `logits`: Model predictions, shape (batch, seq_len, vocab_size)
  - `loss`: Total loss (if labels provided)
  - `intrinsic_loss`: Intrinsic learning objective
  - `homeostatic_loss`: Homeostatic penalty
  - `hidden_states`: Final hidden representations
  - `routing_stats`: Router statistics

##### `get_memory_statistics()`

Get comprehensive memory statistics.

**Returns:**
- Dictionary with memory usage for each component

---

### PragnosiaConfig

Configuration class for model hyperparameters.

```python
from pragnosia.utils.config import PragnosiaConfig

config = PragnosiaConfig(
    # Model architecture
    vocab_size=50257,
    hidden_size=768,
    num_experts=8,
    num_active_experts=2,
    num_hidden_layers=12,

    # Intrinsic learning
    alpha_surprise=0.3,
    beta_temporal=0.25,
    gamma_disagreement=0.25,
    delta_compression=0.2,
    learn_intrinsic_weights=True,

    # Neuroplasticity
    exploration_end=0.3,
    stabilization_end=0.7,
    max_growth_rate=0.01,
    max_pruning_rate=0.01,

    # Memory systems
    hippocampus_capacity=10000,
    neocortex_capacity=50000,

    # GPU memory
    max_gpu_memory_gb=4.0,
    offload_to_cpu=True,
)
```

**Key Parameters:**

- **Model Architecture**
  - `vocab_size`: Vocabulary size
  - `hidden_size`: Hidden dimension
  - `num_experts`: Total number of experts
  - `num_active_experts`: Number of active experts (k)
  - `intermediate_size`: FFN intermediate dimension

- **Intrinsic Learning**
  - `alpha_surprise`: Weight for surprise loss
  - `beta_temporal`: Weight for temporal consistency
  - `gamma_disagreement`: Weight for expert disagreement
  - `delta_compression`: Weight for compression progress
  - `learn_intrinsic_weights`: Whether to learn weights

- **Neuroplasticity**
  - `exploration_end`: End of exploration phase (fraction)
  - `stabilization_end`: End of stabilization phase (fraction)
  - `max_growth_rate`: Maximum neuron growth rate per step
  - `max_pruning_rate`: Maximum neuron pruning rate per step
  - `min_active_params`: Minimum active parameter ratio
  - `max_active_params`: Maximum active parameter ratio

- **Memory Systems**
  - `hippocampus_capacity`: Max experiences in hippocampus
  - `neocortex_capacity`: Max experiences in neocortex
  - `consolidation_threshold`: Loss threshold for consolidation
  - `microsleep_samples`: Samples per micro-sleep

---

### PragnosiaTrainer

Trainer class with continual learning support.

```python
from pragnosia import PragnosiaTrainer

trainer = PragnosiaTrainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    device="cuda",
    output_dir="./outputs",
    logging_steps=100,
    eval_steps=1000,
    save_steps=5000,
)

trainer.train(num_epochs=10)
```

**Parameters:**
- `model`: PragnosiaModel instance
- `config`: PragnosiaConfig instance
- `train_dataloader`: Training data loader
- `val_dataloader`: Validation data loader (optional)
- `optimizer`: PyTorch optimizer (optional, creates AdamW by default)
- `device`: Device to use ("cuda" or "cpu")
- `output_dir`: Directory for checkpoints and logs
- `logging_steps`: Steps between logging
- `eval_steps`: Steps between evaluation
- `save_steps`: Steps between checkpoints

#### Methods

##### `train(num_epochs)`

Train the model for specified epochs.

**Parameters:**
- `num_epochs` (int): Number of training epochs

---

## Component Classes

### HebbianRouter

Hebbian expert router with stability guarantees.

```python
from pragnosia.models.router import HebbianRouter

router = HebbianRouter(
    input_size=768,
    num_experts=8,
    num_active_experts=2,
    learning_rate=0.01,
    lateral_inhibition=0.1,
)
```

#### Methods

##### `forward(hidden_states, return_routing_weights=False)`

Route inputs to experts.

**Returns:**
- `features`: Projected features
- `selected_experts`: List of expert indices
- `routing_weights`: Optional routing weights

##### `hebbian_update(features, expert_errors)`

Update routing scores using Hebbian learning.

##### `check_stability()`

Check routing stability metrics.

**Returns:**
- Dictionary with entropy, balance, variance, and stability flag

---

### ExpertModule

Single expert in the mixture.

```python
from pragnosia.models.expert import ExpertModule

expert = ExpertModule(
    hidden_size=768,
    intermediate_size=3072,
    dropout=0.1,
    expert_id=0,
)
```

#### Methods

##### `forward(x)`

Process input through expert.

##### `grow_neurons(growth_rate)`

Add neurons during exploration phase.

##### `prune_neurons(pruning_rate)`

Remove neurons during stabilization phase.

##### `to_cpu()` / `to_gpu(device)`

Move expert between CPU and GPU.

---

### IntrinsicObjective

Multi-dimensional intrinsic learning objective.

```python
from pragnosia.losses.intrinsic import IntrinsicObjective

intrinsic = IntrinsicObjective(
    alpha=0.3,
    beta=0.25,
    gamma=0.25,
    delta=0.2,
    learn_weights=True,
)
```

#### Methods

##### `forward(current_hidden, previous_hidden, expert_outputs, prediction_logits, targets, same_context)`

Compute intrinsic loss.

**Returns:**
- `total_loss`: Combined intrinsic loss
- `component_losses`: Dictionary of individual components

---

### Hippocampus

Fast episodic memory system.

```python
from pragnosia.memory.hippocampus import Hippocampus

hippo = Hippocampus(
    capacity=10000,
    batch_size=32,
    consolidation_threshold=0.7,
)
```

#### Methods

##### `store(hidden_states, targets, loss, surprise, metadata)`

Store new experience.

##### `sample(batch_size, prioritized=True)`

Sample experiences for replay.

##### `get_consolidation_candidates(threshold)`

Get experiences ready for consolidation.

---

### Neocortex

Slow structured memory system.

```python
from pragnosia.memory.neocortex import Neocortex

neo = Neocortex(
    capacity=50000,
    hidden_size=768,
    num_clusters=32,
)
```

#### Methods

##### `consolidate(experiences, replay_model)`

Consolidate experiences from hippocampus.

##### `retrieve(query, k)`

Retrieve relevant experiences based on query.

---

### PlasticityScheduler

Neuroplasticity phase scheduler.

```python
from pragnosia.utils.plasticity import PlasticityScheduler

scheduler = PlasticityScheduler(
    total_steps=10000,
    exploration_end=0.3,
    stabilization_end=0.7,
)
```

#### Methods

##### `step()`

Advance scheduler and return current phase.

##### `can_grow()` / `can_prune()`

Check if operations are permitted.

##### `get_growth_rate()` / `get_pruning_rate()`

Get current rates.

##### `check_safety_bounds(active_param_ratio, expert_entropy)`

Verify safety constraints.

---

### HomeostasisRegulator

Homeostatic regulation for stability.

```python
from pragnosia.utils.homeostasis import HomeostasisRegulator

homeostasis = HomeostasisRegulator(
    lambda_uncertainty=0.1,
    lambda_surprise=0.1,
    lambda_churn=0.05,
    target_entropy=2.0,
)
```

#### Methods

##### `forward(prediction_entropy, surprise, expert_activations)`

Compute homeostatic penalty.

---

## Utilities

### Memory Statistics

```python
# Get memory usage
stats = model.get_memory_statistics()

print(f"Router: {stats['router_mb']:.2f} MB")
print(f"Active Experts: {stats['active_experts_mb']:.2f} MB")
print(f"Hippocampus: {stats['hippocampus_mb']:.2f} MB")
print(f"Neocortex: {stats['neocortex_mb']:.2f} MB")
print(f"Total: {stats['total_mb']:.2f} MB")
```

### Routing Statistics

```python
# Check routing stability
routing_stats = model.router.check_stability()

print(f"Entropy: {routing_stats['routing_entropy']:.4f}")
print(f"Balance: {routing_stats['expert_balance']:.4f}")
print(f"Stable: {routing_stats['is_stable']}")
```

### Plasticity Statistics

```python
# Get plasticity info
plasticity_stats = scheduler.get_statistics()

print(f"Phase: {plasticity_stats['phase']}")
print(f"Can grow: {plasticity_stats['can_grow']}")
print(f"Can prune: {plasticity_stats['can_prune']}")
print(f"Safety violations: {plasticity_stats['safety_violations']}")
```

## Example Usage

### Basic Training

```python
from pragnosia import PragnosiaModel, PragnosiaTrainer
from pragnosia.utils.config import PragnosiaConfig
from torch.utils.data import DataLoader

# Configuration
config = PragnosiaConfig()

# Model
model = PragnosiaModel(config)

# Trainer
trainer = PragnosiaTrainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
)

# Train
trainer.train(num_epochs=10)
```

### Custom Training Loop

```python
from pragnosia import PragnosiaModel
from pragnosia.utils.plasticity import PlasticityScheduler

model = PragnosiaModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = PlasticityScheduler(total_steps=10000)

for step, batch in enumerate(dataloader):
    # Forward
    outputs = model(batch['input_ids'], labels=batch['labels'])
    loss = outputs['loss']

    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Neuroplasticity
    phase = scheduler.step()
    if scheduler.can_grow():
        for expert in model.experts:
            expert.grow_neurons(scheduler.get_growth_rate())
    elif scheduler.can_prune():
        for expert in model.experts:
            expert.prune_neurons(scheduler.get_pruning_rate())
```

### Inference

```python
model.eval()

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs['logits']
    predictions = torch.argmax(logits, dim=-1)
```
