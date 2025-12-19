# Neuromodulatory Gating for Adaptive Intrinsic Learning

## Overview

Pragnosia now includes **brain-inspired neuromodulation** that dynamically adjusts the strength of intrinsic learning based on the system's current state. This prevents the intrinsic learning collapse problem where curiosity-driven learning shuts down once the model stabilizes.

## The Problem

Without neuromodulation, the model can fall into two failure modes:

1. **Intrinsic Collapse**: When the model stabilizes, intrinsic signals (surprise, disagreement, compression) naturally decrease. The optimizer exploits this by pushing intrinsic weights toward zero as a "shortcut" to minimize total loss.

2. **Over-Stabilization**: The model reaches a local minimum and stops exploring, missing better solutions.

## The Solution: Neuromodulation

Inspired by dopamine and norepinephrine in the brain, the neuromodulator tracks:

- **Novelty**: How much the current state deviates from recent behavior
- **Stability**: Inverse of recent variance (low variance = high stability)
- **Improvement**: Whether the model is still learning or stuck

The modulation signal adjusts intrinsic learning strength:

```
modulation = sigmoid(α * novelty - β * stability + γ * stagnation)
```

Where:
- High novelty → increase intrinsic (explore new patterns)
- High stability → increase intrinsic (prevent collapse)
- Stagnation → increase intrinsic (escape local minima)

## Architecture

### Neuromodulator Module

Location: `/src/pragnosia/utils/neuromodulation.py`

**Key Components:**

1. **History Buffers** (circular buffers of length 100):
   - `loss_history`: Track prediction loss over time
   - `intrinsic_history`: Track intrinsic loss over time
   - `surprise_history`: Track surprise signals

2. **Running Statistics** (EMA with τ=0.9):
   - `ema_loss`: Smoothed average loss
   - `ema_intrinsic`: Smoothed average intrinsic signal
   - `ema_variance`: Smoothed variance (stability indicator)

3. **Modulation Output**:
   - Range: [0.5, 2.0] (never completely shuts off curiosity)
   - Smoothed over time to prevent sudden jumps

### Integration

The neuromodulator is integrated into the `IntrinsicObjective`:

```python
# In intrinsic.py
if self.neuromodulator is not None:
    modulation = self.neuromodulator(
        intrinsic_loss=total_loss,
        prediction_loss=pred_loss,
        surprise=surprise,
    )
    total_loss = total_loss * modulation
```

## Configuration

Enable/disable neuromodulation in `PragnosiaConfig`:

```python
config = PragnosiaConfig(
    use_neuromodulation=True,  # Default: True
    # ... other config
)
```

Tune neuromodulator hyperparameters:

```python
neuromodulator = Neuromodulator(
    history_length=100,        # History buffer size
    novelty_weight=2.0,        # α: Novelty sensitivity
    stability_weight=1.0,      # β: Stability sensitivity
    error_weight=1.5,          # γ: Stagnation sensitivity
    baseline_modulation=1.0,   # Starting modulation value
    tau=0.9,                   # EMA smoothing factor
)
```

## Monitoring

Neuromodulation statistics are logged every 100 steps:

```
Step 100 - Intrinsic components: surprise=0.0000, temporal=0.4187, ...
Step 100 - Neuromodulation: modulation=1.2000, novelty=0.0523, stability=0.8234, ema_loss=0.2341, ema_intrinsic=0.1234, ema_variance=0.0042
```

**Key Metrics:**

- `modulation`: Current multiplier for intrinsic loss (0.5 to 2.0)
  - < 1.0: Reducing intrinsic (model is learning well)
  - = 1.0: Baseline (neutral)
  - > 1.0: Boosting intrinsic (preventing collapse or stagnation)

- `novelty`: Deviation from recent behavior
  - High: Model encountering new patterns
  - Low: Model behavior is predictable

- `stability`: Inverse of variance
  - High: Model is stable (risk of over-stabilization)
  - Low: Model is chaotic

- `ema_loss`: Smoothed prediction loss
- `ema_intrinsic`: Smoothed intrinsic loss
- `ema_variance`: Smoothed variance metric

## Expected Behavior

### Phase 1: Exploration (0-30% of training)
- High novelty → modulation ≈ 1.2-1.5
- Model actively exploring representations
- Intrinsic signals strong

### Phase 2: Stabilization (30-70% of training)
- Decreasing novelty → modulation ≈ 0.8-1.2
- Model consolidating knowledge
- Neuromodulator prevents collapse

### Phase 3: Exploitation (70-100% of training)
- High stability → modulation ≈ 1.0-1.3
- Without neuromodulation: intrinsic would collapse
- With neuromodulation: maintains baseline curiosity

## Benefits

1. **Prevents Intrinsic Collapse**: Even when model stabilizes, curiosity remains active

2. **Adaptive Exploration**: Automatically increases exploration when stuck

3. **Stable Training**: Smooth modulation changes prevent instability

4. **Brain-Like Learning**: Mimics biological neuromodulatory systems

## Comparison: With vs. Without Neuromodulation

### Without Neuromodulation:
```
Step 1000: intrinsic=0.1234
Step 2000: intrinsic=0.0456
Step 3000: intrinsic=0.0000  ❌ Collapsed!
```

### With Neuromodulation:
```
Step 1000: intrinsic=0.1234, modulation=1.0
Step 2000: intrinsic=0.0456, modulation=1.5  ✓ Boosted!
Step 3000: intrinsic=0.0684, modulation=1.2  ✓ Maintained!
```

## Advanced: Modulation Formula

The complete modulation computation:

```python
# 1. Compute novelty (deviation from recent mean)
novelty = |current_loss - recent_mean| + |current_intrinsic - recent_mean|

# 2. Compute stability (inverse variance)
stability = 1 / (1 + recent_variance)

# 3. Compute improvement trend
improvement = early_loss - late_loss  # Positive = improving

# 4. Combine signals
signal = novelty_weight * novelty +
         stability_weight * stability +
         error_weight * max(0, -improvement)

# 5. Apply sigmoid to bound in [0.5, 2.0]
modulation = 0.5 + 1.5 * sigmoid(signal - 1.0)

# 6. Smooth over time (prevent jumps)
modulation = 0.8 * prev_modulation + 0.2 * modulation
```

## Implementation Details

### Warmup Period

The neuromodulator uses a 10-step warmup:
- Steps 0-9: modulation = 1.0 (baseline)
- Step 10+: Active modulation

This prevents instability at the start of training.

### Gradient Flow

The neuromodulator operates with detached gradients:
- It observes intrinsic and prediction losses
- But does not backpropagate through the modulation
- This is "meta-learning" - learning about learning

### Reset Capability

For continual learning across tasks:

```python
model.intrinsic_objective.neuromodulator.reset()
```

This resets history buffers and statistics when switching to a new task.

## Troubleshooting

### Modulation stays at 1.0
- Check that `use_neuromodulation=True` in config
- Verify step count > 10 (warmup period)
- Increase sensitivity: `novelty_weight=3.0`

### Modulation oscillates wildly
- Decrease sensitivity: `novelty_weight=1.0`
- Increase smoothing: `tau=0.95`
- Increase smoothing in modulation update: change `0.8` to `0.9` in code

### Intrinsic still collapses
- Check that floor is applied: `torch.clamp(total_loss, 0.01, 100.0)`
- Verify weight clamping: `torch.clamp(self.alpha, 0.05, 1.0)`
- Increase modulation range: modify sigmoid bounds

## Future Extensions

1. **Task-Specific Modulation**: Different modulation profiles for different task types

2. **Multi-Signal Modulation**: Separate modulators for each intrinsic component

3. **Learned Modulation**: Make modulation parameters learnable

4. **Neuromodulator Attention**: Attend to specific intrinsic components based on context

## References

This implementation is inspired by:
- Dopaminergic reward prediction error signaling in the brain
- Noradrenergic arousal and uncertainty processing
- Meta-learning and learning-to-learn principles
- Computational neuroscience models of curiosity

## Citation

If you use neuromodulation in your research:

```bibtex
@software{pragnosia_neuromodulation,
  title={Neuromodulatory Gating for Continual Learning},
  author={Pragnosia Contributors},
  year={2024},
  url={https://github.com/your-repo/pragnosia}
}
```
