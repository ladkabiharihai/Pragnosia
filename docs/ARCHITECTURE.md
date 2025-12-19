# Pragnosia Architecture Documentation

## Overview

Pragnosia is a neuroscience-inspired systems architecture for continual representation learning. This document provides detailed technical information about each component and how they work together.

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Input Layer                              │
│                  (Token + Position Embeddings)                  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                   Hebbian Router (~100MB)                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Feature Projection: φ(x) → router_size                  │ │
│  │  Routing Scores: r_e for each expert                     │ │
│  │  Lateral Inhibition: a_e = ReLU(r_e - θ - Σ w·a)        │ │
│  │  Top-k Selection: Select num_active_experts              │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                Expert Pool (CPU Storage)                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │Expert 0 │  │Expert 1 │  │Expert 2 │  │Expert 3 │  ...     │
│  │~500MB   │  │~500MB   │  │~500MB   │  │~500MB   │          │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
│       ▲             ▲                                           │
│       └─────────────┘ k=2 loaded to GPU                        │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│              Expert Processing & Combination                    │
│  - Process input through active experts                        │
│  - Weight outputs by routing weights                           │
│  - Combine: output = Σ w_e · expert_e(x)                       │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    Intrinsic Learning                           │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  L_surprise = Σ -log p(x) · ||h_t - f(h_{t-1})||²       │ │
│  │  L_temporal = ||h_t - sg(h_{t-1})||² · I[same_context]  │ │
│  │  L_disagreement = Var_e[f_e(x)] · I[high_uncertainty]   │ │
│  │  L_compression = ΔL_pred / Δt                            │ │
│  │                                                           │ │
│  │  L_total = α·L_surprise + β·L_temporal +                │ │
│  │            γ·L_disagreement + δ·L_compression            │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                  Memory Systems                                 │
│  ┌────────────────────┐        ┌──────────────────────┐        │
│  │   Hippocampus      │───────▶│     Neocortex        │        │
│  │   (~50MB)          │        │     (~100MB)         │        │
│  │                    │ Micro- │                      │        │
│  │ - FIFO queue       │ Sleep  │ - Semantic clusters  │        │
│  │ - Priority-based   │        │ - Structured memory  │        │
│  │ - Fast episodic    │        │ - Slow consolidation │        │
│  └────────────────────┘        └──────────────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Hebbian Router

**Purpose**: Route inputs to appropriate experts using local, reward-free learning.

**Implementation Details**:
- **Input**: Hidden states (batch, seq_len, hidden_size)
- **Output**: Selected expert indices and routing weights
- **Memory**: ~100MB, always GPU-resident
- **Update Rule**: `Δr_e = η_r · corr(φ(x), ΔL_e)`

**Key Features**:
- Local learning (no backpropagation through routing)
- Lateral inhibition for competition
- Provable stability under stationary distributions
- Entropy tracking for load balancing

**Code Location**: `src/pragnosia/models/router.py`

**Stability Proposition**:
Under stationary input distributions, routing scores converge to stable fixed points through:
1. Bounded error reduction (loss cannot be negative)
2. Lateral inhibition (prevents multiple experts from dominating)
3. Homeostatic penalties (suppress runaway routing dominance)

### 2. Expert Modules

**Purpose**: Specialized sub-networks for different input patterns.

**Implementation Details**:
- **Architecture**: 2-layer FFN with GELU activation
- **Size**: ~500MB each
- **Storage**: CPU (inactive), GPU (active)
- **Loading**: Asynchronous, on-demand

**Neuroplasticity**:
- **Growth Phase**: Add neurons during exploration (0-30%)
- **Pruning Phase**: Remove low-importance neurons during stabilization (30-70%)
- **Exploitation Phase**: Fixed architecture (70-100%)

**Pruning Strategy**:
- Importance = weight magnitude (L1 norm)
- Prune least important neurons
- Maintain safety bounds (30-90% active parameters)

**Code Location**: `src/pragnosia/models/expert.py`

### 3. Multi-Dimensional Intrinsic Learning

**Purpose**: Drive representation learning without external rewards.

**Components**:

#### 3.1 Surprise-Weighted Prediction Error
```python
L_surprise = Σ_t -log p(x_t) · ||h_t - f(h_{t-1})||²
```
- Focuses learning on unexpected inputs
- Implements curiosity-driven prioritization
- Higher surprise → stronger learning signal

#### 3.2 Temporal Consistency Loss
```python
L_temporal = ||h_t - sg(h_{t-1})||² · I[same_context]
```
- Encourages stable representations within contexts
- Permits discontinuities at context boundaries
- Stop-gradient prevents temporal credit assignment issues

#### 3.3 Cross-Expert Disagreement
```python
L_disagreement = Var_e[f_e(x)] · I[high_uncertainty]
```
- Internal cognitive conflict as learning signal
- High disagreement indicates uncertainty
- Ensemble diversity encourages exploration

#### 3.4 Compression Progress
```python
L_compression = ΔL_pred / Δt
```
- Reward improvement in prediction ability
- Schmidhuber's compression progress principle
- Measures learning velocity

**Weight Learning**:
- α, β, γ, δ are learnable parameters (default)
- Can be fixed for ablation studies
- Initialized to balanced values (0.25-0.3)

**Code Location**: `src/pragnosia/losses/intrinsic.py`

### 4. Memory Systems

#### 4.1 Hippocampus (Fast Episodic Memory)

**Purpose**: Store recent experiences for quick replay.

**Implementation**:
- **Capacity**: 10,000 experiences (~50MB)
- **Structure**: FIFO queue with priority
- **Sampling**: Priority-based (higher loss/surprise)
- **Consolidation**: Transfer to neocortex when well-learned

**Priority Calculation**:
```python
priority = base + loss + surprise
```

**Consolidation Criteria**:
- Replay count ≥ 3 (well-rehearsed)
- Loss < threshold (successfully learned)

**Code Location**: `src/pragnosia/memory/hippocampus.py`

#### 4.2 Neocortex (Slow Structured Memory)

**Purpose**: Store consolidated knowledge with semantic organization.

**Implementation**:
- **Capacity**: 50,000 experiences (~100MB)
- **Structure**: Semantic clusters (k-means style)
- **Retrieval**: Nearest-neighbor based on query
- **Organization**: Cluster centroids updated incrementally

**Cluster Assignment**:
1. Compute hidden state centroid for experience
2. Find nearest cluster by Euclidean distance
3. Add to cluster (replace oldest if full)
4. Update cluster centroid

**Code Location**: `src/pragnosia/memory/neocortex.py`

### 5. Asynchronous Micro-Sleep Consolidation

**Purpose**: Transfer memories from hippocampus to neocortex with minimal latency.

**Strategy**:

#### Micro-Sleep (Every Forward Pass)
- Transfer 1-2 samples per forward pass
- <1ms overhead
- Background CUDA stream (optional)
- Maintains continuous learning

#### Intensive Consolidation (Periodic)
- Full transfer every N steps (default: 1000)
- Triggered by distribution shift or error spike
- Processes all consolidation candidates
- ~30ms overhead (amortized)

**Implementation**:
```python
# Micro-sleep
if step % microsleep_interval == 0:
    candidates = hippocampus.get_consolidation_candidates()
    neocortex.consolidate(candidates[:2])  # 1-2 samples

# Intensive consolidation
if step % intensive_interval == 0:
    candidates = hippocampus.get_consolidation_candidates()
    neocortex.consolidate(candidates)  # All candidates
```

**Code Location**: `src/pragnosia/models/pragnosia_model.py` (lines 280-300)

### 6. Controlled Neuroplasticity

**Purpose**: Safely grow and prune network structure across developmental phases.

**Phases**:

| Phase | Operations | Safety Bounds |
|-------|-----------|---------------|
| **Exploration** (0-30%) | ✓ Growth<br>✗ Pruning | Active params: [30%, 90%]<br>Entropy: > 2.0<br>Growth rate: ≤ 1% |
| **Stabilization** (30-70%) | ✗ Growth<br>✓ Pruning | Active params: [30%, 90%]<br>Entropy: > 2.0<br>Pruning rate: ≤ 1% |
| **Exploitation** (70-100%) | ✗ Growth<br>✗ Pruning | Active params: [30%, 90%]<br>Entropy: > 2.0 |

**Safety Monitoring**:
- Check bounds every step
- Log violations (should be 0)
- Halt growth/pruning if violation detected

**Code Location**: `src/pragnosia/utils/plasticity.py`

### 7. Homeostatic Regulation

**Purpose**: Maintain operational stability through metabolic constraints.

**Objective**:
```python
L_homeostatic = λ_u·H(uncertainty) + λ_s·H(surprise) + λ_c·churn(experts)
```

**Components**:

1. **Uncertainty Penalty**: Deviation from target entropy
   - Too low → overconfident
   - Too high → underconfident
   - Target: ~2.0 nats

2. **Surprise Penalty**: Excessive surprise is costly
   - Use squared surprise for extreme value sensitivity
   - Penalizes unprepared model state

3. **Churn Penalty**: Expert switching overhead
   - Loading/unloading has cost
   - Penalize frequent changes
   - Promotes stable routing

**Code Location**: `src/pragnosia/utils/homeostasis.py`

### 8. Perceptual Distillation (Multimodal)

**Purpose**: Transfer knowledge from large frozen encoders to lightweight students.

**Application**:
- Vision: CLIP → Lightweight CNN
- Audio: Whisper → Lightweight Conv

**Loss**:
```python
L_distill = ||f_student(x) - sg(f_teacher(x))||²
```

**Deployment**:
- Training: Use teacher for distillation
- Inference: Use student only (~30-50MB)
- Memory savings: 8-13×

**Code Location**: `src/pragnosia/losses/distillation.py`

## Memory Budget

| Component | Size | Location | Notes |
|-----------|------|----------|-------|
| Token Embeddings | ~100MB | GPU | Shared with output head |
| Router | ~100MB | GPU | Always resident |
| Active Experts (2) | ~1000MB | GPU | Dynamic loading |
| Hippocampus | ~50MB | GPU | FIFO queue |
| Neocortex | ~100MB | GPU | Semantic clusters |
| Inactive Experts (6) | ~3000MB | CPU | On-demand loading |
| **Total Active** | **~2.8GB** | **GPU** | **Fits in 4GB** |
| **Total Model** | **~4.1B params** | **Mixed** | **Equivalent to full 4B model** |

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Router Forward | O(1) | Fixed size network |
| Expert Forward | O(k) | k active experts (k=2) |
| Expert Loading | O(1) | Asynchronous, amortized |
| Memory Consolidation | O(1) | Micro-batches (1-2 samples) |
| **Total Inference** | **O(k)** | **Linear in active experts** |

## Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        # Hebbian router update (no gradient)
        model.router.hebbian_update(features, expert_errors)

        # Neuroplasticity
        if plasticity.can_grow():
            model.grow_neurons(rate)
        elif plasticity.can_prune():
            model.prune_neurons(rate)

        # Memory consolidation (async)
        model.micro_sleep_consolidation()

        # Safety checks
        violations = plasticity.check_safety_bounds(
            active_param_ratio,
            expert_entropy
        )
```

## Design Decisions

### Why Hebbian Routing?
- **Local**: No backpropagation through router
- **Efficient**: Low computational overhead
- **Stable**: Provable convergence guarantees
- **Scalable**: Independent expert updates

### Why Multi-Dimensional Intrinsic Learning?
- **Complementary**: Each component captures different aspects
- **Robust**: No single failure point
- **Learnable**: Weights adapt to task
- **Unsupervised**: No external reward needed

### Why Complementary Memory Systems?
- **Fast + Slow**: Balances quick learning and stable knowledge
- **Interference Management**: Hippocampus prevents catastrophic forgetting
- **Consolidation**: Gradual transfer preserves important memories
- **Neuroscience-Inspired**: Mimics biological memory systems

### Why Dynamic Expert Loading?
- **Memory Efficient**: Only k experts on GPU
- **Scalable**: Can handle hundreds of experts
- **Flexible**: Trade-off between memory and computation
- **Practical**: Enables deployment on consumer hardware

## Limitations and Future Work

### Current Limitations
1. **Scale**: Tested up to 4B parameters (simulated to 64B)
2. **Alignment**: No RLHF integration yet
3. **Modalities**: Vision/audio encoders are simple
4. **Biological Realism**: Sacrificed for engineering tractability

### Future Directions
1. **Large-Scale Validation**: Train at 7B+ scale
2. **RLHF Integration**: Combine intrinsic + alignment objectives
3. **Hierarchical Experts**: Experts-of-experts for extreme scale
4. **Real-Time Deployment**: Interactive systems with continuous learning
5. **Advanced Multimodal**: Better vision/audio integration

## References

See paper bibliography for complete references to:
- Hebbian learning and BCM theory
- Intrinsic motivation literature
- Continual learning systems
- Mixture of Experts architectures
- Complementary learning systems theory
