# Pragnosia v0.1 - Development TODO

## Overview
Brain-inspired, energy-efficient, modular multimodal LLM targeting 4GB GPU inference.

**Core Principle:** Total parameters are large, but active parameters per forward pass are small.

---

## Week 1 - Core Skeleton (COMPLETED)
**Goal:** Runnable text-only Pragnosia (~1B sparse MoE model)

- [x] **Repository Setup**
  - [x] Project structure and directory layout
  - [x] Configuration system (YAML/JSON configs)
  - [x] Dependency management (pyproject.toml)
  - [x] Development environment setup

- [x] **Tokenizer + Base Transformer**
  - [x] Text Cortex with embeddings
  - [x] Base Transformer block implementation
  - [x] Positional embeddings (RoPE)
  - [x] Layer normalization (RMSNorm)

- [x] **MoE Layer + Thalamus Router**
  - [x] Expert module implementation
  - [x] Thalamus router with Top-K selection
  - [x] Load balancing loss (aux_loss)
  - [x] Router z-loss for stability
  - [x] Token-to-expert routing logic

- [x] **Memory Optimization**
  - [x] CPU/GPU offloading (Accelerate support)
  - [x] Gradient checkpointing
  - [x] 4-bit weight loading (bitsandbytes)
  - [x] FlashAttention integration (via scaled_dot_product_attention)

**Deliverable:** ~125M-350M sparse MoE model with inference on 4GB GPU - COMPLETE

---

## Week 2 - Plasticity & Energy Awareness (COMPLETED)
**Goal:** Dynamic expert creation/deletion, Active params << Total params

- [x] **Expert Activation Tracking**
  - [x] Per-expert activation counters
  - [x] Routing entropy monitoring
  - [x] Expert utilization metrics

- [x] **Energy Budget System**
  - [x] Energy budget simulation
  - [x] Hard gating enforcement
  - [x] Energy-aware routing decisions

- [x] **Plasticity Engine**
  - [x] Expert growth logic (high entropy, loss spikes, saturation)
  - [x] Expert pruning logic (low activation, redundant gradients)
  - [x] Dynamic architecture adaptation

**Deliverable:** Dynamic expert creation/deletion working - COMPLETE

---

## Week 3 - Multimodal & Scaling
**Goal:** Multimodal Pragnosia v0.1 (3-7B total params)

- [x] **Vision Cortex**
  - [x] Vision encoder integration (basic ViT)
  - [x] MLP connector for image tokens
  - [ ] Image-text alignment training

- [ ] **Scaling**
  - [ ] Scale to 3-7B total parameters
  - [x] LoRA adapter implementation
  - [x] Memory expert for long context

- [x] **Text Cortex Refinement**
  - [x] Basic tokenization support
  - [x] Embedding initialization

**Deliverable:** Multimodal Pragnosia v0.1

---

## Week 4 - Data, Training & Evaluation
**Goal:** Open-source Pragnosia v0.1 with reproducible 4GB pipeline

- [ ] **Data Pipeline**
  - [ ] Dataset preprocessing scripts
  - [ ] GneissWeb integration
  - [ ] OpenWebMath for reasoning
  - [ ] The Stack V2 for coding
  - [ ] COCO/LAION for multimodal

- [ ] **Training**
  - [ ] Curriculum training implementation
  - [ ] Fine-tuning pipeline
  - [ ] Distributed training support (optional)

- [ ] **Evaluation**
  - [ ] Benchmark suite setup
  - [ ] Comparison vs GPT-OSS-20B tasks
  - [ ] Memory/inference profiling

- [x] **Release**
  - [ ] Quantized model weights
  - [x] Documentation (README)
  - [ ] Docker/pip distribution
  - [x] Example inference scripts

**Deliverable:** Open-source Pragnosia v0.1

---

## Architecture Components Status

| Component | Status | Description |
|-----------|--------|-------------|
| Input Router | [x] Complete | Modality detection, complexity estimation, energy budget |
| Text Cortex | [x] Complete | Tokenizer + Embeddings |
| Vision Cortex | [x] Complete | Basic ViT encoder with projection |
| Thalamus Router | [x] Complete | Token routing, energy gating, expert selection |
| Cognitive Cortex | [x] Complete | MoE experts (Language, Reasoning, Memory, Planning) |
| Plasticity Engine | [x] Complete | Expert growth/pruning |
| Output Cortex | [x] Complete | Text logits, multimodal decoding |

---

## Brain Analog Mapping

| Brain Concept | Pragnosia Module |
|---------------|------------------|
| Thalamus | Token / Expert Router |
| Cortex regions | MoE Experts |
| Neuroplasticity | Dynamic expert growth/prune |
| Attention | Sparse routing |
| Metabolic budget | Energy-aware gating |
| Memory | KV cache + memory expert |

---

## Key Techniques Implemented

- [x] Sparse Mixture-of-Experts (MoE)
- [x] Layer-wise streaming / offload support
- [x] ZeRO/Accelerate sharding support
- [x] 4-bit quantization (bitsandbytes)
- [x] Gradient checkpointing
- [x] LoRA adapters
- [x] Dynamic plasticity (grow/prune)
- [x] Energy-aware computation

---

## Test Coverage

- 86 tests passing
- Tests cover:
  - All core modules (attention, MLP, normalization)
  - MoE routing and expert selection
  - Plasticity engine logic
  - Memory optimization utilities
  - Full model forward/backward pass
  - Text generation
  - GPU compatibility
  - LoRA adapters and fine-tuning
  - Memory expert for long context

---

## References

- Multi-Lobar ANN (Alibrahim et al., 2025)
- MiCRo - Mixture of Cognitive Reasoners
- Hugging Face MoE Guide
- AirLLM - 70B on 4GB GPU
- LSP-Offload for fine-tuning
- Spiking Neural Networks research
- Molmo VLM architecture

---

*Last Updated: 2026-01-15*
