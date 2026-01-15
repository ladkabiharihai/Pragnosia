# Pragnosia v0.1 - Development TODO

## Overview
Brain-inspired, energy-efficient, modular multimodal LLM targeting 4GB GPU inference.

**Core Principle:** Total parameters are large, but active parameters per forward pass are small.

---

## Week 1 - Core Skeleton
**Goal:** Runnable text-only Pragnosia (~1B sparse MoE model)

- [ ] **Repository Setup**
  - [ ] Project structure and directory layout
  - [ ] Configuration system (YAML/JSON configs)
  - [ ] Dependency management (requirements.txt / pyproject.toml)
  - [ ] Development environment setup

- [ ] **Tokenizer + Base Transformer**
  - [ ] Implement/integrate tokenizer (BPE or SentencePiece)
  - [ ] Base Transformer block implementation
  - [ ] Positional embeddings (RoPE or learned)
  - [ ] Layer normalization (RMSNorm)

- [ ] **MoE Layer + Thalamus Router**
  - [ ] Expert module implementation
  - [ ] Thalamus router with Top-K selection
  - [ ] Load balancing loss
  - [ ] Token-to-expert routing logic

- [ ] **Memory Optimization**
  - [ ] CPU/GPU offloading (Accelerate/DeepSpeed)
  - [ ] Gradient checkpointing
  - [ ] 4-bit weight loading (bitsandbytes)
  - [ ] FlashAttention integration

**Deliverable:** ~1B sparse MoE model with inference on 4GB GPU

---

## Week 2 - Plasticity & Energy Awareness
**Goal:** Dynamic expert creation/deletion, Active params << Total params

- [ ] **Expert Activation Tracking**
  - [ ] Per-expert activation counters
  - [ ] Routing entropy monitoring
  - [ ] Expert utilization metrics

- [ ] **Energy Budget System**
  - [ ] Energy budget simulation
  - [ ] Hard gating enforcement
  - [ ] Energy-aware routing decisions

- [ ] **Plasticity Engine**
  - [ ] Expert growth logic (high entropy, loss spikes, saturation)
  - [ ] Expert pruning logic (low activation, redundant gradients)
  - [ ] Dynamic architecture adaptation

**Deliverable:** Dynamic expert creation/deletion working

---

## Week 3 - Multimodal & Scaling
**Goal:** Multimodal Pragnosia v0.1 (3-7B total params)

- [ ] **Vision Cortex**
  - [ ] Vision encoder integration (CLIP-ViT / MobileViT)
  - [ ] MLP connector for image tokens
  - [ ] Image-text alignment training

- [ ] **Scaling**
  - [ ] Scale to 3-7B total parameters
  - [ ] LoRA adapter implementation
  - [ ] Memory expert for long context

- [ ] **Text Cortex Refinement**
  - [ ] Improved tokenization
  - [ ] Better embedding initialization

**Deliverable:** Multimodal Pragnosia v0.1

---

## Week 4 - Data, Training & Evaluation
**Goal:** Open-source Pragnosia v0.1 with reproducible 4GB pipeline

- [ ] **Data Pipeline**
  - [ ] Dataset preprocessing scripts
  - [ ] FineWeb-Edu integration
  - [ ] OpenWebMath for reasoning
  - [ ] COCO/LAION for multimodal

- [ ] **Training**
  - [ ] Curriculum training implementation
  - [ ] Fine-tuning pipeline
  - [ ] Distributed training support (optional)

- [ ] **Evaluation**
  - [ ] Benchmark suite setup
  - [ ] Comparison vs GPT-OSS-20B tasks
  - [ ] Memory/inference profiling

- [ ] **Release**
  - [ ] Quantized model weights
  - [ ] Documentation
  - [ ] Docker/pip distribution
  - [ ] Example inference scripts

**Deliverable:** Open-source Pragnosia v0.1

---

## Architecture Components Status

| Component | Status | Description |
|-----------|--------|-------------|
| Input Router | [ ] Not Started | Modality detection, complexity estimation, energy budget |
| Text Cortex | [ ] Not Started | Tokenizer + Embeddings |
| Vision Cortex | [ ] Not Started | CLIP-ViT / MobileViT encoder |
| Thalamus Router | [ ] Not Started | Token routing, energy gating, expert selection |
| Cognitive Cortex | [ ] Not Started | MoE experts (Language, Reasoning, Memory, Planning) |
| Plasticity Engine | [ ] Not Started | Expert growth/pruning |
| Output Cortex | [ ] Not Started | Text logits, multimodal decoding |

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

## Key Techniques to Implement

- [ ] Sparse Mixture-of-Experts (MoE)
- [ ] Layer-wise streaming / AirLLM offload
- [ ] ZeRO-3 / FSDP sharding
- [ ] 4-bit quantization (QLoRA)
- [ ] Gradient checkpointing
- [ ] LoRA adapters
- [ ] Dynamic plasticity (grow/prune)
- [ ] Event-driven computation

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
