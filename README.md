# Pragnosia

**A Brain-Inspired, Energy-Efficient Multimodal Large Language Model**

Pragnosia is a modular, brain-like architecture designed to run, fine-tune, and perform inference on consumer-grade hardware (4GB GPU, 16-24GB RAM) while scaling architecturally from 1B to 450B parameters through sparsity, routing, and plasticity.

## Core Principle

> **Total parameters are large, but active parameters per forward pass are small.**

## Vision

Pragnosia is not an attempt to outscale frontier models. It is an attempt to **redefine efficiency, adaptability, and accessibility** by designing an LLM that behaves more like a brain than a static matrix.

## Key Features

- **Brain-Inspired Architecture**: Modular network with specialized "lobes" or experts, akin to brain regions
- **Sparse Mixture-of-Experts (MoE)**: Only 1-2 experts active per token, enabling 60-450B parameter models to run efficiently
- **Dynamic Plasticity**: Neural growth and pruning - the model can grow or shrink during training
- **Multimodal Integration**: Unified handling of text, images, and audio
- **Energy-Efficient**: Event-driven computation similar to spiking neural networks
- **4GB GPU First**: Designed from the ground up for consumer hardware

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Router                         │
│       (modality + intent + energy budget aware)         │
└───────────────┬───────────────────────┬─────────────────┘
                │                       │
        ┌───────▼───────┐       ┌───────▼────────┐
        │  Text Cortex  │       │ Vision Cortex  │
        │  (Tokenizer + │       │  (ViT / CNN)   │
        │  Embeddings)  │       │                │
        └───────┬───────┘       └───────┬────────┘
                │                       │
                └──────────┬────────────┘
                           ▼
              ┌─────────────────────────────┐
              │    Thalamus Router (MoE)    │
              │    - token routing          │
              │    - energy gating          │
              │    - expert selection       │
              └───────────┬─────────────────┘
                          ▼
     ┌─────────────────────────────────────────────────┐
     │           Cognitive Cortex (MoE)                │
     │  ┌────────┐  ┌────────┐  ┌─────────┐           │
     │  │ Reason │  │ Memory │  │Planning │    ...    │
     │  │ Expert │  │ Expert │  │ Expert  │           │
     │  └────────┘  └────────┘  └─────────┘           │
     │         (only 1-2 experts active/token)        │
     └───────────┬─────────────────────────────────────┘
                 ▼
     ┌─────────────────────────────────────────────────┐
     │              Plasticity Engine                  │
     │    - expert growth    - expert pruning          │
     │    - sparsity enforcement                       │
     └───────────┬─────────────────────────────────────┘
                 ▼
     ┌─────────────────────────────────────────────────┐
     │               Output Cortex                     │
     │    - text logits    - multimodal decoding       │
     └─────────────────────────────────────────────────┘
```

## Brain Analog Mapping

| Brain Concept | Pragnosia Module |
|---------------|------------------|
| Thalamus | Token / Expert Router |
| Cortex regions | MoE Experts |
| Neuroplasticity | Dynamic expert growth/prune |
| Attention | Sparse routing |
| Metabolic budget | Energy-aware gating |
| Memory | KV cache + memory expert |

## Memory Optimization Techniques

Pragnosia employs multiple strategies to enable training and inference on consumer GPUs:

1. **Layer-wise Streaming (AirLLM)**: Load one Transformer layer at a time, requiring only ~1-2GB at any moment
2. **ZeRO-3 Offloading**: Shard optimizer states and gradients across CPU RAM
3. **4-bit Quantization**: QLoRA and bitsandbytes for memory-efficient training
4. **Gradient Checkpointing**: Recompute activations instead of storing them (~50% memory savings)
5. **LoRA Adapters**: Train only a small low-rank subspace of parameters

## Expert Types

| Expert | Function |
|--------|----------|
| Language | Syntax & fluency |
| Reasoning | Math & logic |
| Memory | Long-context recall |
| Planning | Step-by-step reasoning |

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (4GB+ VRAM)
- 16-24GB System RAM

### Installation

```bash
git clone git@github.com:ladkabiharihai/Pragnosia.git
cd Pragnosia
pip install -r requirements.txt
```

### Quick Start

```python
from pragnosia import Pragnosia

# Load model with 4-bit quantization
model = Pragnosia.from_pretrained("pragnosia-v0.1", load_in_4bit=True)

# Text generation
response = model.generate("Explain quantum computing in simple terms")

# Multimodal (text + image)
response = model.generate("Describe this image", image="path/to/image.jpg")
```

## Project Structure

```
pragnosia/
├── pragnosia/
│   ├── __init__.py
│   ├── config.py              # Model configuration
│   ├── model.py               # Main Pragnosia model
│   ├── modules/
│   │   ├── input_router.py    # Input routing logic
│   │   ├── text_cortex.py     # Text encoding
│   │   ├── vision_cortex.py   # Vision encoding
│   │   ├── thalamus.py        # MoE router
│   │   ├── experts.py         # Expert implementations
│   │   ├── plasticity.py      # Growth/pruning engine
│   │   └── output_cortex.py   # Output decoding
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   ├── data.py            # Data loading
│   │   └── curriculum.py      # Curriculum learning
│   └── inference/
│       ├── generate.py        # Generation utilities
│       └── quantize.py        # Quantization helpers
├── configs/                   # Model configurations
├── scripts/                   # Training/eval scripts
├── tests/                     # Unit tests
├── TODO.md                    # Development progress
└── README.md
```

## Development Roadmap

### Week 1 - Core Skeleton
- Repository setup and config system
- Tokenizer + base Transformer
- MoE layer + Thalamus router
- CPU/GPU offloading and gradient checkpointing

### Week 2 - Plasticity & Energy Awareness
- Expert activation tracking
- Energy budget simulation
- Expert growth/pruning logic

### Week 3 - Multimodal & Scaling
- Vision encoder integration
- Scale to 3-7B total parameters
- LoRA-only training

### Week 4 - Data, Training & Evaluation
- Dataset preprocessing
- Benchmarking vs GPT-OSS-20B
- Quantized release and documentation

## Expected Outcomes

| Dimension | Result |
|-----------|--------|
| Raw fluency | Slightly below GPT-OSS-20B |
| Reasoning | Competitive |
| Efficiency | Orders of magnitude better |
| Hardware accessibility | 4GB-first |
| Research novelty | Very high |

## Data Strategy

### Text Data
- **FineWeb-Edu**: High-quality language
- **OpenWebMath**: Reasoning
- **Code datasets**: Logic & structure
- **Wikipedia + Books**: Knowledge

### Multimodal Data
- **LAION-Aesthetics**: Vision grounding
- **COCO**: Image-caption alignment
- **OCR datasets**: Text-vision fusion

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## References

1. [Multi-Lobar ANN](https://www.nature.com/articles/s41598-024-84325-z) - Brain-inspired architecture
2. [MiCRo](https://arxiv.org/html/2506.13331v2) - Mixture of Cognitive Reasoners
3. [Mixture of Experts Explained](https://huggingface.co/blog/moe) - Hugging Face
4. [AirLLM](https://huggingface.co/blog/lyogavin/airllm) - 70B on 4GB GPU
5. [LSP-Offload](https://arxiv.org/html/2406.10181v1) - Practical fine-tuning on commodity GPU
6. [Molmo VLM](https://allenai.org/blog/molmo) - Multimodal architecture
7. [Spiking Neural Networks](https://arxiv.org/abs/2510.27379) - Brain-inspired computing

## License

[TBD]

## Acknowledgments

This project draws on recent advances in sparse and efficient networks, brain-inspired computing, and open-source LLM research.

---

*Pragnosia: Redefining efficiency, adaptability, and accessibility in language models.*
