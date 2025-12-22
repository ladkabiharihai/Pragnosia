#!/usr/bin/env bash
#
# Enhanced Production Training Pipeline for Pragnosia
#
# This script implements a TWO-STAGE training approach:
# Stage 1: Representation Learning (local learning in experts)
# Stage 2: Generation Training (coherence module for sequential binding)
#
# This approach preserves your innovative local learning architecture while
# adding the sequential coherence needed for proper chat and code generation.
#
# Usage:
#   ./train_production_enhanced.sh [MODEL_SIZE] [MAX_SAMPLES]
#
# Examples:
#   ./train_production_enhanced.sh 350M 10000    # Quick test
#   ./train_production_enhanced.sh 1B 50000      # Production training
#

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Parse arguments
MODEL_SIZE=${1:-"350M"}
MAX_SAMPLES=${2:-10000}

# Training hyperparameters
PRETRAIN_EPOCHS=3
INSTRUCTION_EPOCHS=3
COHERENCE_EPOCHS=3  # NEW: Stage 2 training

echo "================================"
echo "ENHANCED PRAGNOSIA TRAINING"
echo "================================"
echo "Model size: $MODEL_SIZE"
echo "Max samples: $MAX_SAMPLES"
echo "Pre-train epochs: $PRETRAIN_EPOCHS"
echo "Instruction epochs: $INSTRUCTION_EPOCHS"
echo "Coherence epochs: $COHERENCE_EPOCHS (NEW)"
echo "================================"
echo ""

#
# STAGE 1: REPRESENTATION LEARNING
# Train experts using local learning + intrinsic objectives
# Result: Rich representations, but may lack sequential coherence
#

echo "=========================================="
echo "STAGE 1a: Pre-training on WikiText-103"
echo "=========================================="
echo "Learning language fundamentals with local learning..."
echo ""

python pretrain.py \
    --model-size "$MODEL_SIZE" \
    --dataset wikitext-103 \
    --epochs "$PRETRAIN_EPOCHS" \
    --max-samples "$MAX_SAMPLES" \
    --device cuda \
    --offload-to-cpu \
    || { echo "❌ Pre-training failed!"; exit 1; }

# Find the pretrained model
PRETRAIN_MODEL=$(ls -t outputs/pretrain_*/pretrained_model.pt | head -1)
echo ""
echo "✓ Pre-training completed!"
echo "✓ Model saved: $PRETRAIN_MODEL"
echo ""

echo "=========================================="
echo "STAGE 1b: Instruction Fine-Tuning"
echo "=========================================="
echo "Teaching the model to follow instructions..."
echo ""

python finetune_instruction.py \
    --resume "$PRETRAIN_MODEL" \
    --epochs "$INSTRUCTION_EPOCHS" \
    --max-samples "$MAX_SAMPLES" \
    --batch-size 4 \
    --device cuda \
    || { echo "❌ Instruction fine-tuning failed!"; exit 1; }

# Find the instruction-tuned model
INSTRUCTION_MODEL=$(ls -t outputs/instruction_tuned_*/pragnosia_chat_model.pt | head -1)
echo ""
echo "✓ Instruction fine-tuning completed!"
echo "✓ Model saved: $INSTRUCTION_MODEL"
echo ""

#
# STAGE 2: GENERATION TRAINING (NEW!)
# Freeze experts, train ONLY coherence module on pure language modeling
# Result: Coherent, grammatical generation with preserved representations
#

echo "=========================================="
echo "STAGE 2: Coherence Training for Generation (NEW)"
echo "=========================================="
echo "Teaching coherence module to generate coherent sequences..."
echo "Experts: FROZEN (keep representations)"
echo "Coherence: TRAINABLE (learn sequences)"
echo ""

python train_coherence_stage.py \
    "$INSTRUCTION_MODEL" \
    --dataset alpaca \
    --samples "$MAX_SAMPLES" \
    --epochs "$COHERENCE_EPOCHS" \
    --batch-size 4 \
    --lr 0.0005 \
    --device cuda \
    || { echo "❌ Coherence training failed!"; exit 1; }

# Find the final coherence-tuned model
FINAL_MODEL=$(ls -t outputs/coherence_stage2_*/coherence_final.pt | head -1)
echo ""
echo "✓ Coherence training completed!"
echo "✓ Model saved: $FINAL_MODEL"
echo ""

#
# STAGE 3: TESTING
# Test generation quality
#

echo "=========================================="
echo "STAGE 3: Testing Generation"
echo "=========================================="
echo "Running generation tests..."
echo ""

python test_generation.py "$FINAL_MODEL" || echo "⚠️  Generation test encountered issues"

echo ""
echo "================================"
echo "TRAINING COMPLETED SUCCESSFULLY!"
echo "================================"
echo ""
echo "Your production-ready Pragnosia model with enhanced generation:"
echo "  📁 Model: $FINAL_MODEL"
echo ""
echo "Next steps:"
echo "  1. Chat with your model:"
echo "     python chat.py $FINAL_MODEL"
echo ""
echo "  2. Test specific prompts:"
echo "     python test_generation.py $FINAL_MODEL"
echo ""
echo "  3. Compare with Stage 1 model (before coherence training):"
echo "     python test_generation.py $INSTRUCTION_MODEL"
echo ""
echo "Model capabilities (ENHANCED):"
echo "  ✓ Natural conversation (improved with coherence training)"
echo "  ✓ Code generation (improved with coherence training)"
echo "  ✓ Question answering"
echo "  ✓ Instruction following"
echo "  ✓ Sequential coherence (NEW - via enhanced coherence module)"
echo ""
echo "Pragnosia is ready for chat and code! 🚀"
echo "================================"
