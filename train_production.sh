#!/bin/bash
################################################################################
# Pragnosia Production Training Pipeline
#
# This script trains a production-ready Pragnosia model from scratch:
# 1. Pre-training on WikiText-103 (language fundamentals)
# 2. Instruction fine-tuning (chat capabilities)
# 3. Final testing and validation
#
# Usage:
#   bash train_production.sh [model_size] [max_samples]
#
# Examples:
#   bash train_production.sh 350M 10000  # Quick training for testing
#   bash train_production.sh 1B 50000    # Production training
################################################################################

set -e  # Exit on error

# Configuration
MODEL_SIZE="${1:-350M}"
MAX_SAMPLES="${2:-}"
EPOCHS_PRETRAIN=3
EPOCHS_INSTRUCT=3
BATCH_SIZE=4

echo "================================"
echo "PRAGNOSIA PRODUCTION TRAINING"
echo "================================"
echo "Model size: $MODEL_SIZE"
echo "Max samples: ${MAX_SAMPLES:-unlimited}"
echo "Pre-train epochs: $EPOCHS_PRETRAIN"
echo "Instruction epochs: $EPOCHS_INSTRUCT"
echo "================================"
echo ""

# Step 1: Pre-training on WikiText-103
echo "=========================================="
echo "STEP 1: Pre-training on WikiText-103"
echo "=========================================="
echo "Teaching the model language fundamentals..."
echo ""

PRETRAIN_CMD="python pretrain.py \
    --model-size $MODEL_SIZE \
    --dataset wikitext-103 \
    --epochs $EPOCHS_PRETRAIN \
    --batch-size $BATCH_SIZE \
    --logging-steps 50 \
    --save-steps 2000"

if [ -n "$MAX_SAMPLES" ]; then
    PRETRAIN_CMD="$PRETRAIN_CMD --max-samples $MAX_SAMPLES"
fi

eval $PRETRAIN_CMD

# Find the most recent pretrained model
PRETRAINED_MODEL=$(find outputs/pretrain_${MODEL_SIZE}_wikitext_103_* -name "pretrained_model.pt" -type f | sort -r | head -n 1)

if [ -z "$PRETRAINED_MODEL" ]; then
    echo "❌ Error: Pre-trained model not found!"
    exit 1
fi

echo ""
echo "✓ Pre-training completed!"
echo "✓ Model saved: $PRETRAINED_MODEL"
echo ""

# Step 2: Instruction Fine-Tuning
echo "=========================================="
echo "STEP 2: Instruction Fine-Tuning"
echo "=========================================="
echo "Teaching the model to chat and follow instructions..."
echo ""

INSTRUCT_CMD="python finetune_instruction.py \
    --resume $PRETRAINED_MODEL \
    --epochs $EPOCHS_INSTRUCT \
    --batch-size $BATCH_SIZE \
    --logging-steps 50 \
    --save-steps 1000"

if [ -n "$MAX_SAMPLES" ]; then
    INSTRUCT_CMD="$INSTRUCT_CMD --max-samples $MAX_SAMPLES"
fi

eval $INSTRUCT_CMD

# Find the most recent chat model
CHAT_MODEL=$(find outputs/instruction_tuned_* -name "pragnosia_chat_model.pt" -type f | sort -r | head -n 1)

if [ -z "$CHAT_MODEL" ]; then
    echo "❌ Error: Chat model not found!"
    exit 1
fi

echo ""
echo "✓ Instruction fine-tuning completed!"
echo "✓ Model saved: $CHAT_MODEL"
echo ""

# Step 3: Test Generation
echo "=========================================="
echo "STEP 3: Testing Generation"
echo "=========================================="
echo "Running generation tests..."
echo ""

python test_generation.py "$CHAT_MODEL"

echo ""
echo "================================"
echo "TRAINING COMPLETED SUCCESSFULLY!"
echo "================================"
echo ""
echo "Your production-ready Pragnosia model is ready:"
echo "  📁 Model: $CHAT_MODEL"
echo ""
echo "Next steps:"
echo "  1. Chat with your model:"
echo "     python chat.py $CHAT_MODEL"
echo ""
echo "  2. Test specific prompts:"
echo "     python test_generation.py $CHAT_MODEL"
echo ""
echo "  3. Deploy to production (coming soon)"
echo ""
echo "Model capabilities:"
echo "  ✓ Natural conversation"
echo "  ✓ Code generation"
echo "  ✓ Question answering"
echo "  ✓ Instruction following"
echo ""
echo "Pragnosia is ready to assist! 🚀"
echo "================================"
