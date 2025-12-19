#!/bin/bash
# Pragnosia Quick Start Script
#
# This script helps you get started with training and using Pragnosia quickly.

set -e  # Exit on error

echo "=================================="
echo "PRAGNOSIA QUICK START"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "Error: pip is not installed"
    exit 1
fi

echo "1. Installing dependencies..."
pip install -q transformers datasets torch tqdm tensorboard matplotlib numpy || {
    echo "Error: Failed to install dependencies"
    exit 1
}
echo "   ✓ Dependencies installed"
echo ""

echo "2. Creating output directories..."
mkdir -p ./outputs
mkdir -p ./data
echo "   ✓ Directories created"
echo ""

echo "=================================="
echo "QUICK START OPTIONS"
echo "=================================="
echo ""
echo "Choose an option:"
echo "  1) Train small test model (350M, ~5 minutes on CPU)"
echo "  2) Train small production model (1B, GPU recommended)"
echo "  3) Train medium model (3B, GPU required)"
echo "  4) Chat with existing model"
echo "  5) Evaluate existing model"
echo "  6) Show usage guide"
echo ""
read -p "Enter option (1-6): " option

case $option in
    1)
        echo ""
        echo "Training small test model..."
        echo "This will train a 350M parameter model for quick testing."
        echo ""
        python train.py \
            --model-size 350M \
            --num-experts 8 \
            --dataset chat \
            --epochs 1 \
            --batch-size 4 \
            --max-samples 500 \
            --logging-steps 10

        echo ""
        echo "=================================="
        echo "Training complete!"
        echo "=================================="
        echo ""
        echo "Next steps:"
        echo "  • Chat with your model:"
        echo "    python chat.py --checkpoint ./outputs/pragnosia_350M_*/final_model.pt"
        echo ""
        echo "  • Evaluate your model:"
        echo "    python evaluate.py --checkpoint ./outputs/pragnosia_350M_*/final_model.pt"
        ;;

    2)
        echo ""
        echo "Training 1B production model..."
        echo "This will train a 1B parameter model (GPU recommended)."
        echo ""
        read -p "Number of epochs (default: 3): " epochs
        epochs=${epochs:-3}

        python train.py \
            --model-size 1B \
            --num-experts 16 \
            --dataset all \
            --epochs $epochs \
            --batch-size 4 \
            --offload-to-cpu

        echo ""
        echo "=================================="
        echo "Training complete!"
        echo "=================================="
        ;;

    3)
        echo ""
        echo "Training 3B medium model..."
        echo "This requires a GPU with at least 8GB VRAM."
        echo ""
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "Cancelled."
            exit 0
        fi

        python train.py \
            --model-size 3B \
            --num-experts 32 \
            --dataset all \
            --epochs 3 \
            --batch-size 2 \
            --offload-to-cpu

        echo ""
        echo "Training complete!"
        ;;

    4)
        echo ""
        echo "Starting chat interface..."
        echo ""

        # Find most recent checkpoint
        checkpoint=$(find ./outputs -name "final_model.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

        if [ -z "$checkpoint" ]; then
            echo "No trained models found in ./outputs/"
            echo "Please train a model first."
            exit 1
        fi

        echo "Using checkpoint: $checkpoint"
        echo ""

        python chat.py --checkpoint "$checkpoint"
        ;;

    5)
        echo ""
        echo "Evaluating model..."
        echo ""

        # Find most recent checkpoint
        checkpoint=$(find ./outputs -name "final_model.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

        if [ -z "$checkpoint" ]; then
            echo "No trained models found in ./outputs/"
            echo "Please train a model first."
            exit 1
        fi

        echo "Using checkpoint: $checkpoint"
        echo ""

        python evaluate.py --checkpoint "$checkpoint" --dataset all --show-examples 5
        ;;

    6)
        echo ""
        cat USAGE.md
        ;;

    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "For more information, see:"
echo "  • USAGE.md - Complete usage guide"
echo "  • train.py --help - Training options"
echo "  • chat.py --help - Chat options"
echo "=================================="
