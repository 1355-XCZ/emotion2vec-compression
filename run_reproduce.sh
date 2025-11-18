#!/bin/bash
# One-click paper experiment reproduction
# 
# This script:
# 1. Prepare evaluation subset data (100 samples per emotion)
# 2. Run evaluation experiment (3 datasets, 47 rate points)
# 3. Generate all paper figures
#
# Estimated runtime: 2-4 hours (depends on GPU performance)

set -e  # Exit immediately on error

echo "================================================================"
echo "Paper Experiment One-click Reproduction Script"
echo "================================================================"
echo ""

# Check Python environment
echo "Checking Python environment..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "❌ PyTorch not installed"; exit 1; }
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Step 1: Prepare evaluation subset data
echo "================================================================"
echo "Step 1/3: Prepare evaluation subset data"
echo "================================================================"
echo "Randomly sample 100 samples per emotion (seed=42, reproducible)"
echo ""

if [ -d "data_subset" ]; then
    echo "⚠️  data_subset directory already exists"
    read -p "Regenerate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data_subset
        python3 prepare_evaluation_subset.py
    else
        echo "Skip data preparation, using existing subset"
    fi
else
    python3 prepare_evaluation_subset.py
fi

echo ""
echo "✅ Data preparation complete"
echo ""

# Step 2 & 3: Run evaluation and plotting
echo "================================================================"
echo "Step 2-3/3: Run evaluation experiment and generate figures"
echo "================================================================"
echo "Datasets: ESD, IEMOCAP, RAVDESS"
echo "Rate points: 47 (10-200 BPF step 5, 200-300 BPF step 20)"
echo "Sample count: 100/emotion"
echo ""
echo "⏰ Estimated runtime: 2-4 hours"
echo ""

python3 reproduce_experiments.py --mode all

echo ""
echo "================================================================"
echo "🎉 Experiment reproduction complete!"
echo "================================================================"
echo ""
echo "Result location:"
echo "  - Evaluation data: evaluation_results/"
echo "  - Paper figures: evaluation_results/figures/"
echo ""
echo "View results:"
echo "  ls -lh evaluation_results/*.json"
echo "  ls -lh evaluation_results/figures/*.png"
echo ""

