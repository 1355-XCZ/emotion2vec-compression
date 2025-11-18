#!/bin/bash

# ================================================================
# Complete Experiment Reproduction Pipeline - One-click execution of all steps
# ================================================================
# Features:
#   1. Check/Create virtual environment (skip if already exists)
#   2. Install dependencies (skip if already installed)
#   3. Prepare evaluation data subset
#   4. Run complete evaluation (47 rate points × 3 datasets)
#   5. Generate all paper figures
#
# Usage:
#   Local execution:    bash reproduce_full_pipeline.sh
#   SLURM cluster:      sbatch scripts/run_full_reproduce.slurm
# ================================================================

set -e  # Exit immediately on error

PROJECT_ROOT=$(pwd)
VENV_DIR="$PROJECT_ROOT/reproduce_venv"

echo "================================================================"
echo "🚀 Emotion RVQ Experiment Complete Reproduction Pipeline"
echo "================================================================"
echo "Project directory: $PROJECT_ROOT"
echo "Start time: $(date)"
echo ""

# ================================================================
# Step 1: Check/Create virtual environment
# ================================================================
echo "================================================================"
echo "Step 1: Check Python virtual environment"
echo "================================================================"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    echo "✅ Found existing virtual environment: $VENV_DIR"
    echo "   Skip environment creation step"
else
    echo "📦 Creating new virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created: $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"
echo "   Python: $(which python)"
echo "   Version: $(python --version)"
echo ""

# ================================================================
# Step 2: Check/Install dependencies
# ================================================================
echo "================================================================"
echo "Step 2: Check project dependencies"
echo "================================================================"

# Check if key packages are installed
NEED_INSTALL=false

if ! python -c "import torch" 2>/dev/null; then
    echo "❌ torch not installed"
    NEED_INSTALL=true
elif ! python -c "import funasr" 2>/dev/null; then
    echo "❌ funasr not installed"
    NEED_INSTALL=true
elif ! python -c "import vector_quantize_pytorch" 2>/dev/null; then
    echo "❌ vector_quantize_pytorch not installed"
    NEED_INSTALL=true
else
    echo "✅ Core dependencies installed"
    echo "   Skip installation step"
fi

if [ "$NEED_INSTALL" = true ]; then
    echo ""
    echo "📦 Installing project dependencies..."
    echo "   This may take 10-15 minutes, please wait..."
    pip install --upgrade pip setuptools wheel -q
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
fi

echo ""

# ================================================================
# Step 3: Verify environment integrity
# ================================================================
echo "================================================================"
echo "Step 3: Verify environment integrity"
echo "================================================================"

python test_dependencies.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Environment verification failed! Please check error messages"
    exit 1
fi

echo ""
echo "✅ Environment verification passed"
echo ""

# ================================================================
# Step 4: Prepare evaluation data subset
# ================================================================
echo "================================================================"
echo "Step 4: Prepare evaluation data subset"
echo "================================================================"
echo "Configuration:"
echo "  - Datasets: ESD, IEMOCAP, RAVDESS"
echo "  - Sample count: 100/emotion"
echo "  - Random seed: 42 (ensure reproducibility)"
echo ""

if [ -d "data_subset" ] && [ -f "data_subset/ESD/subset_info.json" ]; then
    echo "✅ Found existing data subset: data_subset/"
    echo "   Skip data preparation step"
    echo "   (To regenerate, delete the data_subset/ directory)"
else
    echo "📊 Preparing data subset..."
    python prepare_evaluation_subset.py \
        --datasets esd iemocap ravdess \
        --samples 100 \
        --seed 42
    echo "✅ Data subset prepared"
fi

echo ""

# ================================================================
# Step 5: Run complete evaluation
# ================================================================
echo "================================================================"
echo "Step 5: Run complete evaluation"
echo "================================================================"
echo "Configuration:"
echo "  - Rate points: 47"
echo "    • 10-200 BPF (step 5): 39 points"
echo "    • 220-300 BPF (step 20): 5 points"
echo "  - Datasets: 3 (ESD, IEMOCAP, RAVDESS)"
echo "  - Total evaluations: ~75,200"
echo "  - Estimated time: 2-4 hours"
echo ""

# Generate rate points list
RATE_POINTS=$(python3 -c "print(','.join(map(str, list(range(10, 201, 5)) + list(range(220, 301, 20)))))")

OUTPUT_DIR="evaluation_results"
mkdir -p "$OUTPUT_DIR"

# Evaluate each dataset
for DATASET in esd iemocap ravdess; do
    RESULT_FILE="$OUTPUT_DIR/${DATASET^^}_evaluation_results.json"
    
    if [ -f "$RESULT_FILE" ]; then
        echo "⚠️  Found existing evaluation results: $RESULT_FILE"
        echo "   Skip ${DATASET^^} evaluation"
        echo "   (To re-evaluate, delete this file)"
    else
        echo ""
        echo "📊 Evaluating ${DATASET^^}..."
        python run_evaluation.py \
            --dataset "$DATASET" \
            --data-root data_subset \
            --samples 100 \
            --rates "$RATE_POINTS" \
            --output-dir "$OUTPUT_DIR"
        echo "✅ ${DATASET^^} evaluation complete"
    fi
done

echo ""
echo "✅ All dataset evaluations complete"
echo ""

# ================================================================
# Step 6: Generate paper figures
# ================================================================
echo "================================================================"
echo "Step 6: Generate paper figures"
echo "================================================================"
echo "Generating 5 key figures:"
echo "  1. Overall Weighted F1 Score"
echo "  2. Overall Model Confidence"
echo "  3. Per-class Accuracy (Vertical Layout)"
echo "  4. Model Confidence by Emotion (Vertical Layout)"
echo "  5. Confusion Matrices 3×4 Grid"
echo ""

python generate_paper_figures.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Figure generation complete"
    echo ""
    echo "Generated figures:"
    ls -lh evaluation_results/figures/*.png
else
    echo ""
    echo "❌ Figure generation failed"
    exit 1
fi

# ================================================================
# Completion summary
# ================================================================
echo ""
echo "================================================================"
echo "🎉 Experiment reproduction complete!"
echo "================================================================"
echo ""
echo "📁 Generated files:"
echo "   Data subset:         data_subset/"
echo "   Evaluation results:  evaluation_results/*_evaluation_results.json"
echo "   Paper figures:       evaluation_results/figures/"
echo ""
echo "📊 Key figures:"
for fig in evaluation_results/figures/*.png; do
    if [ -f "$fig" ]; then
        echo "   - $(basename "$fig")"
    fi
done
echo ""
echo "Completion time: $(date)"
echo "================================================================"

