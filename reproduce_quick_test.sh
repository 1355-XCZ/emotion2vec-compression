#!/bin/bash

# ================================================================
# Ultra-fast Test Pipeline - 10 samples/emotion
# ================================================================
# Features:
#   1. Check/Create virtual environment (skip if already exists)
#   2. Install dependencies (skip if already installed)
#   3. Prepare quick test data subset (10 samples/emotion)
#   4. Run quick evaluation (4 rate points × 3 datasets)
#   5. Generate all paper figures
#
# Usage:
#   Local execution:    bash reproduce_quick_test.sh
#   SLURM cluster:      sbatch scripts/quick_test_10samples.slurm
#
# Estimated time: 5-10 minutes (first time needs +15 min for installation)
# ================================================================

set -e  # Exit immediately on error

PROJECT_ROOT=$(pwd)
VENV_DIR="$PROJECT_ROOT/reproduce_venv"

echo "================================================================"
echo "⚡ Ultra-fast Test Pipeline (10 samples/emotion)"
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
# Step 4: Prepare quick test data subset
# ================================================================
echo "================================================================"
echo "Step 4: Prepare quick test data subset"
echo "================================================================"
echo "⚠️  This is ultra-fast test version"
echo ""
echo "Configuration:"
echo "  - Datasets: ESD, IEMOCAP, RAVDESS"
echo "  - Sample count: 10/emotion (quick test)"
echo "  - Random seed: 42 (ensure reproducibility)"
echo ""

echo "📊 Preparing quick test data subset..."
python prepare_evaluation_subset.py \
    --datasets esd iemocap ravdess \
    --samples 10 \
    --seed 42 \
    --target-data data_subset_quick

echo "✅ Quick test data subset prepared"
echo ""

# ================================================================
# Step 5: Run quick test evaluation (4 rate points, 10 samples)
# ================================================================
echo "================================================================"
echo "Step 5: Run quick test evaluation"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  - Rate points: 4 (10, 100, 200, 300 BPF)"
echo "  - Datasets: 3 (ESD, IEMOCAP, RAVDESS)"
echo "  - Sample count: 10/emotion"
echo "  - Total evaluations: ~160"
echo "  - Estimated time: 5-10 minutes ⚡"
echo ""

# Test rate points
TEST_RATES="10,100,200,300"

OUTPUT_DIR="evaluation_results_quick"
mkdir -p "$OUTPUT_DIR"

# Evaluate each dataset
for DATASET in esd iemocap ravdess; do
    echo ""
    echo "📊 Evaluating ${DATASET^^}..."
    python run_evaluation.py \
        --dataset "$DATASET" \
        --data-root data_subset_quick \
        --samples 10 \
        --rates "$TEST_RATES" \
        --output-dir "$OUTPUT_DIR"
    echo "✅ ${DATASET^^} evaluation complete"
done

echo ""
echo "✅ All dataset quick test evaluations complete"
echo ""

# ================================================================
# Step 6: Generate paper figures
# ================================================================
echo "================================================================"
echo "Step 6: Generate paper figures"
echo "================================================================"
echo "Generating 5 key figures (based on quick test data):"
echo "  1. Overall Weighted F1 Score"
echo "  2. Overall Model Confidence"
echo "  3. Per-class Accuracy (Vertical Layout)"
echo "  4. Model Confidence by Emotion (Vertical Layout)"
echo "  5. Confusion Matrices 3×4 Grid"
echo ""

# Copy quick test results to standard location for plotting
mkdir -p evaluation_results
cp "$OUTPUT_DIR/rate_sweep_ESD.json" evaluation_results/ESD_evaluation_results.json
cp "$OUTPUT_DIR/rate_sweep_IEMOCAP.json" evaluation_results/IEMOCAP_evaluation_results.json
cp "$OUTPUT_DIR/rate_sweep_RAVDESS.json" evaluation_results/RAVDESS_evaluation_results.json

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
echo "🎉 Quick test complete!"
echo "================================================================"
echo ""
echo "📁 Generated files:"
echo "   Quick test data:    data_subset_quick/"
echo "   Quick test results: evaluation_results_quick/"
echo "   Paper figures:      evaluation_results/figures/"
echo ""
echo "📊 Key figures:"
for fig in evaluation_results/figures/*.png; do
    if [ -f "$fig" ]; then
        echo "   - $(basename "$fig")"
    fi
done
echo ""
echo "⚠️  Reminder: These are 10-sample quick test results"
echo ""
echo "Next steps:"
echo "  1. Check if workflow and figures are normal"
echo "  2. Run 100-sample test: bash reproduce_test_pipeline.sh"
echo "  3. Run full experiment: bash reproduce_full_pipeline.sh"
echo ""
echo "Completion time: $(date)"
echo "================================================================"

