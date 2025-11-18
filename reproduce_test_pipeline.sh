#!/bin/bash

# ================================================================
# Quick Test Pipeline - 4 rate points verification
# ================================================================
# Features:
#   1. Check/Create virtual environment (skip if already exists)
#   2. Install dependencies (skip if already installed)
#   3. Prepare evaluation data subset
#   4. Run quick test evaluation (4 rate points × 3 datasets)
#   5. Generate all paper figures
#
# Usage:
#   Local execution:    bash reproduce_test_pipeline.sh
#   SLURM cluster:      sbatch scripts/test_4points_3datasets.slurm
# ================================================================

set -e  # Exit immediately on error

PROJECT_ROOT=$(pwd)
VENV_DIR="$PROJECT_ROOT/reproduce_venv"

echo "================================================================"
echo "🧪 Quick Test Pipeline (4 rate points)"
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
# Step 5: Run test evaluation (4 rate points)
# ================================================================
echo "================================================================"
echo "Step 5: Run test evaluation"
echo "================================================================"
echo "⚠️  This is a quick test version"
echo ""
echo "Configuration:"
echo "  - Rate points: 4 (10, 100, 200, 300 BPF)"
echo "  - Datasets: 3 (ESD, IEMOCAP, RAVDESS)"
echo "  - Total evaluations: ~6,400"
echo "  - Estimated time: 30-45 minutes"
echo ""
echo "Note: Full experiment uses 47 rate points (run reproduce_full_pipeline.sh)"
echo ""

# Test rate points
TEST_RATES="10,100,200,300"

OUTPUT_DIR="evaluation_results_test"
mkdir -p "$OUTPUT_DIR"

# Evaluate each dataset
for DATASET in esd iemocap ravdess; do
    RESULT_FILE="$OUTPUT_DIR/rate_sweep_${DATASET^^}.json"
    
    if [ -f "$RESULT_FILE" ]; then
        echo ""
        echo "✅ ${DATASET^^} completed (skipped)"
        echo "   Result file: $RESULT_FILE"
    else
        echo ""
        echo "📊 Evaluating ${DATASET^^}..."
        python run_evaluation.py \
            --dataset "$DATASET" \
            --data-root data_subset \
            --samples 100 \
            --rates "$TEST_RATES" \
            --output-dir "$OUTPUT_DIR"
        echo "✅ ${DATASET^^} evaluation complete"
    fi
done

echo ""
echo "✅ All dataset test evaluations complete"
echo ""

# ================================================================
# Step 6: Generate paper figures
# ================================================================
echo "================================================================"
echo "Step 6: Generate paper figures"
echo "================================================================"
echo "Generating 5 key figures (based on test data):"
echo "  1. Overall Weighted F1 Score"
echo "  2. Overall Model Confidence"
echo "  3. Per-class Accuracy (Vertical Layout)"
echo "  4. Model Confidence by Emotion (Vertical Layout)"
echo "  5. Confusion Matrices 3×4 Grid"
echo ""

# Copy test results to standard location for plotting
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
echo "🎉 Test complete!"
echo "================================================================"
echo ""
echo "📁 Generated files:"
echo "   Data subset:     data_subset/"
echo "   Test results:    evaluation_results_test/"
echo "   Paper figures:   evaluation_results/figures/"
echo ""
echo "📊 Key figures:"
for fig in evaluation_results/figures/*.png; do
    if [ -f "$fig" ]; then
        echo "   - $(basename "$fig")"
    fi
done
echo ""
echo "⚠️  Reminder: These are 4-rate-point test results"
echo "   To run full experiment, use: bash reproduce_full_pipeline.sh"
echo ""
echo "Completion time: $(date)"
echo "================================================================"

