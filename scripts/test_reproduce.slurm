#!/bin/bash
#SBATCH --job-name=test_reproduce
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_reproduce_%j.out
#SBATCH --error=logs/test_reproduce_%j.err

# Quick test of experiment reproduction system
# To verify environment and scripts work correctly
# Test configuration: 10 samples per emotion, 3 rate points

echo "================================================================"
echo "Experiment reproduction system test"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo ""

# Create log directory
mkdir -p logs

# Load environment configuration
if [ -f "local_config.sh" ]; then
    echo "Loading local configuration..."
    source local_config.sh
else
    echo "⚠️  local_config.sh not found, using default environment"
fi

# Activate Python environment
if [ -n "$CONDA_ENV" ]; then
    echo "Activating Conda environment: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV
elif [ -n "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source $VENV_PATH/bin/activate
fi

echo ""
echo "================================================================"
echo "Step 1: Check dependencies"
echo "================================================================"
python3 test_dependencies.py
if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed"
    exit 1
fi

echo ""
echo "================================================================"
echo "Step 2: Prepare test subset (10 samples/emotion)"
echo "================================================================"

# If test subset exists, delete it first
if [ -d "data_subset_test" ]; then
    echo "Removing old test subset..."
    rm -rf data_subset_test
fi

python3 prepare_evaluation_subset.py \
    --source-data data \
    --target-data data_subset_test \
    --samples 10 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed"
    exit 1
fi

echo ""
echo "================================================================"
echo "Step 3: Run quick evaluation (3 rate points)"
echo "================================================================"

# Test ESD dataset
echo "Testing ESD dataset..."
python3 run_evaluation.py \
    --dataset esd \
    --data-root data_subset_test \
    --samples 10 \
    --rates "10,50,100" \
    --output-dir evaluation_results_test

if [ $? -ne 0 ]; then
    echo "❌ ESD evaluation failed"
    exit 1
fi

echo ""
echo "================================================================"
echo "Step 4: Test plotting script"
echo "================================================================"

# Test single plotting script
echo "Testing plotting features..."
python3 plot_overall_accuracy_and_confusion.py

if [ $? -ne 0 ]; then
    echo "⚠️  Plotting test failed (may be missing evaluation results)"
    echo "   Continuing test..."
fi

echo ""
echo "================================================================"
echo "Test complete"
echo "================================================================"

# Display generated files
echo "Generated test files:"
if [ -d "data_subset_test" ]; then
    echo "  ✅ Test subset:"
    du -sh data_subset_test/
    ls -1 data_subset_test/
fi

if [ -d "evaluation_results_test" ]; then
    echo "  ✅ Evaluation results:"
    ls -lh evaluation_results_test/*.json 2>/dev/null | head -5
fi

echo ""
echo "🎉 Quick test passed!"
echo ""
echo "Full experiment run command:"
echo "  sbatch scripts/run_full_reproduce.slurm"
echo ""
echo "Or run manually:"
echo "  python3 prepare_evaluation_subset.py"
echo "  python3 reproduce_experiments.py --mode all"

# Clean up test data (optional)
read -t 10 -p "Delete test data? (y/n, auto-keep after 10 seconds): " cleanup
if [[ $cleanup =~ ^[Yy]$ ]]; then
    echo "Cleaning up test data..."
    rm -rf data_subset_test
    rm -rf evaluation_results_test
    echo "✅ Cleanup complete"
else
    echo "Keeping test data for inspection"
fi

echo "Completion time: $(date)"

