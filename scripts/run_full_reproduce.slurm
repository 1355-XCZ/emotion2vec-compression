#!/bin/bash
#SBATCH --job-name=full_reproduce
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/full_reproduce_%j.out
#SBATCH --error=logs/full_reproduce_%j.err

# ================================================================
# Complete Experiment Reproduction Pipeline - SLURM version
# Uses intelligent script reproduce_full_pipeline.sh
# 47 rate points × 3 datasets
# ================================================================

set -e

echo "================================================================"
echo "🚀 SLURM Complete Experiment Reproduction Pipeline (47 rate points)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

mkdir -p logs

echo "System information:"
echo "  Python: $(which python3) ($(python3 --version))"
echo "  Working directory: $(pwd)"
echo ""

echo "GPU information:"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

echo "================================================================"
echo "Run complete experiment pipeline"
echo "================================================================"
echo ""

# Call intelligent complete script
bash reproduce_full_pipeline.sh

RESULT=$?

echo ""
echo "================================================================"
echo "SLURM job complete"
echo "================================================================"

if [ $RESULT -eq 0 ]; then
    echo "✅ Complete experiment successful!"
    echo ""
    echo "Generated files:"
    echo "  Evaluation results: evaluation_results/"
    ls -lh evaluation_results/*_evaluation_results.json 2>/dev/null || true
    echo ""
    echo "  Paper figures: evaluation_results/figures/"
    ls -lh evaluation_results/figures/*.png 2>/dev/null || true
    echo ""
    echo "🎉 All experiment results generated, ready for paper!"
else
    echo "❌ Experiment failed, please check logs"
    echo "  Output: logs/full_reproduce_${SLURM_JOB_ID}.out"
    echo "  Error: logs/full_reproduce_${SLURM_JOB_ID}.err"
fi

echo ""
echo "Completion time: $(date)"

exit $RESULT
