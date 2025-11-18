#!/bin/bash
#SBATCH --job-name=final_eval_iemocap
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/final_eval_iemocap_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/final_eval_iemocap_%j.err

# ===========================
# Final version evaluation - IEMOCAP dataset
# ===========================
# Rate points: 10,20,30,40,60,80,100,120,140,160
# Per emotion: 50 samples
# Random seed: 1344871 (reproducible)
# ===========================

echo "================================================================"
echo "Final evaluation: IEMOCAP dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================================================"

# Project directory
PROJECT_DIR="/data/gpfs/projects/punim2341/haoguangzhou/voice/my-research/to_ready/MyAudioResearchModel"
cd "$PROJECT_DIR" || exit 1

# Load local_config
if [ -f "local_config.sh" ]; then
    source local_config.sh
else
    echo "Error: local_config.sh does not exist"
    exit 1
fi

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load GCC/11.3.0
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Set Python path
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Check GPU
nvidia-smi

echo ""
echo "================================================================"
echo "Configuration information"
echo "================================================================"
echo "dataset: IEMOCAP"
echo "Samples per emotion: 50"
echo "Random seed: 1344871"
echo "Rate points: 10,20,30,40,60,80,100,120,140,160"
echo "Output directory: evaluation_results/final"
echo "================================================================"
echo ""

# Run evaluation
python run_evaluation.py \
    --dataset iemocap \
    --samples 50 \
    --rates "10,20,30,40,60,80,100,120,140,160" \
    --output-dir evaluation_results/final \
    --rvq-checkpoint checkpoints/grouped_rvq_best.pt \
    --entropy-checkpoint checkpoints/entropy_model_best.pt

exit_code=$?

echo ""
echo "================================================================"
echo "Task complete"
echo "Exit Code: $exit_code"
echo "End Time: $(date)"
echo "================================================================"

exit $exit_code

