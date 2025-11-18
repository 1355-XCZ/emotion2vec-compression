#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --job-name=train_rvq
#SBATCH --output=/data/gpfs/projects/punim2341/haoguangzhou/logs/train_rvq_%j.out
#SBATCH --error=/data/gpfs/projects/punim2341/haoguangzhou/logs/train_rvq_%j.err

# Load local configuration (user needs to create based on local_config.sh.example)
if [ -f ./local_config.sh ]; then
    source ./local_config.sh
fi

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load GCC/11.3.0
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# Activate virtual environment (if configured)
if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Enter project directory
PROJECT_DIR="/data/gpfs/projects/punim2341/haoguangzhou/voice/my-research/to_ready/MyAudioResearchModel"
cd "$PROJECT_DIR"

# Set environment variables
export PROJECT_ROOT="$PROJECT_DIR"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Run RVQ training
echo "Starting RVQ training (using 100h Chinese-English data)..."
echo "Time: $(date)"

python train_rvq.py

echo "Training complete!"
echo "Time: $(date)"

