#!/bin/bash
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=extract_features
#SBATCH --output=./logs/extract_features_%j.out
#SBATCH --error=./logs/extract_features_%j.err

# Load local configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_DIR/local_config.sh" ]; then
    source "$PROJECT_DIR/local_config.sh"
fi

# Enter project directory
cd "$PROJECT_DIR" || exit 1

# Load modules
module purge
module load GCCcore/11.3.0 Python/3.10.4 GCC/11.3.0 OpenMPI/4.1.4
module load FFmpeg/4.4.2 espeak-ng/1.52 CUDA/11.8.0 cuDNN/8.7.0.84-CUDA-11.8.0

# Activate virtual environment (if configured)
if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Set environment variables (project is now self-contained)
export PYTHONUNBUFFERED=1
export PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=========================================="
echo "Extract evaluation dataset emotion2vec features"
echo "=========================================="
echo "Time: $(date)"

python extract_eval_features.py

echo ""
echo "=========================================="
echo "Completion time: $(date)"
echo "=========================================="


