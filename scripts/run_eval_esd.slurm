#!/bin/bash
#SBATCH --job-name=eval_esd
#SBATCH --output=logs/eval_esd_%j.out
#SBATCH --error=logs/eval_esd_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Load local configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load modules
module purge
module load GCCcore/11.3.0 Python/3.10.4 GCC/11.3.0 OpenMPI/4.1.4
module load FFmpeg/4.4.2 espeak-ng/1.52 CUDA/11.8.0 cuDNN/8.7.0.84-CUDA-11.8.0

# Load configuration file
if [ -f "$PROJECT_DIR/local_config.sh" ]; then
    source "$PROJECT_DIR/local_config.sh"
fi

cd "$PROJECT_ROOT" || exit 1

# Activate virtual environment
if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
source "$VENV_PATH/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Run evaluation
cd "${PROJECT_ROOT}"
python run_evaluation.py --dataset esd "$@"

