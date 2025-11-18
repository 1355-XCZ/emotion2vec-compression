#!/bin/bash
#SBATCH --job-name=test_env_install
#SBATCH --account=punim2341
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_env_%j.out
#SBATCH --error=logs/test_env_%j.err

# Strict environment test script
# From scratch: create virtual environment, install dependencies, verify environment

set -e  # Exit immediately on error

echo "================================================================"
echo "Environment integrity test (from scratch)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

# Create log directory
mkdir -p logs

# Define virtual environment path
VENV_PATH="./test_venv_${SLURM_JOB_ID}"

# Clean up possible old test environment
if [ -d "$VENV_PATH" ]; then
    echo "Removing old test environment: $VENV_PATH"
    rm -rf "$VENV_PATH"
fi

# Display system information
echo "System information:"
echo "  Python: $(which python3) ($(python3 --version))"
echo "  Pip: $(which pip3)"
echo "  Working directory: $(pwd)"
echo ""

# Check GPU
echo "GPU information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  nvidia-smi not available"
fi

echo ""
echo "================================================================"
echo "Step 1: Create virtual environment"
echo "================================================================"

python3 -m venv "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "❌ Virtual environment creation failed"
    exit 1
fi

echo "✅ Virtual environment created: $VENV_PATH"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "✅ Virtual environment activated"
echo "  Python: $(which python)"
echo "  Pip: $(which pip)"
echo ""

echo "================================================================"
echo "Step 2: Upgrade pip"
echo "================================================================"

pip install --upgrade pip setuptools wheel
echo "✅ Pip upgraded"
echo ""

echo "================================================================"
echo "Step 3: Install dependencies (from requirements.txt)"
echo "================================================================"

pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Dependency installation failed"
    deactivate
    rm -rf "$VENV_PATH"
    exit 1
fi

echo "✅ All dependencies installed"
echo ""

echo "================================================================"
echo "Step 4: Run dependency check script"
echo "================================================================"

python test_dependencies.py

TEST_RESULT=$?

echo ""
echo "================================================================"
echo "Step 5: Clean up test environment"
echo "================================================================"

# Deactivate virtual environment
deactivate

# Delete test virtual environment
echo "Removing test environment: $VENV_PATH"
rm -rf "$VENV_PATH"

echo "✅ Test environment cleaned up"
echo ""

echo "================================================================"
echo "Test result summary"
echo "================================================================"

if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Environment test fully passed!"
    echo ""
    echo "All dependency packages can be correctly installed and imported."
    echo ""
    echo "Next steps:"
    echo "  1. Install dependencies in your working environment:"
    echo "     pip install -r requirements.txt"
    echo ""
    echo "  2. Run quick test:"
    echo "     sbatch scripts/test_reproduce.slurm"
    echo ""
    echo "  3. Run full experiment:"
    echo "     sbatch scripts/run_full_reproduce.slurm"
    exit 0
else
    echo "❌ Dependency check failed"
    echo ""
    echo "Although dependency packages installed successfully, some packages cannot be imported correctly."
    echo "Please check logs: logs/test_env_${SLURM_JOB_ID}.out"
    exit 1
fi
