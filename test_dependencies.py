#!/usr/bin/env python3
"""
Test if all dependencies are correctly installed

Used to verify if environment configuration is complete
"""

import sys
import importlib

# Define all required dependency packages
REQUIRED_PACKAGES = {
    # Core deep learning
    'torch': 'PyTorch (deep learning framework)',
    'numpy': 'NumPy (numerical computation)',
    
    # Audio and model
    'funasr': 'FunASR (Emotion2Vec)',
    'modelscope': 'ModelScope (model download)',
    'torchaudio': 'TorchAudio (audio processing)',
    
    # Machine learning
    'sklearn': 'scikit-learn (evaluation metrics)',
    'scipy': 'SciPy (statistical analysis)',
    
    # RVQ
    'vector_quantize_pytorch': 'vector-quantize-pytorch (VQ implementation)',
    
    # Data processing
    'tqdm': 'tqdm (progress bar)',
    'yaml': 'PyYAML (configuration file)',
    
    # Visualization
    'matplotlib': 'Matplotlib (plotting)',
    'seaborn': 'Seaborn (statistical plots)',
    'pandas': 'Pandas (data processing)',
    
    # Transformer related
    'einops': 'einops (tensor operations)',
    'transformers': 'Transformers (pretrained models)',
}


def check_package(package_name, description):
    """Check if a single package can be imported"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {description:40s} v{version}")
        return True
    except ImportError as e:
        print(f"❌ {description:40s} NOT FOUND")
        print(f"   Error: {e}")
        return False


def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✅ CUDA available: {device_name} (CUDA {cuda_version})")
            return True
        else:
            print(f"⚠️  CUDA not available (using CPU)")
            return False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False


def check_project_structure():
    """Check project structure"""
    from pathlib import Path
    
    print("\n" + "="*80)
    print("Checking project structure")
    print("="*80)
    
    required_files = [
        'config.py',
        'grouped_rvq.py',
        'entropy_model.py',
        'rate_controller.py',
        'run_evaluation.py',
        'reproduce_experiments.py',
        'prepare_evaluation_subset.py',
        'checkpoints/grouped_rvq_best.pt',
        'checkpoints/entropy_model_best.pt',
    ]
    
    required_dirs = [
        'datasets',
        'evaluation',
        'scripts',
    ]
    
    all_ok = True
    
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file:45s} ({size:,} bytes)")
        else:
            print(f"❌ {file:45s} NOT FOUND")
            all_ok = False
    
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            files = len(list(Path(dir_name).glob('*.py')))
            print(f"✅ {dir_name + '/':<45s} ({files} Python files)")
        else:
            print(f"❌ {dir_name + '/':<45s} NOT FOUND")
            all_ok = False
    
    return all_ok


def main():
    print("="*80)
    print("Dependency Check Tool")
    print("="*80)
    print()
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python version too low, need >= 3.8")
        sys.exit(1)
    else:
        print("✅ Python version meets requirements")
    
    print()
    print("="*80)
    print("Checking dependency packages")
    print("="*80)
    
    # Check all packages
    failed_packages = []
    for package, description in REQUIRED_PACKAGES.items():
        if not check_package(package, description):
            failed_packages.append(package)
    
    print()
    print("="*80)
    print("Checking GPU support")
    print("="*80)
    check_cuda()
    
    print()
    structure_ok = check_project_structure()
    
    # Summary
    print()
    print("="*80)
    print("Check Summary")
    print("="*80)
    
    if failed_packages:
        print(f"❌ {len(failed_packages)} packages missing:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print()
        print("Please run the following command to install missing packages:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All dependency packages installed")
    
    if not structure_ok:
        print("⚠️  Project structure incomplete, please check missing files")
        sys.exit(1)
    else:
        print("✅ Project structure complete")
    
    print()
    print("🎉 Environment configuration complete, ready to run experiments!")
    print()
    print("Next steps:")
    print("  1. Prepare data: python3 prepare_evaluation_subset.py")
    print("  2. Run experiment: python3 reproduce_experiments.py --mode all")
    print("  or one-click run: bash run_reproduce.sh")


if __name__ == '__main__':
    main()

