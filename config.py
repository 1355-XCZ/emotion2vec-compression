"""
Configuration Management - Emotion RVQ Information Bottleneck Experiment
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class GroupedRVQConfig:
    """Grouped RVQ Model Configuration"""
    feature_dim: int = 768              # emotion2vec feature dimension
    num_groups: int = 12                # number of groups
    group_dim: int = 64                 # dimension per group (768/12=64)
    
    # Fine layers within groups
    num_fine_layers: int = 3            # Number of fine quantization layers per group (compatible with old checkpoint)
    fine_codebook_size: int = 128       # Fine layer codebook size
    
    # SKIP mechanism
    enable_skip: bool = True            # Enable SKIP mechanism
    use_full_ranking_ecvq: bool = True  # Enable full-ranking ECVQ reordering (solves rate control issues)
    ecvq_topk: Optional[int] = None     # Top-N trade-off (8/16 gives 95%+ benefit, None=full ranking)
    
    # RVQ training parameters (EMA mode)
    decay: float = 0.99                 # EMA decay rate
    commitment_weight: float = 0.25     # Commitment loss weight
    kmeans_init: bool = True            # K-means initialization
    kmeans_iters: int = 10              # K-means iterations
    threshold_ema_dead_code: float = 2.0

    # Training control (Note: num_epochs is defined in TrainingConfig)
    eval_interval: int = 1              # Validate every epoch
    save_interval: int = 5              # Save every 5 epochs
    
    # Early stopping mechanism
    enable_early_stopping: bool = True   # Enable early stopping
    early_stopping_patience: int = 10    # Tolerance epochs (stop if no improvement)
    early_stopping_min_delta: float = 0.0001  # Minimum improvement threshold (0.01% reconstruction rate)
    
    @property
    def total_codebook_size_with_skip(self) -> int:
        """Codebook size including SKIP"""
        if self.enable_skip:
            return self.fine_codebook_size + 1  # +1 for SKIP
        return self.fine_codebook_size
    
    @property
    def max_bits_per_frame(self) -> float:
        """Theoretical maximum bits/frame"""
        import numpy as np
        return self.num_groups * self.num_fine_layers * np.log2(self.fine_codebook_size)


@dataclass
class EntropyModelConfig:
    """Unconditional entropy model configuration - only q(z)"""
    # Model structure
    d_model: int = 256                  # Transformer hidden dimension
    num_layers: int = 6                 # Transformer layers
    num_heads: int = 8                  # Attention heads
    d_ff: int = 1024                    # FFN hidden dimension
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 16                # Batch size (reduced for 16-layer RVQ)
    num_epochs: int = 5                 # Training epochs
    train_data_fraction: float = 1.0    # Training data fraction (1.0=100%)
    
    # Early stopping mechanism
    enable_early_stopping: bool = True   # Enable early stopping
    early_stopping_patience: int = 2     # Tolerance epochs (stop if no improvement)
    early_stopping_min_delta: float = 0.01  # Minimum improvement threshold (1% loss decrease)
    
    # Performance optimization
    enable_optimizations: bool = True   # Enable performance optimizations
    max_frames_per_sample: int = 128     # Crop long sequences (reduce memory)
    use_amp: bool = True                # Mixed precision
    use_tf32: bool = True               # TF32 acceleration
    
    def get_effective_config(self):
        """Return actual optimization configuration"""
        if not self.enable_optimizations:
            return {
                'max_frames_per_sample': 0,
                'frame_stride': 1,
                'use_amp': False,
                'use_tf32': False,
            }
        return {
            'max_frames_per_sample': self.max_frames_per_sample,
            'frame_stride': 1,  # Frame downsampling stride (1=no downsampling)
            'use_amp': self.use_amp,
            'use_tf32': self.use_tf32,
        }
    
    # Context (intra-frame AR)
    max_seq_len: int = 100              # Intra-frame maximum sequence length
    causal: bool = True                 # Causal mask
    
    # SKIP position weighting
    skip_weight: float = 0.5            # SKIP weight (balance normal encoding and SKIP)
    
    # Target bpf grid (for index extraction, quantile gating) - friend scheme
    target_bpf_grid: List[float] = field(default_factory=lambda: [
        # Very low bitrate (0-20 bpf): dense ⭐
        2, 5, 10, 15, 20, 25,
        # Low bitrate (20-100 bpf): dense ⭐ user care
        30, 35, 40, 50, 60, 80, 100,
        # Mid-high bitrate (100-300 bpf): medium density
        150, 200, 300,
        # High bitrate (300-1344 bpf): sparse
        500, 1344,
        # No quantization baseline (original features)
        float('inf')  # No quantization, use original features directly
    ])  # 19 points (18 quantization points + 1 no-quantization baseline)

    # Training optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    max_grad_norm: float = 1.0


@dataclass
class RateControlConfig:
    """Rate control configuration"""
    # Target bitrates list (bps)
    target_rates_bps: List[int] = field(default_factory=lambda: [
        500,   # 10 bpf
        1000,  # 20 bpf
        2000,  # 40 bpf
        4000,  # 80 bpf
        8000   # 160 bpf
    ])

    lambda_min: float = 1e-4
    lambda_max: float = 32.0  # Extended to 16.0 to cover lower bitrates
    lambda_init: float = 16.0  # Start from middle value (binary search initial hint)

    # Binary search parameters
    rate_tolerance_bpf: float = 1.0  # Tolerance: 1 bpf (acceptable if not converged)
    max_binary_search_iters: int = 50
    
    # Dual update parameters
    dual_lr: float = 0.01
    dual_momentum: float = 0.9
    dual_update_interval: int = 100
    
    # Frame rate
    frame_rate_hz: float = 50.0
    
    @property
    def frame_duration_sec(self) -> float:
        """Frame duration (seconds)"""
        return 1.0 / self.frame_rate_hz
    
    def bps_to_bpf(self, bps: float) -> float:
        """bps → bits/frame"""
        return bps * self.frame_duration_sec
    
    def bpf_to_bps(self, bpf: float) -> float:
        """bits/frame → bps"""
        return bpf / self.frame_duration_sec


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Evaluation dataset list (using object-oriented design)
    dataset_names: List[str] = field(default_factory=lambda: ['IEMOCAP', 'RAVDESS', 'ESD'])
    
    def __post_init__(self):
        """If rate_sweep_rates_bpf is None, use target_bpf_grid from training"""
        if self.rate_sweep_rates_bpf is None:
            # Get target_bpf_grid from EntropyModelConfig
            # Need to be set at runtime
            pass
    
    # Method 1: Rate sweep (main method)
    enable_rate_sweep: bool = True
    # Prior knowledge: Below 50 BPF is critical region, focus on dense testing
    rate_sweep_rates_bpf: List[float] = field(default_factory=lambda: [
        5, 10, 15, 20, 25, 30, 40, 50,  # Focus: 5-50 BPF dense sampling
        100, 200,  # High bitrate reference points
        float('inf')  # No quantization baseline
    ])
    # 35 test points, 26 values in 0-100 bpf range (dense) ⭐
    
    # Method 2: Layer sweep (auxiliary method)
    enable_layer_sweep: bool = True
    # Prior knowledge: Little change after 5 layers/group, focus on testing 1-5 layers/group
    layer_sweep_layers: List[int] = field(default_factory=lambda: [
        12*1,  # 12 layers (12 groups × 1 layer/group)
        12*2,  # 24 layers (12 groups × 2 layers/group)
        12*3,  # 36 layers (12 groups × 3 layers/group)
        12*4,  # 48 layers (12 groups × 4 layers/group)
        12*5,  # 60 layers (12 groups × 5 layers/group)
    ])  # 5 points, focus on low layer count region
    
    # emotion2vec classifier configuration
    emotion2vec_model: str = "iic/emotion2vec_plus_base"
    emotion2vec_hub: str = "ms"  # modelscope
    
    # Results output
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_plots: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    # Data root directory (read from environment variable or configuration file)
    data_root: Optional[str] = None
    
    # Training data (50h EN+ZH)
    train_data_dir: str = "emilia_vevo_training_50h"
    
    # Feature file naming pattern
    feature_suffix: str = "_ev2_frame.npy"
    label_suffix: str = "_emotion.txt"  # Label file suffix (training data has no labels, used during evaluation)
    
    # Data split
    train_split: float = 0.9
    seed: int = 1344871  # Unified random seed
    
    # Data processing
    max_samples: Optional[int] = None   # Maximum number of samples (for quick testing)
    max_frames: int = 256               # Maximum frames (used in index extraction phase, prevent OOM)
    min_frames: int = 10                # Minimum frames
    normalize_features: bool = False    # ⚠️ Disable normalization (avoid train-test inconsistency)
    mean_std_path: Optional[str] = "ev2_mean_std_100h_EN_ZH.npz"  # Normalization parameters path (disabled)
    supported_languages: Optional[List[str]] = field(default_factory=lambda: ['EN', 'ZH'])  # Using 100h EN+ZH data
    
    def __post_init__(self):
        """Read data_root from environment variable (if not set)"""
        if self.data_root is None:
            self.data_root = os.environ.get("DATA_ROOT", "/data/gpfs/projects/punim2341/haoguangzhou/data")
    
    # Emotion label mapping (training data has no labels, used for evaluation placeholder)
    @property
    def emotion_label_map(self) -> dict:
        """Emotion label mapping (placeholder, training data has no labels)"""
        return {}  # Training data has no labels, return empty dict
    
    @property
    def train_data_path(self) -> str:
        return os.path.join(self.data_root, self.train_data_dir)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # General training parameters
    batch_size: int = 32  # Used for index extraction phase
    num_epochs: int = 20  # RVQ training epochs
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Data sampling
    use_bucketing: bool = True
    num_buckets: int = 10
    
    # Optimizer
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate scheduling
    scheduler: str = "cosine"
    warmup_steps: int = 4000
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Output directory (using project internal directory)
    output_dir: str = "checkpoints"
    exp_name: Optional[str] = None
    
    # Checkpoint control
    resume_from_checkpoint: bool = False  # Whether to resume from checkpoint
    load_rvq_checkpoint: bool = True      # Whether to load RVQ when training entropy model
    
    # Monitoring
    log_interval: int = 50
    eval_interval: int = 1
    save_interval: int = 5
    
    # TensorBoard
    use_tensorboard: bool = False  # Minimal version doesn't enable tensorboard
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.exp_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_name = f"exp_{timestamp}"
    
    @property
    def checkpoint_dir(self) -> str:
        return self.output_dir
    
    @property
    def log_dir(self) -> str:
        return os.path.join(self.output_dir, "logs")
    
    @property
    def results_dir(self) -> str:
        return os.path.join(self.output_dir, "results")


@dataclass
class SlurmConfig:
    """Slurm job configuration"""
    account: str = "punim2341"
    partition: str = "gpu-h100"
    
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 4
    gpus: int = 1
    memory: str = "32G"
    time_limit: str = "08:00:00"
    
    # Use environment variables or current directory (user can configure via local_config.sh)
    project_root: str = os.environ.get("PROJECT_ROOT", os.getcwd())
    log_dir: str = os.environ.get("LOG_PATH", "./logs")
    venv_path: str = os.environ.get("VENV_PATH", "")
    
    modules: List[str] = field(default_factory=lambda: [
        "GCCcore/11.3.0",
        "Python/3.10.4",
        "GCC/11.3.0",
        "CUDA/11.8.0",
        "cuDNN/8.7.0.84-CUDA-11.8.0"
    ])


def get_default_config():
    """Get default configuration"""
    return {
        'grouped_rvq': GroupedRVQConfig(),
        'entropy_model': EntropyModelConfig(),
        'rate_control': RateControlConfig(),
        'evaluation': EvaluationConfig(),
        'data': DataConfig(),
        'training': TrainingConfig(),
        'slurm': SlurmConfig(),
    }

