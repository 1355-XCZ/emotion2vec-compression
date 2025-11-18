# Emotion Information Bottleneck: Precise Rate-Distortion Control for Emotion2Vec Representations

**English** | [中文](#中文版)

---

## Overview

This repository implements a **precise bitrate control framework** for compressing emotion2vec representations based on information bottleneck theory. We achieve ±1 BPF (bits per frame) accuracy through:

- **Grouped Residual Vector Quantization (Grouped-RVQ)**: 12 groups × 3 layers with adaptive SKIP mechanism
- **Intra-frame Autoregressive Entropy Model**: Transformer-based conditional probability estimation
- **Entropy-Constrained Vector Quantization (ECVQ)**: Rate-distortion optimization with binary search

### Key Findings

- **Stepwise degradation**: Emotion information exhibits clear critical thresholds at **200 BPF** and **100 BPF**
- **Neutral robustness**: Neutral emotion maintains ~100% accuracy even at 10 BPF extreme compression
- **Emotion hierarchy**: High-arousal emotions (anger, sadness, surprise) degrade significantly below 100 BPF
- **Continuous control**: Achieves fine-grained bitrate sampling (10-300 BPF) vs. traditional discrete layer switching

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ / 12.2+
- GPU with 16GB+ memory (for training) or 8GB+ (for evaluation)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/MyAudioResearchModel.git
cd MyAudioResearchModel

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

**Create your local configuration file** (copy and modify):

```bash
cp local_config.sh.example local_config.sh
```

Edit `local_config.sh` with your settings:

```bash
# local_config.sh
export PROJECT_ROOT="/path/to/MyAudioResearchModel"
export VENV_PATH="${PROJECT_ROOT}/venv"
export DATASET_ROOT="${PROJECT_ROOT}/data"          # Raw audio datasets
export EVAL_FEATURES_ROOT="${PROJECT_ROOT}/data"    # Extracted emotion2vec features
export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
```

**Note**: This file contains sensitive paths and is git-ignored. See `local_config.sh.example` for a template.

---

## Reproduction Pipeline

We provide a **complete reproduction pipeline** for reviewers to regenerate all paper results.

### Step 1: Environment Verification

```bash
# Test environment installation (creates temporary venv)
sbatch scripts/test_env.slurm

# Or test locally
python test_dependencies.py
```

### Step 2: Prepare Evaluation Subset

The repository includes a **reproducible evaluation subset** (`data_subset/`) with 100 samples per emotion:

```bash
# If you need to regenerate:
python prepare_evaluation_subset.py \
    --source-data data \
    --target-data data_subset \
    --samples 100 \
    --seed 42
```

**Data Statistics**:
- **ESD**: 5 emotions × 100 = 500 samples
- **IEMOCAP**: 4 emotions × 100 = 400 samples  
- **RAVDESS**: 7 emotions × 100 = 700 samples

**Important Note**: The evaluation subset (`data_subset/`) is a smaller version sampled from the full test datasets. Due to the reduced sample pool, the specific samples selected by the random seed (42) may differ from those in the paper's full evaluation. However, **the overall trends and conclusions remain consistent** across different sample selections. This subset is designed to enable quick verification of the reproduction pipeline while maintaining representative results.

### Step 3: Run Full Experiment

**Option A: Single Command (Recommended for reviewers)**

```bash
# On SLURM cluster (47 rate points × 3 datasets)
sbatch scripts/run_full_reproduce.slurm

# Or locally
bash reproduce_full_pipeline.sh
```

**Option B: Quick Test (4 rate points for verification)**

```bash
bash reproduce_test_pipeline.sh
```

**Option C: Step-by-Step**

```bash
# 1. Run evaluation for each dataset
python run_evaluation.py --dataset esd --data-root data_subset
python run_evaluation.py --dataset iemocap --data-root data_subset
python run_evaluation.py --dataset ravdess --data-root data_subset

# 2. Generate paper figures
python generate_paper_figures.py
```

### Step 4: View Results

All results are saved in `evaluation_results/`:

```bash
evaluation_results/
├── rate_sweep_ESD.json           # Evaluation results
├── rate_sweep_IEMOCAP.json
├── rate_sweep_RAVDESS.json
└── figures/                       # Paper figures
    ├── 1_Overall_Weighted_F1.png
    ├── 2_Overall_Model_Confidence.png
    ├── 3_Emotion_Accuracy_Vertical.png
    ├── 4_Emotion_Confidence_Vertical.png
    └── 5_Confusion_Matrices_3x4.png
```

---

## Training (Optional)

Pre-trained checkpoints are provided. To retrain from scratch:

### 1. Prepare Training Data

**Note**: Due to copyright restrictions and dataset size limitations, the complete training dataset (Emilia, 100 hours) is not included in this repository. If you need access to the full training data for reproduction purposes, please contact the author via email: **haoguangz@student.unimelb.edu.au**

```bash
# Download Emilia dataset (100h Chinese-English)
# Extract emotion2vec features
python extract_train_features.py --data-dir /path/to/emilia
```

### 2. Train Grouped-RVQ

```bash
# On SLURM cluster
sbatch scripts/run_train_rvq.slurm

# Or locally
python train_rvq.py
```

**Training Details**:
- **Duration**: ~48-72 hours on single L40S GPU
- **Memory**: 32GB GPU memory
- **Output**: `checkpoints/grouped_rvq_best.pt`

### 3. Train Entropy Model

```bash
# On SLURM cluster
sbatch scripts/run_train_entropy.slurm

# Or locally
python train_entropy.py
```

**Training Details**:
- **Duration**: ~24-36 hours on single L40S GPU
- **Memory**: 64GB GPU memory
- **Data**: 9 λ values × 37,722 samples = 339,498 augmented samples
- **Output**: `checkpoints/entropy_model_best.pt`

---

## Project Structure

```
MyAudioResearchModel/
├── config.py                    # Configuration dataclasses
├── grouped_rvq.py              # Grouped-RVQ implementation
├── entropy_model.py            # Autoregressive Transformer entropy model
├── rate_controller.py          # Binary search for λ optimization
├── bucket_sampler.py           # Length-based batch sampling
├── data_loader.py              # DataLoader with bucketing support
├── datasets/                   # Dataset implementations
│   ├── base_dataset.py         # Abstract base class
│   ├── esd_dataset.py          # ESD (English + Chinese)
│   ├── iemocap_dataset.py      # IEMOCAP
│   └── ravdess_dataset.py      # RAVDESS
├── evaluation/                 # Evaluation pipeline
│   ├── emotion_classifier.py   # Emotion2vec classifier wrapper
│   ├── method_rate_sweep.py    # Rate sweep evaluation
│   ├── method_layer_sweep.py   # Layer sweep evaluation
│   └── analyzer.py             # Result analysis and plotting
├── train_rvq.py               # Train Grouped-RVQ (Stage 1)
├── train_entropy.py           # Train entropy model (Stage 2)
├── run_evaluation.py          # Main evaluation script
├── generate_paper_figures.py  # Generate all paper figures
├── reproduce_full_pipeline.sh # Complete reproduction script
├── reproduce_test_pipeline.sh # Quick test script
├── test_dependencies.py       # Environment verification
├── prepare_evaluation_subset.py # Data subset preparation
├── data_subset/               # Evaluation data (committed for reviewers)
│   ├── ESD/
│   ├── IEMOCAP/
│   └── RAVDESS/
├── checkpoints/               # Pre-trained models
│   ├── grouped_rvq_best.pt
│   └── entropy_model_best.pt
├── scripts/                   # SLURM job scripts
│   ├── run_train_rvq.slurm
│   ├── run_train_entropy.slurm
│   ├── run_full_reproduce.slurm
│   ├── test_env.slurm
│   └── test_reproduce.slurm
├── requirements.txt           # Python dependencies
├── local_config.sh.example    # Configuration template
└── README.md                  # This file
```

---

## Core Methodology

### 1. Grouped-RVQ Architecture

```python
# 768-dim emotion2vec features → 12 groups of 64-dim
# Each group: 3 layers residual quantization + SKIP option

Input: x_t ∈ ℝ^768
Split: [x_{t,1}, ..., x_{t,12}]  # 12 groups × 64 dim

For each group g:
  For each layer m = 1,2,3:
    z_{t,g,m} ∈ {0,...,127, SKIP}
    if z_{t,g,m} ≠ SKIP:
      residual -= codebook[z_{t,g,m}]

Output: 36 tokens per frame (12 groups × 3 layers)
```

**Key Features**:
- **Total codebook count**: 36 (12 groups × 3 layers)
- **Codebook size**: 128 per layer
- **SKIP mechanism**: Adaptive early stopping for simple groups
- **Average perplexity**: >95% across all layers

### 2. Entropy Model Training

```python
# Multi-λ data augmentation strategy
Λ = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0}

For each sample x:
  For each λ ∈ Λ:
    z^(λ) = ECVQ_encode(x, λ)
    Training_set += z^(λ)

# Autoregressive modeling (intra-frame only)
q(z_l | z_{<l}) = Softmax(Transformer(Embed(z_{<l})))
```

**Model Configuration**:
- **Architecture**: Causal Transformer
- **Layers**: 6
- **Hidden dim**: 256
- **Attention heads**: 8
- **Vocabulary**: 129 (128 codewords + 1 SKIP)
- **Context**: Intra-frame (L=36 tokens)

### 3. ECVQ Rate-Distortion Optimization

```python
# At each position (g,m), select action minimizing:
J(k) = ||r_{g,m} - c_k||^2 + λ · (-log₂ q(k | context))
       \_________________/     \____________________/
         Distortion             Code length (bits)

# Binary search for λ to achieve target BPF
while |BPF_actual - BPF_target| > tolerance:
  λ_new = binary_search_update(λ, BPF_actual, BPF_target)
```

**Convergence**: Typically 5-10 iterations to achieve ±1 BPF accuracy.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{zhou2025emotion,
  title={Emotion Information Bottleneck: Precise Rate-Distortion Control for Emotion Recognition},
  author={Zhou, Haoguang},
  school={University of Melbourne},
  year={2025},
  type={Master's Thesis}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Troubleshooting

### Common Issues

**1. Import errors for `modelscope` or `funasr`**
```bash
pip install --upgrade modelscope funasr
```

**2. CUDA out of memory during evaluation**
```bash
# Reduce batch size in config.py
batch_size = 4  # Default is 8
```

**3. Binary search fails to converge at low BPF (e.g., 10 BPF)**
```bash
# This is expected - increase max_iterations or adjust lambda_init
max_iterations = 100  # Default is 50
```

**4. Dataset not found errors**
```bash
# Ensure local_config.sh is properly configured
source local_config.sh
echo $DATASET_ROOT  # Should point to your data directory
```

---

## Contact

For questions or issues, please contact:
- **Email**: haoguangz@student.unimelb.edu.au
- **GitHub Issues**: [Open an issue](https://github.com/yourusername/MyAudioResearchModel/issues)

---

# 中文版

## 概述

本仓库实现了基于信息瓶颈理论的 **emotion2vec 表征精确码率控制框架**。通过以下方法实现 ±1 BPF（bits per frame）精度：

- **分组残差向量量化（Grouped-RVQ）**：12组 × 3层，带自适应SKIP机制
- **帧内自回归熵模型**：基于Transformer的条件概率估计
- **熵约束向量量化（ECVQ）**：率失真优化 + 二分搜索

### 核心发现

- **阶梯式退化**：情感信息在 **200 BPF** 和 **100 BPF** 处存在明显临界阈值
- **中性情感鲁棒性**：即使在10 BPF极端压缩下仍保持~100%准确率
- **情感层次结构**：高唤醒情感（愤怒、悲伤、惊讶）在100 BPF以下显著退化
- **连续控制**：实现细粒度码率采样（10-300 BPF）vs 传统离散层切换

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ / 12.2+
- GPU：16GB+内存（训练）或 8GB+（评估）

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/MyAudioResearchModel.git
cd MyAudioResearchModel

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 配置

**创建本地配置文件**（复制并修改）：

```bash
cp local_config.sh.example local_config.sh
```

编辑 `local_config.sh`：

```bash
# local_config.sh
export PROJECT_ROOT="/path/to/MyAudioResearchModel"
export VENV_PATH="${PROJECT_ROOT}/venv"
export DATASET_ROOT="${PROJECT_ROOT}/data"          # 原始音频数据集
export EVAL_FEATURES_ROOT="${PROJECT_ROOT}/data"    # 提取的emotion2vec特征
export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
```

**注意**：此文件包含敏感路径，已被git忽略。参见 `local_config.sh.example` 模板。

---

## 复现流程

我们为审稿人提供了**完整的复现管道**以重新生成所有论文结果。

### 步骤1：环境验证

```bash
# 测试环境安装（创建临时虚拟环境）
sbatch scripts/test_env.slurm

# 或本地测试
python test_dependencies.py
```

### 步骤2：准备评估子集

仓库包含**可复现的评估子集**（`data_subset/`），每个情感100个样本：

```bash
# 如需重新生成：
python prepare_evaluation_subset.py \
    --source-data data \
    --target-data data_subset \
    --samples 100 \
    --seed 42
```

**数据统计**：
- **ESD**: 5情感 × 100 = 500样本
- **IEMOCAP**: 4情感 × 100 = 400样本
- **RAVDESS**: 7情感 × 100 = 700样本

**重要说明**：评估子集（`data_subset/`）是从完整测试数据集中抽样的较小版本。由于样本池减少，随机种子（42）选择的具体样本可能与论文完整评估中的样本不同。但是，**整体趋势和结论在不同样本选择下保持一致**。此子集旨在快速验证复现管道的同时保持结果的代表性。

### 步骤3：运行完整实验

**方式A：单命令（推荐审稿人使用）**

```bash
# SLURM集群（47码率点 × 3数据集）
sbatch scripts/run_full_reproduce.slurm

# 或本地运行
bash reproduce_full_pipeline.sh
```

**方式B：快速测试（4码率点验证）**

```bash
bash reproduce_test_pipeline.sh
```

**方式C：分步执行**

```bash
# 1. 对每个数据集运行评估
python run_evaluation.py --dataset esd --data-root data_subset
python run_evaluation.py --dataset iemocap --data-root data_subset
python run_evaluation.py --dataset ravdess --data-root data_subset

# 2. 生成论文图表
python generate_paper_figures.py
```

### 步骤4：查看结果

所有结果保存在 `evaluation_results/`：

```bash
evaluation_results/
├── rate_sweep_ESD.json           # 评估结果
├── rate_sweep_IEMOCAP.json
├── rate_sweep_RAVDESS.json
└── figures/                       # 论文图表
    ├── 1_Overall_Weighted_F1.png
    ├── 2_Overall_Model_Confidence.png
    ├── 3_Emotion_Accuracy_Vertical.png
    ├── 4_Emotion_Confidence_Vertical.png
    └── 5_Confusion_Matrices_3x4.png
```

---

## 训练（可选）

已提供预训练模型。如需从头训练：

### 1. 准备训练数据

**注意**：由于版权限制和数据集大小限制，完整的训练数据集（Emilia，100小时）未包含在本仓库中。如需获取完整训练数据用于复现，请通过邮箱联系作者：**haoguangz@student.unimelb.edu.au**

```bash
# 下载Emilia数据集（100h中英文）
# 提取emotion2vec特征
python extract_train_features.py --data-dir /path/to/emilia
```

### 2. 训练Grouped-RVQ

```bash
# SLURM集群
sbatch scripts/run_train_rvq.slurm

# 或本地
python train_rvq.py
```

**训练详情**：
- **时长**：单张L40S GPU约48-72小时
- **显存**：32GB
- **输出**：`checkpoints/grouped_rvq_best.pt`

### 3. 训练熵模型

```bash
# SLURM集群
sbatch scripts/run_train_entropy.slurm

# 或本地
python train_entropy.py
```

**训练详情**：
- **时长**：单张L40S GPU约24-36小时
- **显存**：64GB
- **数据**：9个λ值 × 37,722样本 = 339,498增强样本
- **输出**：`checkpoints/entropy_model_best.pt`

---

## 核心方法

### 1. Grouped-RVQ架构

```python
# 768维emotion2vec特征 → 12组64维
# 每组：3层残差量化 + SKIP选项

输入: x_t ∈ ℝ^768
分割: [x_{t,1}, ..., x_{t,12}]  # 12组 × 64维

对于每组g:
  对于每层m = 1,2,3:
    z_{t,g,m} ∈ {0,...,127, SKIP}
    if z_{t,g,m} ≠ SKIP:
      residual -= codebook[z_{t,g,m}]

输出: 每帧36个token（12组 × 3层）
```

### 2. 熵模型训练

```python
# 多λ数据增强策略
Λ = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0}

对于每个样本x:
  对于每个λ ∈ Λ:
    z^(λ) = ECVQ_encode(x, λ)
    训练集 += z^(λ)

# 自回归建模（仅帧内）
q(z_l | z_{<l}) = Softmax(Transformer(Embed(z_{<l})))
```

### 3. ECVQ率失真优化

```python
# 在每个位置(g,m)，选择最小化以下目标的动作：
J(k) = ||r_{g,m} - c_k||^2 + λ · (-log₂ q(k | context))
       \_________________/     \____________________/
         失真                    码长（比特）

# 二分搜索λ以达到目标BPF
while |BPF_actual - BPF_target| > tolerance:
  λ_new = binary_search_update(λ, BPF_actual, BPF_target)
```

---

## 引用

如果您在研究中使用此代码，请引用：

```bibtex
@mastersthesis{zhou2025emotion,
  title={Emotion Information Bottleneck: Precise Rate-Distortion Control for Emotion Recognition},
  author={Zhou, Haoguang},
  school={University of Melbourne},
  year={2025},
  type={Master's Thesis}
}
```

---

## 常见问题

**1. `modelscope`或`funasr`导入错误**
```bash
pip install --upgrade modelscope funasr
```

**2. 评估时CUDA内存不足**
```bash
# 在config.py中减小batch size
batch_size = 4  # 默认是8
```

**3. 低BPF时二分搜索无法收敛（如10 BPF）**
```bash
# 这是预期的 - 增加最大迭代次数或调整lambda_init
max_iterations = 100  # 默认50
```

**4. 数据集未找到错误**
```bash
# 确保local_config.sh正确配置
source local_config.sh
echo $DATASET_ROOT  # 应指向你的数据目录
```

---

## 联系方式

如有问题，请联系：
- **邮箱**：haoguangz@student.unimelb.edu.au
- **GitHub Issues**：[提交问题](https://github.com/yourusername/MyAudioResearchModel/issues)

---

**Note**: Replace `yourusername` with your actual GitHub username before publishing.
