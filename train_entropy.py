"""
Train unconditional entropy model q(z) (Release Version)
Only train q(z), does not include conditional model q(z|y)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging
import json
import math
import random
import numpy as np

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from entropy_model import create_entropy_model
from data_loader import create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"✅ Random seed set: {seed}")


def masked_token_kl(student_logits, teacher_logits, valid_mask, temperature=1.0):
    """
    Calculate token-level KL(student || teacher), only count valid frames.
    logits: (B*T*L, V)
    valid_mask: (B, T) → flattened and repeated L times
    """
    BTL, V = student_logits.shape
    # Temperature smoothing
    t = temperature
    s_logprob = F.log_softmax(student_logits / t, dim=-1)
    t_prob     = F.softmax(teacher_logits / t, dim=-1).detach()

    kl = F.kl_div(s_logprob, t_prob, reduction='none', log_target=False)  # (BTL, V)
    kl = kl.sum(dim=-1)  # (BTL,)

    # Flatten mask
    # valid_mask: (B, T) → (B, T, L) → (BTL,)
    B = valid_mask.size(0); T = valid_mask.size(1)
    L = BTL // (B * T)
    mask_tokens = valid_mask.unsqueeze(-1).expand(-1, -1, L).reshape(-1).float()
    if mask_tokens.sum() == 0:
        return student_logits.new_tensor(0.0)
    return (kl * mask_tokens).sum() / mask_tokens.sum()


def extract_indices_from_rvq(rvq_model, dataloader, device, target_bpf=None, ecvq_lambda=None):
    """
    Extract discrete indices for all samples using trained RVQ (supports SKIP ECVQ)
    
    Args:
        rvq_model: trained GroupedResidualVQ
        dataloader: data loader
        device: device
        target_bpf: target bitrate (bits per frame), uses quantile gating (complex way)
        ecvq_lambda: directly specify λ value (simple way, used first)
    
    Returns:
        all_indices: List of (T, L) tensors (uint8 format)
        all_labels: List of scalar labels
        all_valid_masks: List of (T,) bool tensors
    """
    rvq_model.eval()
    all_indices = []
    all_labels = []
    all_valid_masks = []
    
    logger.info("Extracting discrete indices...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            lengths = batch['lengths'].to(device, non_blocking=True)
            
            B, T, D = features.shape
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            valid_bt = (t_idx < lengths.unsqueeze(1))  # (B, T)
            
            # Encode with RVQ
            if ecvq_lambda is not None:
                # Simple way: directly use λ value (old code way)
                _, indices, _, _ = rvq_model(
                    features,
                    lambda_rate=torch.tensor(ecvq_lambda, device=device),
                    entropy_model=None,  # Key: go through simplified ECVQ branch
                    valid_mask=valid_bt
                )
            else:
                # Quantile gating way (new way)
                _, indices, _, _ = rvq_model(
                    features,
                    lambda_rate=None,
                    entropy_model=None,
                    valid_mask=valid_bt,
                    target_bpf=target_bpf  # Use target bpf quantile gating
                )
            
            # Store per sample (keep variable length, use uint8 to save memory)
            for i in range(B):
                T_i = int(lengths[i].item())
                # Key fix: index range 0-128, use uint8 storage (8x smaller)
                idx_i = indices[i, :T_i, :].to(torch.uint8).cpu()  # (T_i, L)
                all_indices.append(idx_i)
                all_labels.append(labels[i].cpu())  # scalar
                all_valid_masks.append(torch.ones(T_i, dtype=torch.bool))  # (T_i,)
    
    logger.info(f"✅ Extraction complete: {len(all_indices)} samples")
    return all_indices, all_labels, all_valid_masks


def train_entropy_model(
    entropy_model,
    all_indices,
    all_labels,
    all_valid_masks,
    training_config,
    entropy_config,
    model_name="q_z",
    teacher_model=None  # New: pass in q(z) as teacher (only used for q(z|y))
):
    """
    Train entropy model (supports conditional model with KL traction)

    Args:
        entropy_model: AutoregressiveEntropyModel or ConditionalEntropyModel
        all_indices: List of (T, L) discrete indices
        all_labels: List of labels
        all_valid_masks: List of (T,) bool masks
        training_config: Training configuration
        entropy_config: entropy model configuration
        model_name: model name ("q_z" or "q_z_y")
        teacher_model: optional teacher model (q(z)), used for KL traction for q(z|y)
    """
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    entropy_model = entropy_model.to(device)
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()  # teacher fixed

    # Only optimize parameters that require training (supports "fine-tune from q(z)")
    optim_params = [p for p in entropy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=entropy_config.learning_rate,
        weight_decay=entropy_config.weight_decay
    )
    
    # Create directory
    checkpoint_dir = Path(training_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"Start training entropy model: {model_name}")
    logger.info(f"  - Number of samples: {len(all_indices)}")
    logger.info(f"  - Vocabulary size: {entropy_model.V}")
    logger.info(f"  - Intra-frame sequence length: {entropy_model.L}")
    logger.info("=" * 80)
    
    best_loss = float('inf')
    start_epoch = 1
    num_samples = len(all_indices)
    
    # Early stopping mechanism
    patience_counter = 0
    best_epoch = 0
    
    # Decide whether to resume training from checkpoint based on config (only load checkpoint from same experiment)
    if training_config.resume_from_checkpoint:
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
        if checkpoint_path.exists():
            logger.info(f"Resume training from current experiment directory: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            entropy_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 1) + 1
            logger.info(f"✅ Training resumed: epoch={start_epoch}, best_loss={best_loss:.4f}")
        else:
            logger.warning(
                f"⚠️  Config requires resuming from checkpoint, but does not exist in current experiment directory: {checkpoint_path}\n"
                f"   Training from scratch. To train from scratch, set TrainingConfig.resume_from_checkpoint=False"
            )
    else:
        logger.info("🚀 Train entropy model from scratch (not loading checkpoint)")
    
    # Apply data fraction (for fast experiments)
    if entropy_config.train_data_fraction < 1.0:
        subset_size = int(num_samples * entropy_config.train_data_fraction)
        all_indices = all_indices[:subset_size]
        all_labels = all_labels[:subset_size]
        all_valid_masks = all_valid_masks[:subset_size]
        num_samples = len(all_indices)
        logger.info(f"⚙️  Using {entropy_config.train_data_fraction:.1%} of training data, total {num_samples} samples")
    
    batch_size = entropy_config.batch_size
    num_epochs = entropy_config.num_epochs
    
    # Get actual optimization config (considering master switch)
    opt_cfg = entropy_config.get_effective_config()
    
    # Log printing (using borders to show hierarchy)
    logger.info("")
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│ ⚙️  Entropy Model Training Configuration" + " " * 49 + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(f"│  Basic parameters:                                                       │")
    logger.info(f"│    • batch_size: {batch_size:<58} │")
    logger.info(f"│    • num_epochs: {num_epochs:<58} │")
    logger.info(f"│    • Training samples: {num_samples} ({entropy_config.train_data_fraction:.0%})" + " " * (70 - len(f"{num_samples} ({entropy_config.train_data_fraction:.0%})") - 20) + "│")
    logger.info(f"│                                                                              │")
    
    opt_status = '✅ Enabled' if entropy_config.enable_optimizations else '❌ Disabled'
    logger.info(f"│  Performance optimization:                                                   │")
    logger.info(f"│    • Master switch: {opt_status:<58} │")
    
    if entropy_config.enable_optimizations:
        max_frames = opt_cfg['max_frames_per_sample']
        desc_max_frames = 'Unlimited' if max_frames == 0 else f'Limit to {max_frames} frames'
        logger.info(f"│    • Frame cropping: {max_frames} ({desc_max_frames})" + " " * (70 - len(f"{max_frames} ({desc_max_frames})") - 18) + "│")
        
        stride = opt_cfg['frame_stride']
        desc_stride = 'Keep all' if stride == 1 else f'Sample 1 every {stride} frames'
        logger.info(f"│    • Frame downsampling: {stride} ({desc_stride})" + " " * (70 - len(f"{stride} ({desc_stride})") - 22) + "│")
        
        amp_status = '✅ BF16' if opt_cfg['use_amp'] else '❌ FP32'
        logger.info(f"│    • Mixed precision: {amp_status:<55} │")
        
        tf32_status = '✅ Enabled' if opt_cfg['use_tf32'] else '❌ Disabled'
        logger.info(f"│    • TF32 acceleration: {tf32_status:<53} │")
    
    logger.info("└" + "─" * 78 + "┘")
    logger.info("")
    
    # Enable TF32 acceleration (A100/H100)
    if opt_cfg['use_tf32'] and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    # Used to record whether empty batch has been warned
    _empty_batch_warned = False
    
    for epoch in range(start_epoch, num_epochs + 1):
        entropy_model.train()
        total_loss = 0
        total_ce_loss = 0
        total_kl_loss = 0
        total_acc = 0
        num_batches = 0
        
        # Shuffle
        import random
        indices_list = list(range(num_samples))
        random.shuffle(indices_list)
        
        pbar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch}/{num_epochs}")
        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices_list = indices_list[start_idx:end_idx]
            
            # Collect batch (need padding)
            batch_data = [all_indices[i] for i in batch_indices_list]
            batch_labels = torch.stack([all_labels[i] for i in batch_indices_list])
            batch_masks = [all_valid_masks[i] for i in batch_indices_list]
            
            # Performance optimization: frame cropping and downsampling (apply based on master switch)
            if opt_cfg['max_frames_per_sample'] > 0 or opt_cfg['frame_stride'] > 1:
                cropped_data, cropped_masks, cropped_labels = [], [], []
                for idx, mask, label in zip(batch_data, batch_masks, batch_labels):
                    original_T = idx.shape[0]
                    
                    # Frame downsampling (improvement: if original sequence length < stride, keep original)
                    if opt_cfg['frame_stride'] > 1:
                        if original_T >= opt_cfg['frame_stride']:
                            idx = idx[::opt_cfg['frame_stride']]
                            mask = mask[::opt_cfg['frame_stride']]
                        # If original sequence length < stride, keep original (avoid empty sequence)
                        # This way even short sequences can be processed
                    
                    # Frame cropping
                    T_i = idx.shape[0]
                    if opt_cfg['max_frames_per_sample'] > 0 and T_i > opt_cfg['max_frames_per_sample']:
                        import random
                        start = random.randint(0, T_i - opt_cfg['max_frames_per_sample'])
                        idx = idx[start:start + opt_cfg['max_frames_per_sample']]
                        mask = mask[start:start + opt_cfg['max_frames_per_sample']]
                    
                    # Fallback: skip empty sequences (theoretically shouldn't happen, but keep just in case)
                    if idx.shape[0] > 0:
                        cropped_data.append(idx)
                        cropped_masks.append(mask)
                        cropped_labels.append(label)
                
                batch_data = cropped_data
                batch_masks = cropped_masks
                batch_labels = torch.stack(cropped_labels) if cropped_labels else batch_labels[:0]
            
            # Skip empty batch (extreme case where all samples are filtered)
            if len(batch_data) == 0:
                # Log warning (only on first encounter)
                if not _empty_batch_warned:
                    logger.warning(
                        f"⚠️  Epoch {epoch}: Encountered empty batch (all samples filtered), "
                        f"possibly due to frame_stride={opt_cfg['frame_stride']} or data too short. "
                        f"Recommend checking data length or adjusting frame_stride."
                    )
                    _empty_batch_warned = True
                continue
            
            # Padding
            max_T = max(item.shape[0] for item in batch_data)
            L = batch_data[0].shape[1]
            B_batch = len(batch_data)
            
            indices_batch = torch.full((B_batch, max_T, L), 
                                      entropy_model.SKIP_ID, 
                                      dtype=torch.long)
            valid_mask_batch = torch.zeros(B_batch, max_T, dtype=torch.bool)
            
            for i, (idx, mask) in enumerate(zip(batch_data, batch_masks)):
                T_i = idx.shape[0]
                # Convert uint8 indices back to long for embedding
                indices_batch[i, :T_i, :] = idx.to(torch.long)
                valid_mask_batch[i, :T_i] = mask
            
            # Move to GPU (optimization: pin_memory + non_blocking)
            indices_batch = indices_batch.pin_memory().to(device, non_blocking=True)
            batch_labels = batch_labels.pin_memory().to(device, non_blocking=True)
            valid_mask_batch = valid_mask_batch.pin_memory().to(device, non_blocking=True)
            
            # Forward pass (decide whether to use mixed precision based on master switch)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=opt_cfg['use_amp'] and torch.cuda.is_available()):
                if entropy_model.condition_on_label:
                    logits, targets = entropy_model(indices_batch, batch_labels)
                else:
                    logits, targets = entropy_model(indices_batch, None)

                # Calculate loss (with mask)
                logits_flat = logits.reshape(-1, entropy_model.V)
                targets_flat = targets.reshape(-1)

                ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                ce = ce.view(B_batch, max_T, L)

                # Optional enhancement: weight for SKIP positions
                # skip_weight < 1.0: reduce SKIP weight, encourage normal encoding (avoid excessive SKIP)
                # skip_weight > 1.0: increase SKIP weight, improve p_skip (for extremely low bitrate)
                # skip_weight = 1.0: no weighting (default)
                skip_weight = getattr(entropy_config, "skip_weight", 1.0)
                if abs(skip_weight - 1.0) > 1e-6:  # Apply if != 1
                    targets_3d = targets.view(B_batch, max_T, L)
                    token_weights = torch.ones_like(ce)
                    token_weights[targets_3d == entropy_model.SKIP_ID] = skip_weight
                    ce = ce * token_weights

                # Apply mask (only count valid frames)
                mask_3d = valid_mask_batch.unsqueeze(-1).float()
                ce_masked = ce * mask_3d
                ce_loss = ce_masked.sum() / max(mask_3d.sum(), 1.0)

                # KL traction (only when teacher_model is provided and weight > 0)
                kl_loss = logits_flat.new_tensor(0.0)
                if teacher_model is not None and getattr(entropy_config, "kl_tether_weight", 0.0) > 0:
                    with torch.no_grad():
                        # teacher forward (without label)
                        t_logits, _ = teacher_model(indices_batch, None)
                    t_logits_flat = t_logits.reshape(-1, teacher_model.V)
                    kl_loss = masked_token_kl(
                        student_logits=logits_flat.float(),
                        teacher_logits=t_logits_flat.float(),
                        valid_mask=valid_mask_batch,
                        temperature=getattr(entropy_config, "kl_tether_temperature", 1.0),
                    ) * entropy_config.kl_tether_weight

            # Calculate loss outside AMP block to ensure it's always defined
            loss = ce_loss + kl_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                optim_params,
                entropy_config.max_grad_norm
            )
            optimizer.step()
            
            # Calculate accuracy (fix: expand frame-level mask to token-level)
            with torch.inference_mode():
                # Recalculate preds outside AMP (avoid precision issues)
                preds = logits_flat.float().argmax(dim=-1)  # (B*T*L,)
                # Expand frame-level mask (B, T) to token-level (B, T, L), then flatten to (B*T*L,)
                mask_tokens = valid_mask_batch.unsqueeze(-1).expand(-1, -1, L).reshape(-1).float()
                if mask_tokens.sum() > 0:
                    acc = ((preds == targets_flat).float() * mask_tokens).sum() / mask_tokens.sum()
                else:
                    acc = torch.tensor(0.0)
            
            # Accumulate statistics (after continue, ensure all variables are defined)
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kl_loss += kl_loss.item()
            total_acc += acc.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.3f}',
                'bpf': f'{ce_loss.item() / math.log(2):.2f}'   # Calculate bpf using CE, excluding KL
            })
        
        # Check if there are valid batches (prevent division by zero error when all batches are skipped)
        if num_batches == 0:
            logger.warning(f"⚠️  Epoch {epoch}: All batches were skipped, data filtering may be too strict!")
            continue
        
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_acc = total_acc / num_batches
        avg_bpf = avg_ce_loss / math.log(2)  # Calculate bpf using CE, excluding KL

        # Progress display
        progress_pct = epoch / num_epochs * 100
        logger.info("")
        logger.info(f"{'='*80}")
        logger.info(f"Epoch {epoch}/{num_epochs} ({progress_pct:.1f}%) - {model_name}")
        logger.info(f"  Total Loss: {avg_loss:.4f} | CE: {avg_ce_loss:.4f} | KL: {avg_kl_loss:.4f}")
        logger.info(f"  Acc: {avg_acc:.3f} | BPF: {avg_bpf:.2f}")
        logger.info(f"{'='*80}")
        
        # Early stopping mechanism: based on Loss (lower is better)
        if entropy_config.enable_early_stopping:
            # Check if there's significant improvement
            improvement = best_loss - avg_loss  # Loss decrease as positive value
            
            # First epoch or significant improvement
            if best_loss == float('inf') or improvement > entropy_config.early_stopping_min_delta * abs(avg_loss):
                # Significant improvement (relative improvement > 1%), save best model and reset patience
                best_loss = avg_loss
                best_epoch = epoch
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': entropy_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'acc': avg_acc,
                    'hparams': {
                        'd_model': entropy_config.d_model,
                        'num_layers': entropy_config.num_layers,
                        'num_heads': entropy_config.num_heads,
                        'd_ff': entropy_config.d_ff,
                        'dropout': entropy_config.dropout,
                        'max_seq_len': entropy_config.max_seq_len,
                        'causal': entropy_config.causal,
                        'skip_weight': entropy_config.skip_weight,
                        'learning_rate': entropy_config.learning_rate,
                        'weight_decay': entropy_config.weight_decay,
                        'warmup_steps': entropy_config.warmup_steps,
                        'max_grad_norm': entropy_config.max_grad_norm,
                    }
                }
                save_path = checkpoint_dir / f"{model_name}_best.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, save_path)
                logger.info(f"✅ Saved best model: {save_path} (Loss improvement: {improvement:.4f}, {improvement/best_loss*100:.2f}%)")
            else:
                # No significant improvement, increase patience counter
                patience_counter += 1
                relative_improvement = improvement / best_loss * 100 if best_loss > 0 else 0
                logger.info(f"⏸️  No significant improvement (improvement: {relative_improvement:.3f}%, threshold: {entropy_config.early_stopping_min_delta*100:.1f}%)")
                logger.info(f"   Early stopping: {patience_counter}/{entropy_config.early_stopping_patience}")
                
                # Check if should early stop
                if patience_counter >= entropy_config.early_stopping_patience:
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"🛑 Early Stopping triggered!")
                    logger.info(f"   - Best epoch: {best_epoch}")
                    logger.info(f"   - Best Loss: {best_loss:.4f}")
                    logger.info(f"   - No significant improvement for {patience_counter} consecutive epochs")
                    logger.info("=" * 80)
                    break  # Exit training loop early
        else:
            # Not using early stopping, keep original logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': entropy_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'acc': avg_acc,
                    'hparams': {
                        'd_model': entropy_config.d_model,
                        'num_layers': entropy_config.num_layers,
                        'num_heads': entropy_config.num_heads,
                        'd_ff': entropy_config.d_ff,
                        'dropout': entropy_config.dropout,
                        'max_seq_len': entropy_config.max_seq_len,
                        'causal': entropy_config.causal,
                        'skip_weight': entropy_config.skip_weight,
                        'learning_rate': entropy_config.learning_rate,
                        'weight_decay': entropy_config.weight_decay,
                        'warmup_steps': entropy_config.warmup_steps,
                        'max_grad_norm': entropy_config.max_grad_norm,
                    }
                }
                save_path = checkpoint_dir / f"{model_name}_best.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, save_path)
                logger.info(f"✅ Saved best model: {save_path}")
    
    logger.info("=" * 80)
    logger.info(f"✅ {model_name} training complete! Best Loss: {best_loss:.4f}")
    logger.info("=" * 80)


def main():
    """Main function: Train unconditional entropy model q(z) (Release Version)"""
    # Load configuration
    config = get_default_config()
    grouped_rvq_config = config['grouped_rvq']
    entropy_model_config = config['entropy_model']
    data_config = config['data']
    training_config = config['training']
    
    # Set random seed (ensure reproducibility)
    set_seed(data_config.seed)
    
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load trained RVQ
    logger.info("Initializing RVQ model...")
    rvq_model = GroupedResidualVQ(grouped_rvq_config).to(device)
    
    # Load RVQ checkpoint (using local training checkpoint)
    rvq_checkpoint_path = Path('checkpoints/grouped_rvq_best.pt')
    
    if rvq_checkpoint_path.exists():
        logger.info(f"Loading RVQ checkpoint: {rvq_checkpoint_path}")
        checkpoint = torch.load(rvq_checkpoint_path, map_location=device)
        rvq_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ RVQ checkpoint loaded")
    else:
        raise FileNotFoundError(
            f"RVQ checkpoint does not exist: {rvq_checkpoint_path}\n"
            f"Please train RVQ first: python train_rvq.py or sbatch run_train_rvq.slurm"
        )
    
    # 2. Load data (q(z) uses emilia dataset)
    # Key fix: index extraction stage uses EntropyModelConfig parameters (not TrainingConfig)
    logger.info("Loading data (q(z) uses emilia dataset)...")
    
    # Create temporary config, use EntropyModelConfig batch_size and max_frames
    from dataclasses import replace
    
    extract_training_config = replace(
        training_config,
        batch_size=entropy_model_config.batch_size,  # Use entropy model batch_size
        num_workers=0  # Reduce workers to prevent extra memory usage
    )
    
    extract_data_config = replace(
        data_config,
        max_frames=entropy_model_config.max_frames_per_sample,  # Use entropy model max_frames
        max_samples=int(37722 * entropy_model_config.train_data_fraction) if entropy_model_config.train_data_fraction < 1.0 else None  # Key: index extraction also applies data fraction
    )
    
    logger.info(f"Index extraction config: batch_size={entropy_model_config.batch_size}, "
                f"max_frames={entropy_model_config.max_frames_per_sample} frames "
                f"({entropy_model_config.max_frames_per_sample/50:.1f} seconds)")
    
    train_loader, val_loader = create_dataloaders(
        extract_data_config, extract_training_config, grouped_rvq_config,
        use_bucketing=True,   # Enable bucketing to speed up training
        num_buckets=10
    )
    
    # 3. Extract discrete indices from training set using RVQ (include SKIP, let entropy model learn SKIP probability)
    logger.info("Extracting training set indices (use simplified ECVQ to generate samples with SKIP)...")
    
    # Use multiple λ values to generate diverse SKIP patterns (restore old code simple way)
    # Larger λ means more SKIP, lower bitrate
    lambda_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    logger.info(f"Using {len(lambda_grid)} λ values: {lambda_grid}")
    
    # Create indices cache directory
    indices_cache_dir = Path("indices_cache")
    indices_cache_dir.mkdir(exist_ok=True)
    
    train_indices, train_labels, train_masks = [], [], []
    
    for idx, lam in enumerate(lambda_grid):
        cache_file = indices_cache_dir / f"lambda_{lam:.2f}.pt"
        
        # Check cache
        if cache_file.exists():
            logger.info(f"  λ={lam}: Loading from cache...")
            cached = torch.load(cache_file)
            idxs = cached['indices']
            labs = cached['labels']
            masks = cached['masks']
            logger.info(f"    ✅ Loaded from cache: {len(idxs)} samples")
        else:
            logger.info(f"  λ={lam} Extracting indices...")
            idxs, labs, masks = extract_indices_from_rvq(
                rvq_model, train_loader, device, ecvq_lambda=lam
            )
            
            # Immediately save to disk (prevent loss from crash)
            torch.save({
                'indices': idxs,
                'labels': labs,
                'masks': masks
            }, cache_file)
            logger.info(f"    💾 Saved to cache: {cache_file.name}")
        
        train_indices += idxs
        train_labels += labs
        train_masks += masks
        logger.info(f"    → Accumulated {len(train_indices)} samples")
    
    logger.info(f"✅ Total extracted {len(train_indices)} samples (contains various SKIP patterns, covers wide bitrate range)")
    
    # 4. Train unconditional entropy model q(z) (includes SKIP probability)
    logger.info("\n" + "=" * 80)
    logger.info("Train unconditional entropy model q(z) (Release Version)")
    logger.info("  Note: Training data includes SKIP, model can learn true SKIP probability")
    logger.info("=" * 80)
    
    entropy_model_q_z = create_entropy_model(
        entropy_model_config, 
        grouped_rvq_config
    )
    
    train_entropy_model(
        entropy_model_q_z,
        train_indices,
        train_labels,
        train_masks,
        training_config,
        entropy_model_config,
        model_name="entropy_model"  # Save as entropy_model_best.pt (matches evaluation script)
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ Unconditional entropy model q(z) training complete!")
    logger.info("✅ Checkpoint saved: checkpoints/entropy_model_best.pt")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
