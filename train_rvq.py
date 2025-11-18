"""
Train Grouped RVQ (Stage 1)
Goal: Train stable codebook
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging
import json
import sys
import random
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except:
    HAS_TENSORBOARD = False

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from data_loader import create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"✅ Random seed set: {seed} (deterministic mode)")
    else:
        torch.backends.cudnn.benchmark = True
        logger.info(f"✅ Random seed set: {seed}")


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing learning rate scheduler"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Update learning rate (with zero-division protection)"""
        self.current_step += 1
        
        if self.total_steps <= self.warmup_steps:
            # Small data/few steps: simple warmup
            lr = self.base_lr * min(1.0, self.current_step / max(1, self.warmup_steps))
        else:
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr = self.base_lr * self.current_step / self.warmup_steps
            else:
                # Cosine annealing (with protection)
                denom = max(1, self.total_steps - self.warmup_steps)
                progress = (self.current_step - self.warmup_steps) / denom
                progress = float(np.clip(progress, 0.0, 1.0))
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def _accumulate_codebook_usage(usage_acc, indices, valid_bt, skip_id, K):
    """
    Accumulate codebook usage statistics (by batch, avoid dimension mismatch)
    
    Args:
        usage_acc: accumulator (List[dict] or None)
        indices: (B, T, L) codebook indices
        valid_bt: (B, T) bool mask
        skip_id: SKIP token id
        K: codebook size
    
    Returns:
        updated usage_acc
    """
    B, T, L = indices.shape
    
    # Initialize accumulator (including histogram for perplexity)
    if usage_acc is None:
        usage_acc = [{'used': set(), 'count': 0, 'skips': 0, 'histogram': {}} for _ in range(L)]
    
    # Move to CPU for processing (avoid GPU memory usage)
    idx_cpu = indices.cpu()
    mask_cpu = valid_bt.cpu()
    
    # Statistics per layer
    for l in range(L):
        layer_idx = idx_cpu[:, :, l]  # (B, T)
        layer_idx = layer_idx[mask_cpu]  # (N_valid,) only valid frames
        
        if layer_idx.numel() == 0:
            continue
        
        # Count SKIP
        skips = (layer_idx == skip_id).sum().item()
        
        # Count used codewords (excluding SKIP)
        non_skip = layer_idx[layer_idx != skip_id]
        if non_skip.numel() > 0:
            usage_acc[l]['used'].update(non_skip.tolist())
            
            # Accumulate histogram (for perplexity calculation)
            for idx in non_skip.tolist():
                usage_acc[l]['histogram'][idx] = usage_acc[l]['histogram'].get(idx, 0) + 1
        
        # Accumulate count
        usage_acc[l]['count'] += layer_idx.numel()
        usage_acc[l]['skips'] += skips
    
    return usage_acc


def train_model(rvq_config, data_config, training_config):
    """Train Grouped RVQ"""
    # Set random seed for reproducibility
    set_seed(data_config.seed, deterministic=False)
    
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    for d in [training_config.checkpoint_dir, training_config.log_dir, training_config.results_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader = create_dataloaders(
        data_config, training_config, rvq_config,
        use_bucketing=training_config.use_bucketing,
        num_buckets=training_config.num_buckets
    )
    
    # Create model
    logger.info("Creating model...")
    model = GroupedResidualVQ(rvq_config).to(device)
    
    # Initialize codebook (kmeans_init requires one forward pass first)
    logger.info("Initializing codebook...")
    with torch.no_grad():
        # Create dummy batch for initialization
        dummy_batch = next(iter(train_loader))
        dummy_features = dummy_batch['features'][:min(8, len(dummy_batch['features']))].to(device)
        dummy_lengths = dummy_batch['lengths'][:min(8, len(dummy_batch['lengths']))].to(device)
        
        # Create mask
        B, T, D = dummy_features.shape
        t_idx = torch.arange(T, device=device).unsqueeze(0)
        valid_bt = (t_idx < dummy_lengths.unsqueeze(1))
        
        # Forward pass to initialize codebook
        _ = model(dummy_features, valid_mask=valid_bt)
    
    # Check trainable parameters
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_learnable = sum(p.numel() for p in learnable_params)
    
    logger.info(f"✅ Codebook initialization complete")
    logger.info(f"   - Total parameters: {n_params_total}")
    logger.info(f"   - Trainable parameters: {n_params_learnable}")
    
    # Decide whether to create optimizer based on trainable parameters
    if n_params_learnable == 0:
        # EMA mode: codebook auto-updates in forward, no optimizer needed
        optimizer = None
        scheduler = None
        use_step_scheduler = False
        logger.info("⚙️  EMA mode: codebook auto-updates via EMA, not using optimizer")
    else:
        # Learnable mode: optimizer needed
        optimizer = torch.optim.AdamW(
            learnable_params,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=training_config.betas
        )
        logger.info(f"⚙️  Learnable mode: using optimizer to update {n_params_learnable} parameters")
    
        # Learning rate scheduler (with warmup)
        total_steps = len(train_loader) * training_config.num_epochs
        if training_config.scheduler == "cosine":
            scheduler = WarmupCosineScheduler(
                optimizer, 
                warmup_steps=training_config.warmup_steps,
                total_steps=total_steps,
                min_lr=1e-6
            )
            use_step_scheduler = True  # Update per step
        elif training_config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
            use_step_scheduler = False  # Update per epoch
        else:
            scheduler = None
            use_step_scheduler = False
    
    # TensorBoard
    writer = None
    if training_config.use_tensorboard and HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=training_config.log_dir)
    
    # Training
    logger.info("=" * 80)
    logger.info(f"Start training Grouped RVQ:")
    logger.info(f"  - feature dimension: {rvq_config.feature_dim}")
    logger.info(f"  - Groups: {rvq_config.num_groups} groups × {model.group_dim} dim")
    logger.info(f"  - Layers per group: {rvq_config.num_fine_layers}")
    logger.info(f"  - codebook size: {rvq_config.fine_codebook_size}")
    logger.info(f"  - SKIP: {'enabled' if rvq_config.enable_skip else 'disabled'}")
    logger.info(f"  - Training set: {len(train_loader.dataset)} | Validation set: {len(val_loader.dataset)}")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    best_val_cosine = 0.0  # Early stopping based on reconstruction rate (higher is better)
    global_step = 0
    
    # Early stopping mechanism
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, training_config.num_epochs + 1):
        # If using bucket sampler, update epoch to ensure different shuffle per round
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        
        # ===== Training =====
        model.train()
        total_recon = 0
        total_commit = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{training_config.num_epochs}")
        for batch in pbar:
            features = batch['features'].to(device, non_blocking=True)  # (B, T, 768)
            labels = batch['labels'].to(device, non_blocking=True)       # (B,)
            lengths = batch['lengths'].to(device, non_blocking=True)     # (B,)
            
            # Create valid frame mask (avoid padding pollution)
            B, T, D = features.shape
            t_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
            valid_bt = (t_idx < lengths.unsqueeze(1))  # (B, T) bool, for VQ mask
            mask = valid_bt.float().unsqueeze(-1)  # (B, T, 1) float, for loss calculation
            
            # Forward pass (Stage 1 doesn't use ECVQ, but pass valid_mask to mask padding)
            reconstructed, indices, commit_loss, stats = model(
                features,
                lambda_rate=None,  # Stage 1: don't use ECVQ
                entropy_model=None,
                valid_mask=valid_bt  # Mask padding frames to avoid polluting codebook
            )
            
            # Print shape verification for first batch (fix: move after forward)
            if global_step == 0:
                logger.info(f"✅ Input shape verification: {features.shape}")
                logger.info(f"✅ Sequence length range: {lengths.min().item()}-{lengths.max().item()}")
                logger.info(f"✅ Valid frame ratio: {valid_bt.sum().item()}/{B*T} = {valid_bt.float().mean():.2%}")
                logger.info(f"✅ Commit loss type check: {type(commit_loss)}, value: {float(commit_loss.mean()):.4f}")
                logger.info(f"   (Confirm this is the 'codebook/commitment' term to add to total loss)")
            
            # Masked MSE (avoid padding pollution)
            mse_num = ((reconstructed - features) ** 2 * mask).sum()
            mse_den = (mask.sum() * D).clamp_min(1.0)
            recon_loss = mse_num / mse_den
            
            # Commit loss (scaled by valid frame ratio) - fix: don't divide by D
            valid_frames = mask.squeeze(-1).sum()  # Number of valid frames
            valid_ratio = (valid_frames / (B * T)).detach()  # ∈ (0,1]
            commit_loss_scalar = commit_loss.mean() * valid_ratio
            total_loss = recon_loss + commit_loss_scalar
            
            # Backward pass (skip in EMA mode, codebook auto-updates)
            if optimizer is not None:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
                optimizer.step()
                
                # Learning rate scheduling (per step)
                if scheduler is not None and use_step_scheduler:
                    current_lr = scheduler.step()
                else:
                    current_lr = optimizer.param_groups[0]['lr']
            else:
                # EMA mode: codebook already auto-updated in forward, loss only for monitoring
                current_lr = training_config.learning_rate
            
            total_recon += recon_loss.item()
            total_commit += commit_loss_scalar.item()
            num_batches += 1
            
            pbar.set_postfix({
                'recon': f'{recon_loss.item():.4f}',
                'commit': f'{commit_loss_scalar.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # TensorBoard logging
            if writer and global_step % training_config.log_interval == 0:
                writer.add_scalar('train/recon', recon_loss.item(), global_step)
                writer.add_scalar('train/commit', commit_loss_scalar.item(), global_step)
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                
                # Health monitoring: residual energy curve (should decrease by layer)
                if 'residual_energies' in stats:
                    for layer_idx, energy in enumerate(stats['residual_energies']):
                        writer.add_scalar(f'health/residual_energy_layer_{layer_idx}', energy, global_step)
            
            global_step += 1
        
        avg_recon = total_recon / num_batches
        avg_commit = total_commit / num_batches
        
        logger.info(f"Epoch {epoch} Training - Recon: {avg_recon:.4f}, Commit: {avg_commit:.4f}")
        
        # ===== Validation =====
        if epoch % training_config.eval_interval == 0:
            model.eval()
            total_val_recon = 0
            total_val_cosine_sum = 0  # Changed to weighted sum
            total_valid_frames = 0    # Total valid frames
            num_val_batches = 0
            
            # Codebook utilization accumulator (accumulate by batch, avoid dimension mismatch)
            usage_acc = None
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device, non_blocking=True)
                    lengths = batch['lengths'].to(device, non_blocking=True)
                    
                    # Create valid frame mask
                    B, T, D = features.shape
                    t_idx = torch.arange(T, device=device).unsqueeze(0)
                    valid_bt = (t_idx < lengths.unsqueeze(1))  # (B, T) bool
                    mask = valid_bt.float().unsqueeze(-1)  # (B, T, 1)
                    
                    reconstructed, indices, _, stats = model(features, valid_mask=valid_bt)
                    
                    # Accumulate codebook usage statistics (only valid frames)
                    usage_acc = _accumulate_codebook_usage(
                        usage_acc, indices, valid_bt, 
                        model.skip_token_id, rvq_config.fine_codebook_size
                    )
                    
                    # Masked MSE
                    mse_num = ((reconstructed - features) ** 2 * mask).sum()
                    mse_den = (mask.sum() * D).clamp_min(1.0)
                    val_recon = (mse_num / mse_den).item()
                    total_val_recon += val_recon
                    
                    # Masked Cosine Similarity (weighted by valid frames)
                    mask_flat = mask.reshape(-1)  # (B*T, 1)
                    features_flat = features.reshape(-1, D)
                    reconstructed_flat = reconstructed.reshape(-1, D)
                    
                    # Filter out padding frames
                    valid_idx = mask_flat.squeeze() > 0
                    if valid_idx.any():
                        features_valid = features_flat[valid_idx]
                        reconstructed_valid = reconstructed_flat[valid_idx]
                        # Accumulate: weighted by valid frame count
                        cos_sim_sum = F.cosine_similarity(features_valid, reconstructed_valid, dim=1).sum().item()
                        total_val_cosine_sum += cos_sim_sum
                        total_valid_frames += valid_idx.sum().item()
                    
                    num_val_batches += 1
            
            val_recon = total_val_recon / num_val_batches
            val_cosine = total_val_cosine_sum / max(total_valid_frames, 1)  # Weighted by valid frame count
            val_rmse = (val_recon ** 0.5)
            
            # Restoration rate estimation
            restoration_rate = val_cosine * 100
            
            logger.info(f"Epoch {epoch} Validation:")
            logger.info(f"  - Recon: {val_recon:.4f}, RMSE: {val_rmse:.4f}")
            logger.info(f"  - CosSim: {val_cosine:.4f} → Restoration rate: {restoration_rate:.2f}%")
            
            if writer:
                writer.add_scalar('val/recon', val_recon, epoch)
                writer.add_scalar('val/rmse', val_rmse, epoch)
                writer.add_scalar('val/cosine_similarity', val_cosine, epoch)
                writer.add_scalar('val/restoration_rate', restoration_rate, epoch)
            
            # Health monitoring: codebook utilization (entire validation set, accumulated by batch)
            logger.info("Analyzing codebook utilization (entire validation set)...")
            
            # Generate statistics from accumulator (including perplexity)
            import math
            usage_stats = {}
            perplexities = []
            
            for l, acc in enumerate(usage_acc):
                total = acc['count']
                unique = len(acc['used'])
                util = (unique / rvq_config.fine_codebook_size) if total > 0 else 0.0
                skip_rate = (acc['skips'] / total) if total > 0 else 0.0
                
                # Calculate perplexity (perplexity = 2^entropy)
                if len(acc['histogram']) > 0 and total > 0:
                    # Calculate probability distribution
                    probs = [count / total for count in acc['histogram'].values()]
                    # Calculate entropy (bits)
                    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                    # Perplexity
                    perplexity = 2 ** entropy
                else:
                    perplexity = 0.0
                
                perplexities.append(perplexity)
                
                usage_stats[f'layer_{l}'] = {
                    'utilization': util,
                    'unique_codes': unique,
                    'total_codes': rvq_config.fine_codebook_size,
                    'skip_rate': skip_rate,
                    'perplexity': perplexity,
                }
            
            avg_utilization = sum(s['utilization'] for s in usage_stats.values()) / max(1, len(usage_stats))
            avg_perplexity = sum(perplexities) / max(1, len(perplexities))
            logger.info(f"  - Average utilization: {avg_utilization:.2%}")
            logger.info(f"  - Average perplexity: {avg_perplexity:.1f} / {rvq_config.fine_codebook_size} (ideal value ≈ codebook size)")
            
            # Print details for each layer (all layers, including perplexity)
            num_layers = len(usage_stats)
            logger.info(f"  - Details per layer (all {num_layers} layers):")
            for layer_idx in range(num_layers):
                s = usage_stats[f'layer_{layer_idx}']
                logger.info(f"    Layer{layer_idx:3d}: Util={s['utilization']:5.1%} ({s['unique_codes']:3d}/{s['total_codes']:3d} codes), "
                          f"Perplexity={s['perplexity']:6.1f}, SKIP={s['skip_rate']:5.1%}")
            
            if writer:
                writer.add_scalar('health/avg_codebook_utilization', avg_utilization, epoch)
                writer.add_scalar('health/avg_perplexity', avg_perplexity, epoch)
                for layer_key, layer_stats in usage_stats.items():
                    layer_idx = int(layer_key.split('_')[1])
                    writer.add_scalar(f'health/codebook_util_layer_{layer_idx}', layer_stats['utilization'], epoch)
                    writer.add_scalar(f'health/perplexity_layer_{layer_idx}', layer_stats['perplexity'], epoch)
                    writer.add_scalar(f'health/skip_rate_layer_{layer_idx}', layer_stats['skip_rate'], epoch)
            
            # Early stopping mechanism: based on restoration rate (cosine similarity)
            if rvq_config.enable_early_stopping:
                # Check for significant improvement
                improvement = val_cosine - best_val_cosine
                
                if improvement > rvq_config.early_stopping_min_delta:
                    # Improvement found, save best model and reset patience
                    best_val_cosine = val_cosine
                    best_val_loss = val_recon
                    best_epoch = epoch
                    patience_counter = 0
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                        'val_loss': val_recon,
                        'val_cosine': val_cosine,
                        'rvq_config': {k: v for k, v in rvq_config.__dict__.items() if not k.startswith('_')},
                    }
                    save_path = Path(training_config.checkpoint_dir) / "grouped_rvq_best.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, save_path)
                    logger.info(f"✅ Saved best model: {save_path} (restoration rate improvement: +{improvement*100:.3f}%)")
                else:
                    # No improvement, increase patience counter
                    patience_counter += 1
                    logger.info(f"⏸️  No significant improvement (improvement: {improvement*100:.3f}%, threshold: {rvq_config.early_stopping_min_delta*100:.3f}%)")
                    logger.info(f"   Early stopping: {patience_counter}/{rvq_config.early_stopping_patience}")
                    
                    # Check if should early stop
                    if patience_counter >= rvq_config.early_stopping_patience:
                        logger.info("")
                        logger.info("=" * 80)
                        logger.info(f"🛑 Early Stopping triggered!")
                        logger.info(f"   - Best epoch: {best_epoch}")
                        logger.info(f"   - Best restoration rate: {best_val_cosine*100:.2f}%")
                        logger.info(f"   - No significant improvement for {patience_counter} consecutive epochs")
                        logger.info("=" * 80)
                        break  # Exit training loop early
            else:
                # Not using early stopping, keep original logic (based on val_loss)
                if val_recon < best_val_loss:
                    best_val_loss = val_recon
                    best_val_cosine = val_cosine
                    best_epoch = epoch
                    
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'val_loss': val_recon,
                    'val_cosine': val_cosine,
                    'rvq_config': {k: v for k, v in rvq_config.__dict__.items() if not k.startswith('_')},
                }
                save_path = Path(training_config.checkpoint_dir) / "grouped_rvq_best.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, save_path)
                logger.info(f"✅ Saved best model: {save_path}")
        
        # Learning rate scheduling (per epoch, only for non-step-level scheduler)
        if scheduler is not None and not use_step_scheduler:
            scheduler.step()
        
        # Periodically save checkpoints
        if epoch % training_config.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                'val_loss': val_recon if epoch % training_config.eval_interval == 0 else None,
                'rvq_config': {k: v for k, v in rvq_config.__dict__.items() if not k.startswith('_')},
            }
            save_path = Path(training_config.checkpoint_dir) / f"grouped_rvq_epoch{epoch}.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint: {save_path}")
    
    logger.info("=" * 80)
    logger.info(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 80)
    
    if writer:
        writer.close()
    
    # Save training summary
    summary = {
        'best_val_loss': best_val_loss,
        'total_epochs': training_config.num_epochs,
        'rvq_config': {k: v for k, v in rvq_config.__dict__.items() if not k.startswith('_')},
    }
    summary_path = Path(training_config.results_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved: {summary_path}")


def main():
    """Main function"""
    import os
    
    # Load configuration
    config = get_default_config()
    grouped_rvq_config = config['grouped_rvq']
    data_config = config['data']
    training_config = config['training']
    
    # Test mode: quick validation of flow
    if os.environ.get('TEST_MODE', 'false').lower() == 'true':
        test_epochs = int(os.environ.get('TEST_EPOCHS', '3'))
        logger.info("=" * 80)
        logger.info(f"⚠️  Test mode: Run only {test_epochs} epochs to validate flow")
        logger.info("=" * 80)
        
        # Override configuration
        from dataclasses import replace
        training_config = replace(training_config, num_epochs=test_epochs)
        data_config = replace(data_config, max_samples=1000)  # Only use 1000 samples
        logger.info(f"  - Training epochs: {test_epochs}")
        logger.info(f"  - Max samples: 1000")
    
    # Training
    train_model(grouped_rvq_config, data_config, training_config)


if __name__ == "__main__":
    main()

