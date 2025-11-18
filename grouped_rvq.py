"""
Grouped RVQ + SKIP mechanism implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
import math
import logging

from vector_quantize_pytorch import VectorQuantize

logger = logging.getLogger(__name__)


class GroupedResidualVQ(nn.Module):
    """
    Grouped Residual Vector Quantization + SKIP mechanism
    
    Architecture:
        768-dim → 12 groups × 64-dim
        Each group:
            (Optional) Coarse layer (K=256)
            Fine layers 1-3 (K=128 + SKIP)
    """
    
    def __init__(self, config):
        """
        Args:
            config: GroupedRVQConfig instance
        """
        super().__init__()
        self.config = config
        
        assert config.feature_dim % config.num_groups == 0, \
            f"feature_dim ({config.feature_dim}) must be divisible by num_groups ({config.num_groups})"
        
        # Grouping parameters
        self.feature_dim = config.feature_dim
        self.num_groups = config.num_groups
        self.group_dim = config.feature_dim // config.num_groups
        
        # Fine layers within groups (each group independent, using EMA mode)
        self.fine_vqs = nn.ModuleList()
        for g in range(config.num_groups):
            group_vqs = nn.ModuleList()
            for m in range(config.num_fine_layers):
                vq = VectorQuantize(
                    dim=self.group_dim,
                    codebook_size=config.fine_codebook_size,
                    decay=config.decay,  # EMA mode: codebook auto-updates, no optimizer needed
                    commitment_weight=config.commitment_weight,
                    kmeans_init=config.kmeans_init,
                    kmeans_iters=config.kmeans_iters,
                    threshold_ema_dead_code=config.threshold_ema_dead_code,
                )
                group_vqs.append(vq)
            self.fine_vqs.append(group_vqs)
        
        # SKIP mechanism (fix: use non-negative id)
        self.enable_skip = config.enable_skip
        self.skip_token_id = config.fine_codebook_size  # K as SKIP id (non-negative)
        
        logger.info(f"✅ GroupedResidualVQ initialized:")
        logger.info(f"   - Feature dimension: {self.feature_dim}")
        logger.info(f"   - Groups: {self.num_groups} groups × {self.group_dim} dims")
        logger.info(f"   - Fine layers: {config.num_fine_layers} layers/group × {config.fine_codebook_size} codes")
        logger.info(f"   - SKIP: {'Enabled' if self.enable_skip else 'Disabled'}")
    
    def forward(
        self,
        x: torch.Tensor,
        lambda_rate: Optional[torch.Tensor] = None,
        entropy_model: Optional[nn.Module] = None,
        valid_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_bpf: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass (supports ECVQ + padding masking + real entropy model decision)
        
        Args:
            x: (B, T, feature_dim) input features
            lambda_rate: (scalar) Lagrange multiplier λ (for ECVQ)
            entropy_model: Entropy model q(k|context) (for computing real code length)
            valid_mask: (B, T) bool/byte tensor, marking valid frames (mask padding)
            labels: (B,) emotion labels (for conditional entropy model)
        
        Returns:
            quantized: (B, T, feature_dim) quantized features
            indices: (B, T, num_total_layers) codebook indices (including SKIP)
            commit_loss: (scalar) commitment loss
            stats: dict statistics
        """
        B, T, D = x.shape
        assert D == self.feature_dim
        
        # Handle valid_mask (mask padding frames)
        if valid_mask is None:
            valid_bt = torch.ones(B, T, dtype=torch.bool, device=x.device)
        else:
            valid_bt = valid_mask.to(torch.bool)
        
        # Reshape to grouped form: (B, T, num_groups, group_dim)
        x_grouped = x.view(B, T, self.num_groups, self.group_dim)
        
        # Store results
        all_indices = []        # All layer indices
        all_commit_losses = []  # All layer commitment losses
        reconstructed_grouped = torch.zeros_like(x_grouped)
        
        total_skips = 0
        total_codes = 0
        
        # Health monitoring: residual energy (per layer)
        residual_energies = []  # For diagnosing "whether layer learns effective direction"
        
        # ===== Fine layers within groups (residual quantization + SKIP) =====
        # Pre-allocate intra-frame history buffer (for entropy model decision)
        total_layers = self.num_groups * self.config.num_fine_layers
        all_indices_buffer = torch.full(
            (B, T, total_layers), self.skip_token_id,
            device=x.device, dtype=torch.long
        )
        
        for g in range(self.num_groups):
            # Current group input
            x_g = x_grouped[:, :, g, :]  # (B, T, group_dim)
            
            # Initialize residual
            residual = x_g.clone()
            group_reconstructed = torch.zeros_like(x_g)
            
            # Layer-by-layer quantization
            for m, vq in enumerate(self.fine_vqs[g]):
                # Masked quantization (only update codebook for valid frames)
                quantized, indices, commit_loss = self._quantize_with_mask(vq, residual, valid_bt)
                
                # Current position (intra-frame order: group outer, layer inner)
                current_pos = g * self.config.num_fine_layers + m
                
                # ECVQ decision (using real entropy model)
                if self.enable_skip and lambda_rate is not None and entropy_model is not None:
                    # Get intra-frame history (all tokens before current position)
                    indices_history = all_indices_buffer[:, :, :current_pos]  # (B, T, L')
                    
                    if self.config.use_full_ranking_ecvq:
                        # Use full/Top-N ECVQ reranking (true ECVQ)
                        BT = B * T
                        L_prime = indices_history.shape[-1]
                        
                        # Predict next token probability
                        history_flat = indices_history.reshape(BT, L_prime)
                        labels_expanded = labels.unsqueeze(1).repeat(1, T).reshape(BT) if labels is not None else None
                        probs = entropy_model.predict_next_token_prob(history_flat, labels_expanded)
                        probs = probs.view(B, T, -1)  # (B, T, K+1)
                        
                        # Get codebook matrix
                        E = self._get_codebook_weight(vq).to(residual.device, residual.dtype)
                        
                        # Top-N configuration (optional)
                        topn = getattr(self.config, 'ecvq_topk', None)
                        
                        # Full reranking
                        indices_with_skip, quantized_with_skip = self._ecvq_rerank(
                            residual=residual,
                            probs=probs,
                            E=E,
                            lambda_rate=lambda_rate,
                            valid_bt=valid_bt,
                            topk=topn
                        )
                    else:
                        # Use original Top-1+SKIP method (backward compatible)
                        vq_codebook = self._get_codebook_weight(vq)
                        indices_with_skip, quantized_with_skip = self._ecvq_decision_with_entropy(
                            residual=residual,
                            quantized=quantized,
                            indices=indices,
                            entropy_model=entropy_model,
                            indices_history=indices_history,
                            labels=labels,
                            lambda_rate=lambda_rate,
                            valid_bt=valid_bt,
                            codebook=vq_codebook,
                            use_full_ranking=False
                        )
                    
                    skip_mask = (indices_with_skip == self.skip_token_id) & valid_bt
                    total_skips += skip_mask.sum().item()
                    total_codes += valid_bt.sum().item()
                    
                elif self.enable_skip and (target_bpf is not None):
                    # Quantile gating: control SKIP rate based on target bpf (friend's approach)
                    max_bpf = self.num_groups * self.config.num_fine_layers * math.log2(self.config.fine_codebook_size)
                    target_skip = float(1.0 - max(0.0, min(target_bpf, max_bpf)) / max_bpf)
                    
                    # ΔD normalization
                    distortion_send = torch.sum((residual - quantized) ** 2, dim=-1)  # (B, T)
                    distortion_skip = torch.sum(residual ** 2, dim=-1)  # (B, T)
                    denom = (distortion_skip + 1e-8)
                    delta = (distortion_skip - distortion_send) / denom  # (B, T), in [0,1]
                    
                    # Only compute quantile threshold τ on valid frames
                    valid_delta = delta[valid_bt]
                    if bool(valid_delta.numel() > 0):
                        tau = torch.quantile(valid_delta, q=target_skip)  # scalar
                        skip_mask = (delta <= tau) & valid_bt
                    else:
                        skip_mask = torch.zeros_like(valid_bt)
                    
                    indices_with_skip = indices.clone()
                    indices_with_skip[skip_mask] = self.skip_token_id
                    quantized_with_skip = quantized.clone()
                    quantized_with_skip[skip_mask] = 0.0
                    
                    total_skips += skip_mask.sum().item()
                    total_codes += valid_bt.sum().item()
                    
                elif self.enable_skip and lambda_rate is not None:
                    # Without entropy model: simplified ECVQ (only distortion, no bits)
                    distortion_skip = torch.sum(residual ** 2, dim=-1)  # (B, T)
                    distortion_send = torch.sum((residual - quantized) ** 2, dim=-1)  # (B, T)
                    
                    # Unscaled ratio decision (more robust)
                    skip_mask = ((distortion_skip - distortion_send) / (distortion_skip + 1e-8) < lambda_rate * 1e-2) & valid_bt
                    
                    indices_with_skip = indices.clone()
                    indices_with_skip[skip_mask] = self.skip_token_id
                    quantized_with_skip = quantized.clone()
                    quantized_with_skip[skip_mask] = 0.0
                    
                    total_skips += skip_mask.sum().item()
                    total_codes += valid_bt.sum().item()
                else:
                    # Standard RVQ (no SKIP)
                    indices_with_skip = indices
                    quantized_with_skip = quantized
                    total_codes += valid_bt.sum().item()
                
                # Update history buffer
                all_indices_buffer[:, :, current_pos] = indices_with_skip
                
                # Update reconstruction and residual
                all_indices.append(indices_with_skip)
                group_reconstructed += quantized_with_skip
                residual = residual - quantized_with_skip
                
                all_commit_losses.append(commit_loss)
                
                # Record residual energy (health monitoring, only count valid frames)
                if bool(valid_bt.any().item()):
                    residual_energy = (residual[valid_bt] ** 2).mean().item()
                else:
                    residual_energy = 0.0
                residual_energies.append(residual_energy)
            
            # Store reconstruction for this group
            reconstructed_grouped[:, :, g, :] = group_reconstructed
        
        # ===== 3. Merge results =====
        # Reshape back to original shape
        quantized = reconstructed_grouped.view(B, T, self.feature_dim)
        
        # Merge indices: (B, T, num_groups * num_fine_layers)
        indices = torch.stack(all_indices, dim=-1)  # (B, T, num_total_layers)
        
        # Merge losses
        commit_loss = torch.stack(all_commit_losses).mean()
        
        # Statistics
        skip_rate = total_skips / max(total_codes, 1) if total_codes > 0 else 0.0
        stats = {
            'skip_rate': skip_rate,
            'total_skips': total_skips,
            'total_codes': total_codes,
            'num_layers': len(all_indices),
            'residual_energies': residual_energies,  # Residual energy per layer (health monitoring)
        }
        
        return quantized, indices, commit_loss, stats
    
    def encode(
        self,
        x: torch.Tensor,
        lambda_rate: float = 1.0,
        entropy_model: Optional[nn.Module] = None,
        valid_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Encode (get indices and bitrate information)
        
        Args:
            x: (B, T, feature_dim) input features
            lambda_rate: Lagrange multiplier
            entropy_model: Entropy model
            valid_mask: (B, T) bool tensor, marking valid frames
            labels: (B,) emotion labels (for conditional entropy model)
        
        Returns:
            indices: (B, T, num_total_layers) codebook indices
            info: dict containing bitrate etc information
        """
        lambda_tensor = torch.tensor(lambda_rate, device=x.device, dtype=x.dtype)
        _, indices, _, stats = self.forward(x, lambda_tensor, entropy_model, valid_mask, labels)
        
        # Compute bitrate (if entropy model available)
        if entropy_model is not None:
            bits = entropy_model.compute_bits(indices, labels=None, valid_mask=valid_mask)
            if valid_mask is not None:
                valid_frames = valid_mask.sum().item()
            else:
                B, T = indices.shape[:2]
                valid_frames = B * T
            bits_per_frame = bits.item() / max(valid_frames, 1)
        else:
            # Theoretical upper bound (uniform distribution)
            import math
            bits_per_layer = math.log2(self.config.fine_codebook_size)
            bits_per_frame = bits_per_layer * stats['num_layers'] * (1 - stats['skip_rate'])
        
        info = {
            'indices': indices,
            'bits_per_frame': float(bits_per_frame),
            'skip_rate': stats['skip_rate'],
        }
        
        return indices, info
    
    def _quantize_with_mask(
        self, 
        vq: nn.Module, 
        residual: torch.Tensor, 
        valid_bt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masked quantization (only update codebook for valid frames, avoid padding contamination)
        
        Args:
            vq: VectorQuantize module
            residual: (B, T, group_dim) residual
            valid_bt: (B, T) bool mask, marking valid frames
        
        Returns:
            quantized: (B, T, group_dim) quantized vector
            indices: (B, T) codebook indices
            commit_loss: commitment loss
        """
        B, T, Dg = residual.shape
        
        # If all valid, directly call VQ
        if bool(valid_bt.all().item()):
            quantized, indices, commit_loss = vq(residual)
            return quantized, indices, commit_loss
        
        # If all invalid, return zeros
        if not bool(valid_bt.any().item()):
            quantized = torch.zeros_like(residual)
            indices = torch.full((B, T), self.skip_token_id, 
                                device=residual.device, dtype=torch.long)
            commit_loss = torch.tensor(0.0, device=residual.device)
            return quantized, indices, commit_loss
        
        # Partially valid: only quantize valid frames
        flat_in = residual[valid_bt]  # (N_valid, Dg)
        q_flat, idx_flat, commit = vq(flat_in.unsqueeze(0))  # Add batch dimension
        q_flat = q_flat.squeeze(0)
        idx_flat = idx_flat.squeeze(0)
        
        # Scatter back to (B, T, Dg) / (B, T)
        quantized = torch.zeros_like(residual)
        quantized[valid_bt] = q_flat
        
        indices = torch.full((B, T), self.skip_token_id, 
                            device=residual.device, dtype=torch.long)
        indices[valid_bt] = idx_flat
        
        return quantized, indices, commit
    
    def _get_codebook_weight(self, vq: nn.Module) -> torch.Tensor:
        """
        Return VQ layer codebook matrix E: (K, Dg)
        Compatible with different vector_quantize_pytorch version field naming
        """
        # Try common codebook field names
        candidates = []
        
        # 1. Direct fields
        for name in ['codebook', 'embedding', 'embeddings', 'embed']:
            if hasattr(vq, name):
                obj = getattr(vq, name)
                if hasattr(obj, 'weight'):
                    candidates.append(obj.weight)
                elif torch.is_tensor(obj):
                    candidates.append(obj)
        
        # 2. _codebook sub-object (current version)
        if hasattr(vq, '_codebook'):
            cb = getattr(vq, '_codebook')
            for name in ['embed', 'weight', 'embedding']:
                if hasattr(cb, name):
                    w = getattr(cb, name)
                    if torch.is_tensor(w):
                        candidates.append(w)
        
        # 3. EMA version may be under ema_codebook
        if hasattr(vq, 'ema_codebook'):
            ema = getattr(vq, 'ema_codebook')
            for name in ['embedding', 'embed', 'weight']:
                if hasattr(ema, name):
                    w = getattr(ema, name)
                    if hasattr(w, 'weight'):
                        candidates.append(w.weight)
                    elif torch.is_tensor(w):
                        candidates.append(w)
        
        # Return first found tensor
        for w in candidates:
            if torch.is_tensor(w):
                # Handle multiple codebooks case: (num_codebooks, K, D) → take first codebook → (K, D)
                if w.ndim == 3 and w.size(0) == 1:
                    return w[0]  # (K, D)
                elif w.ndim == 2:
                    return w  # (K, D)
        
        raise RuntimeError(f"Unable to locate VQ codebook weight, tried fields: {list(vars(vq).keys())}")
    
    @torch.inference_mode()
    def _ecvq_rerank(
        self,
        residual: torch.Tensor,          # (B, T, Dg)
        probs: torch.Tensor,             # (B, T, K+1) from q(k|history), last column is SKIP
        E: torch.Tensor,                 # (K, Dg) codebook matrix
        lambda_rate: torch.Tensor,       # scalar
        valid_bt: torch.Tensor,          # (B, T) valid frame mask
        topk: Optional[int] = None       # if given, first take Euclidean distance Top-N
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full/Top-N ECVQ reranking: choose rate-distortion optimal among all codewords + SKIP
        
        True ECVQ decision: J(k) = ΔD(k) + λ·(-log₂ q(k|ctx))
        """
        B, T, Dg = residual.shape
        K = E.size(0)
        
        # Compute all codeword ΔD(k) = ||e_k||² - 2·r·e_k (ignore ||r||², doesn't affect argmin)
        r = residual.reshape(B * T, Dg)                      # (BT, Dg)
        E_norm2 = (E ** 2).sum(dim=-1)                       # (K,)
        
        # Distance matrix (vectorized computation)
        D_full = E_norm2.unsqueeze(0) - 2.0 * (r @ E.t())   # (BT, K)
        D_full = D_full.view(B, T, K)                        # (B, T, K)
        
        if topk is not None and topk < K:
            # Top-N tradeoff: first select Top-N by Euclidean distance, then do rate-distortion reranking
            _, idx_topk = torch.topk(-D_full, k=topk, dim=-1)  # negative sign: smaller distance preferred
            
            # Extract corresponding candidates from probs
            p_codes = probs[..., :K].gather(-1, idx_topk).clamp_min(1e-12)  # (B, T, N)
            bits_codes = -torch.log2(p_codes)
            
            # Candidate ΔD
            D_codes = D_full.gather(-1, idx_topk)
            
            # Combined J(k) = D(k) + λ·bits(k)
            J_codes = D_codes + lambda_rate * bits_codes                     # (B, T, N)
            
            # SKIP cost
            bits_skip = -torch.log2(probs[..., K].clamp_min(1e-12))          # (B, T)
            J_skip = bits_skip * lambda_rate                                  # (B, T)
            
            # Merge SKIP
            J_all = torch.cat([J_codes, J_skip.unsqueeze(-1)], dim=-1)       # (B, T, N+1)
            k_star_local = J_all.argmin(dim=-1)                              # (B, T)
            
            # Map local Top-N indices back to global 0..K mapping
            chosen_is_skip = (k_star_local == topk)
            idx_global = torch.zeros_like(k_star_local)
            
            # Use advanced indexing for batch restoration
            mask_send = ~chosen_is_skip
            if mask_send.any():
                b_idx, t_idx = torch.where(mask_send)
                idx_global[b_idx, t_idx] = idx_topk[b_idx, t_idx, k_star_local[b_idx, t_idx]]
            idx_global[chosen_is_skip] = K  # use K to represent SKIP
            
        else:
            # Full reranking
            bits_codes = -torch.log2(probs[..., :K].clamp_min(1e-12))        # (B, T, K)
            J_codes = D_full + lambda_rate * bits_codes                       # (B, T, K)
            
            # SKIP cost (distortion=0)
            bits_skip = -torch.log2(probs[..., K].clamp_min(1e-12))           # (B, T)
            J_skip = bits_skip * lambda_rate                                   # (B, T)
            
            # Concatenate and select optimal
            J_all = torch.cat([J_codes, J_skip.unsqueeze(-1)], dim=-1)        # (B, T, K+1)
            idx_global = J_all.argmin(dim=-1)                                  # (B, T); 0..K, K=SKIP
        
        # Write back indices and quantized vectors
        chosen_is_skip = (idx_global == K)
        
        indices_final = idx_global.clone()                                     # (B, T)
        indices_final[chosen_is_skip] = self.skip_token_id                     # mark as SKIP
        
        quantized_final = torch.zeros_like(residual)                           # (B, T, Dg)
        
        # Vectorized construction of quantized vectors
        if (~chosen_is_skip).any():
            mask_send = ~chosen_is_skip
            quantized_final[mask_send] = E[idx_global[mask_send]]
        
        # Mask invalid frames
        if not valid_bt.all():
            quantized_final[~valid_bt] = 0.0
            indices_final[~valid_bt] = self.skip_token_id
        
        return indices_final, quantized_final
    
    @torch.inference_mode()
    def _ecvq_decision_with_entropy(
        self,
        residual: torch.Tensor,
        quantized: torch.Tensor,
        indices: torch.Tensor,
        entropy_model: nn.Module,
        indices_history: torch.Tensor,
        labels: Optional[torch.Tensor],
        lambda_rate: torch.Tensor,
        valid_bt: torch.Tensor,
        codebook: torch.Tensor,  # new: codebook (K, Dg)
        use_full_ranking: bool = True  # new: whether to use full reranking
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ECVQ decision (using real entropy model)
        
        Supports two modes:
        1. use_full_ranking=True: Full reranking (true ECVQ)
        2. use_full_ranking=False: Top-1+SKIP approximation (original method)
        
        Args:
            residual: (B, T, Dg) current residual
            quantized: (B, T, Dg) VQ quantized vector
            indices: (B, T) VQ returned codebook indices
            entropy_model: trained entropy model
            indices_history: (B, T, L') intra-frame decided index history
            labels: (B,) emotion labels (needed for conditional model)
            lambda_rate: Lagrange multiplier λ
            valid_bt: (B, T) bool mask
            codebook: (K, Dg) current layer codebook
            use_full_ranking: whether to use full reranking
        
        Returns:
            final_indices: (B, T) final decision (including SKIP)
            final_quantized: (B, T, Dg) final quantized vector
        """
        B, T, Dg = residual.shape
        K = codebook.shape[0]  # codebook size
        
        # Use entropy model to predict next token probability distribution
        BT = B * T
        L_prime = indices_history.shape[-1]
        
        # Construct flat history (allow L'=0, i.e., BOS-only prior)
        history_flat = indices_history.reshape(BT, L_prime)  # (B*T, L')
        
        # If conditional model, expand labels
        if entropy_model.condition_on_label and labels is not None:
            labels_expanded = labels.unsqueeze(1).repeat(1, T).reshape(BT)  # (B*T,)
        else:
            labels_expanded = None
        
        # Use entropy model to predict (L'==0 means BOS-only prior, model learns distribution itself)
        probs = entropy_model.predict_next_token_prob(history_flat, labels_expanded)
        probs = probs.view(B, T, -1)  # (B, T, V)
        
        if use_full_ranking:
            # ===== Full reranking (true ECVQ) =====
            # 1) Compute all codeword distortions
            r_flat = residual.view(BT, Dg)  # (BT, Dg)
            
            # ||e_k||^2: (K,)
            codebook_norm2 = (codebook ** 2).sum(dim=-1)  # (K,)
            
            # ||r - e_k||^2 = ||r||^2 + ||e_k||^2 - 2*r·e_k
            r_norm2 = (r_flat ** 2).sum(dim=-1, keepdim=True)  # (BT, 1)
            inner_prod = r_flat @ codebook.T  # (BT, K)
            D_all = r_norm2 + codebook_norm2.unsqueeze(0) - 2 * inner_prod  # (BT, K)
            D_all = D_all.view(B, T, K)  # (B, T, K)
            
            # SKIP distortion (don't send any codeword)
            D_skip = (residual ** 2).sum(dim=-1)  # (B, T)
            
            # 2) Compute all candidate bits
            bits_codes = -torch.log2(probs[:, :, :K].clamp_min(1e-12))  # (B, T, K)
            bits_skip = -torch.log2(probs[:, :, self.skip_token_id].clamp_min(1e-12))  # (B, T)
            
            # 3) Compute rate-distortion cost J(k) = D(k) + λ·bits(k)
            J_codes = D_all + lambda_rate.unsqueeze(-1) * bits_codes  # (B, T, K)
            J_skip = D_skip + lambda_rate * bits_skip  # (B, T)
            
            # 4) Concatenate SKIP and select optimal
            J_all = torch.cat([J_codes, J_skip.unsqueeze(-1)], dim=-1)  # (B, T, K+1)
            k_star = J_all.argmin(dim=-1)  # (B, T)
            
            # 5) Apply decision
            chosen_is_skip = (k_star == K)
            
            final_indices = k_star.clone()
            final_indices[chosen_is_skip] = self.skip_token_id
            
            # Construct quantized vector
            final_quantized = torch.zeros_like(residual)
            for b in range(B):
                for t in range(T):
                    if not chosen_is_skip[b, t]:
                        final_quantized[b, t] = codebook[k_star[b, t]]
            
        else:
            # ===== Top-1+SKIP approximation (original method) =====
            # Compute distortion
            distortion_send = ((residual - quantized) ** 2).sum(dim=-1)  # (B, T)
            distortion_skip = (residual ** 2).sum(dim=-1)  # (B, T)
            delta_distortion = distortion_skip - distortion_send  # (B, T)
            
            # Scale normalization
            energy = residual.pow(2).mean(dim=-1).clamp_min(1e-6)  # (B, T)
            delta_distortion = delta_distortion / energy
        
        # Compute bits_send (send codeword k)
        indices_clamped = indices.clamp(0, probs.shape[-1] - 1)
        probs_send = probs.gather(-1, indices_clamped.unsqueeze(-1)).squeeze(-1)  # (B, T)
        probs_send = probs_send.clamp_min(1e-12)
        bits_send = -torch.log2(probs_send)  # (B, T)
        
        # Compute bits_skip (skip)
        probs_skip = probs[:, :, self.skip_token_id].clamp_min(1e-12)  # (B, T)
        bits_skip = -torch.log2(probs_skip)  # (B, T)
        
        # Δbits = bits_send - bits_skip
        delta_bits = bits_send - bits_skip  # (B, T)
        
        # ECVQ decision: choose SKIP when ΔD < λ·Δbits (benefit insufficient)
        should_skip = (delta_distortion < lambda_rate * delta_bits)
        should_skip = should_skip & valid_bt  # only decide on valid frames
        
        # Apply decision
        final_indices = indices.clone()
        final_indices[should_skip] = self.skip_token_id
        
        final_quantized = quantized.clone()
        final_quantized[should_skip] = 0.0
        
        return final_indices, final_quantized
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode (reconstruct features from indices)
        
        Args:
            indices: (B, T, num_total_layers) codebook indices
        
        Returns:
            reconstructed: (B, T, feature_dim) reconstructed features
        
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "GroupedResidualVQ.decode not yet implemented (needs layer-by-layer codebook lookup and residual accumulation).\n"
            "Training phase doesn't need decode, raising error here to avoid misuse.\n"
            "Implementation needs: look up embedding vectors from VQ codebook, accumulate residuals layer by layer."
        )
    
    def get_codebook_usage(self, indices: torch.Tensor) -> dict:
        """
        Calculate codebook utilization
        
        Args:
            indices: (B, T, num_layers) codebook indices
        
        Returns:
            usage_stats: dict utilization statistics per layer
        """
        B, T, num_layers = indices.shape
        usage_stats = {}
        
        for layer in range(num_layers):
            layer_indices = indices[:, :, layer].reshape(-1).cpu()  # move to CPU for processing
            
            # Exclude SKIP (unified as K)
            non_skip = layer_indices[layer_indices != self.skip_token_id]
            
            if non_skip.numel() > 0:
                unique_codes = torch.unique(non_skip)
                utilization = unique_codes.numel() / self.config.fine_codebook_size
            else:
                utilization = 0.0
            
            usage_stats[f'layer_{layer}'] = {
                'utilization': float(utilization),
                'unique_codes': int(unique_codes.numel()) if non_skip.numel() > 0 else 0,
                'total_codes': self.config.fine_codebook_size,
                'skip_rate': float((layer_indices == self.skip_token_id).sum().item()) / max(layer_indices.numel(), 1),
            }
        
        return usage_stats


def create_grouped_rvq(config) -> GroupedResidualVQ:
    """
    Create grouped RVQ model
    
    Args:
        config: GroupedRVQConfig instance
    
    Returns:
        model instance
    """
    return GroupedResidualVQ(config)


