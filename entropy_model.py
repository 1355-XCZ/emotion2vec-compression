"""
Unconditional Autoregressive Entropy Model q(z) - Release Version
Only removed conditional label-related code, all other scientific logic fully preserved

Core Features fully preserved:
1. Right-shift input (avoid self-leakage)
2. Intra-frame AR (sequence length = G×M, not T*G*M)
3. SKIP non-negative (id = K)
4. Add group/layer embedding
5. predict_next_token_prob (ECVQ decision)
6. All computation and training methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (intra-frame position)"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) where L = intra-frame sequence length
        Returns:
            (B, L, D)
        """
        return x + self.pe[:x.size(1), :]


class AutoregressiveEntropyModel(nn.Module):
    """
    Unconditional Autoregressive Entropy Model (intra-frame AR) - only q(z)
    
    Fix points:
    1. Sequence = (G×M) tokens per frame (not cross-frame)
    2. Input right-shifted by one position, avoid self-leakage
    3. SKIP id = K (non-negative)
    4. Add group/layer position embedding
    """
    
    def __init__(self, config, rvq_config):
        """
        Args:
            config: EntropyModelConfig instance
            rvq_config: GroupedRVQConfig instance
        """
        super().__init__()
        self.config = config
        self.rvq_config = rvq_config
        
        # Vocabulary size
        self.K = rvq_config.fine_codebook_size  # codebook size
        self.SKIP_ID = self.K                   # SKIP uses K (non-negative)
        self.V = self.K + 1                     # vocabulary = K codes + 1 SKIP
        self.BOS_ID = self.V                    # BOS (input only)
        
        # Intra-frame sequence length
        self.num_groups = rvq_config.num_groups
        self.num_layers_per_group = rvq_config.num_fine_layers
        self.L = self.num_groups * self.num_layers_per_group  # intra-frame length
        
        # Token Embedding (including BOS)
        self.token_embedding = nn.Embedding(
            num_embeddings=self.V + 1,  # V output classes + 1 BOS
            embedding_dim=config.d_model
        )
        
        # Positional encoding (intra-frame)
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=self.L)
        
        # Group/Layer embedding (let model know "which group/layer this is")
        self.group_embedding = nn.Embedding(self.num_groups, config.d_model)
        self.layer_embedding = nn.Embedding(self.num_layers_per_group, config.d_model)
        
        # Pre-compute group_ids and layer_ids (fixed order)
        self.register_buffer('group_ids', self._create_group_ids())
        self.register_buffer('layer_ids', self._create_layer_ids())
        
        # Transformer encoder (causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output layer: predict next token (output classes = V, excluding BOS)
        self.output_proj = nn.Linear(config.d_model, self.V)
        
        # Unconditional model marker
        self.condition_on_label = False
        
        # Cache causal mask (constant intra-frame length L, avoid repeated construction and transfer)
        self.register_buffer(
            "causal_mask_buf",
            self.generate_causal_mask(self.L, device=torch.device("cpu"))
        )
        
        # Mask cache (ChatGPT suggestion - avoid rebuilding each time)
        self._mask_cache = {}

        logger.info(f"✅ AutoregressiveEntropyModel initialized (unconditional version - only q(z)):")
        logger.info(f"   - Vocabulary: V={self.V} (K={self.K} + SKIP={self.SKIP_ID})")
        logger.info(f"   - Intra-frame sequence length: L={self.L} ({self.num_groups} groups × {self.num_layers_per_group} layers)")
        logger.info(f"   - BOS id: {self.BOS_ID}")
        logger.info(f"   - Causal mask cache: registered as buffer")
    
    def _create_group_ids(self) -> torch.Tensor:
        """
        Create fixed group_ids sequence
        Order: g outer, m inner
        Example G=2, M=3: [0,0,0, 1,1,1]
        """
        ids = []
        for g in range(self.num_groups):
            for m in range(self.num_layers_per_group):
                ids.append(g)
        return torch.tensor(ids, dtype=torch.long)
    
    def _create_layer_ids(self) -> torch.Tensor:
        """
        Create fixed layer_ids sequence
        Order: g outer, m inner
        Example G=2, M=3: [0,1,2, 0,1,2]
        """
        ids = []
        for g in range(self.num_groups):
            for m in range(self.num_layers_per_group):
                ids.append(m)
        return torch.tensor(ids, dtype=torch.long)
    
    def forward(
        self,
        indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None  # Keep parameter interface compatibility, but not using
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (for training, teacher forcing)
        
        Args:
            indices: (B, T, L) codebook index sequence
                     where L = num_groups * num_layers_per_group
            labels: (B,) keep parameter compatibility, unconditional model not using
        
        Returns:
            logits: (B*T, L, V) predicted distribution
            targets: (B*T, L) target indices
        """
        B, T, L = indices.shape
        assert L == self.L, f"Expected L={self.L}, got L={L}"
        
        # Flatten as intra-frame sequence: (B*T, L)
        seq = indices.view(B * T, L)
        
        # Unified SKIP encoding: -1 → K
        seq = torch.where(seq < 0, torch.full_like(seq, self.SKIP_ID), seq)
        
        # Input = right-shifted by one position (position 0 uses BOS)
        inp = torch.roll(seq, shifts=1, dims=1)
        inp[:, 0] = self.BOS_ID
        
        # Token Embedding
        x = self.token_embedding(inp)  # (B*T, L, d_model)
        
        # Add positional encoding (intra-frame)
        x = self.pos_encoding(x)
        
        # Add group/layer embedding (let model know position)
        x = x + self.group_embedding(self.group_ids).unsqueeze(0)  # broadcast (1, L, d) → (B*T, L, d)
        x = x + self.layer_embedding(self.layer_ids).unsqueeze(0)
        
        # Causal mask (intra-frame, use cache)
        causal_mask = self.causal_mask_buf.to(x.device, non_blocking=True)
        
        # Transformer
        x = self.transformer(x, mask=causal_mask)  # (B*T, L, d_model)
        
        # Predict next token
        logits = self.output_proj(x)  # (B*T, L, V)
        
        # Target: original sequence (not right-shifted)
        targets = seq  # (B*T, L)
        
        return logits, targets
    
    def compute_nll(
        self,
        indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,  # Keep parameter compatibility
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute negative log likelihood -log q(z) (support padding masking)
        
        Args:
            indices: (B, T, L) codebook indices
            labels: (B,) keep parameter compatibility, unconditional model not using
            valid_mask: (B, T) bool tensor, marking valid frames (mask padding)
        
        Returns:
            nll_per_sample: (B,) NLL per sample (bits, only count valid frames)
        """
        B, T, L = indices.shape
        
        # Forward pass
        logits, targets = self.forward(indices, labels)  # logits: (B*T, L, V), targets: (B*T, L)
        
        # Compute cross entropy (each position)
        logits_flat = logits.reshape(-1, self.V)  # (B*T*L, V)
        targets_flat = targets.reshape(-1)  # (B*T*L,)
        
        ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*T*L,)
        ce = ce.view(B, T, L)  # (B, T, L)
        
        # Convert to bits (log₂)
        nll_bits = ce / math.log(2)  # (B, T, L)
        
        # Apply valid_mask (only count valid frames)
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1).float()  # (B, T, 1)
            nll_bits = nll_bits * mask  # (B, T, L)
        
        # Sum (per sample, only count valid frames)
        nll_per_sample = nll_bits.sum(dim=(1, 2))  # (B,)
        
        return nll_per_sample
    
    def compute_bits(
        self,
        indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,  # Keep parameter compatibility
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total bits required for encoding (whole batch, only count valid frames)
        
        Args:
            indices: (B, T, L) codebook indices
            labels: (B,) keep parameter compatibility, unconditional model not using
            valid_mask: (B, T) bool tensor, marking valid frames
        
        Returns:
            total_bits: scalar tensor
        """
        nll_per_sample = self.compute_nll(indices, labels, valid_mask)
        return nll_per_sample.sum()
    
    def compute_rate_bps(
        self,
        indices: torch.Tensor,
        frame_rate_hz: float = 50.0,
        labels: Optional[torch.Tensor] = None,  # Keep parameter compatibility
        valid_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute bitrate (bits per second, only count valid frames)
        
        Args:
            indices: (B, T, L) codebook indices
            frame_rate_hz: frame rate (Hz)
            labels: (B,) keep parameter compatibility, unconditional model not using
            valid_mask: (B, T) bool tensor, marking valid frames
        
        Returns:
            rate_bps: bitrate (bps)
        """
        B, T, L = indices.shape
        
        total_bits = self.compute_bits(indices, labels, valid_mask).item()
        
        # Compute valid duration (only count valid frames)
        if valid_mask is not None:
            total_valid_frames = valid_mask.sum().item()
        else:
            total_valid_frames = B * T
        
        total_time_sec = total_valid_frames / frame_rate_hz
        
        rate_bps = total_bits / total_time_sec if total_time_sec > 0 else 0.0
        return rate_bps
    
    def predict_next_token_prob(
        self,
        indices_history: torch.Tensor,
        labels: Optional[torch.Tensor] = None  # Keep parameter compatibility
    ) -> torch.Tensor:
        """
        Predict next token probability distribution (for ECVQ decision)
        
        Args:
            indices_history: (B, L') existing index sequence (L' < L)
            labels: (B,) keep parameter compatibility, unconditional model not using
        
        Returns:
            probs: (B, V) next token probability distribution
        """
        # ChatGPT suggestion: inference mode wraps entire function
        with torch.inference_mode():
            B, L_prime = indices_history.shape
            
            # Unified SKIP
            seq = torch.where(indices_history < 0,
                             torch.full_like(indices_history, self.SKIP_ID),
                             indices_history)
            
            # Add BOS
            inp = torch.cat([torch.full((B, 1), self.BOS_ID, device=seq.device, dtype=seq.dtype),
                            seq], dim=1)  # (B, L'+1)
            
            # Embedding
            x = self.token_embedding(inp)
            x = self.pos_encoding(x)
            
            # Group/layer embedding (only to L'+1)
            x = x + self.group_embedding(self.group_ids[:L_prime+1]).unsqueeze(0)
            x = x + self.layer_embedding(self.layer_ids[:L_prime+1]).unsqueeze(0)
            
            # Transformer (causal mask)
            # ChatGPT suggestion: use cached mask (compatible with old checkpoint)
            if hasattr(self, '_get_causal_mask'):
                mask = self._get_causal_mask(L_prime + 1, x.device)
            else:
                mask = self.generate_causal_mask(L_prime + 1, x.device)
            x = self.transformer(x, mask=mask)
            
            # Only take last position
            logits = self.output_proj(x[:, -1, :])  # (B, V)
            probs = F.softmax(logits, dim=-1)
            
            return probs
    
    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask (upper triangle, float additive mask)
        
        Args:
            seq_len: sequence length
            device: device
        
        Returns:
            mask: (seq_len, seq_len) float tensor, -inf means masked
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, 0.0)
        return mask  # (L, L) float, more compatible with different PyTorch versions


def create_entropy_model(config, rvq_config):
    """
    Create unconditional entropy model q(z)
    
    Args:
        config: EntropyModelConfig instance
        rvq_config: GroupedRVQConfig instance
    
    Returns:
        model instance
    """
    logger.info("Creating unconditional entropy model q(z)")
    return AutoregressiveEntropyModel(config, rvq_config)


# ============================================================================
# Training utilities
# ============================================================================

def train_entropy_model_step(
    model: AutoregressiveEntropyModel,
    indices: torch.Tensor,
    labels: Optional[torch.Tensor],
    optimizer: torch.optim.Optimizer
) -> dict:
    """
    Single step entropy model training
    
    Args:
        model: entropy model
        indices: (B, T, L) codebook indices
        labels: (B,) keep parameter compatibility, unconditional model not using
        optimizer: optimizer
    
    Returns:
        metrics: dict training metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits, targets = model(indices, labels)
    
    # Compute loss
    logits_flat = logits.reshape(-1, model.V)
    targets_flat = targets.reshape(-1)
    
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.max_grad_norm)
    optimizer.step()
    
    # Compute accuracy
    with torch.inference_mode():
        preds = logits_flat.argmax(dim=-1)
        acc = (preds == targets_flat).float().mean()
    
    metrics = {
        'loss': loss.item(),
        'acc': acc.item(),
        'bpf': loss.item() / math.log(2),  # bits per frame (approximate)
    }

    return metrics
