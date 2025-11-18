"""
Rate Controller
Implements dual update and binary search to reach target bitrate
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RateController:
    """
    Rate Controller
    
    Controls bitrate by adjusting Lagrange multiplier λ:
        - Training phase: Dual update λ ← [λ + η(R - r_target)]₊
        - Inference phase: Binary search to find optimal λ
    """
    
    def __init__(self, config):
        """
        Args:
            config: RateControlConfig instance
        """
        self.config = config
        
        # Current λ value
        self.lambda_current = config.lambda_init
        
        # Dual update parameters
        self.dual_lr = config.dual_lr
        self.dual_momentum = config.dual_momentum
        self.lambda_velocity = 0.0  # Momentum term
        
        # Binary search range
        self.lambda_min = config.lambda_min
        self.lambda_max = config.lambda_max
        
        logger.info(f"✅ RateController initialized:")
        logger.info(f"   - λ initial value: {self.lambda_current:.4f}")
        logger.info(f"   - λ range: [{self.lambda_min:.4f}, {self.lambda_max:.4f}]")
        logger.info(f"   - Dual learning rate: {self.dual_lr}")
    
    def dual_update(self, current_rate_bpf: float, target_rate_bpf: float) -> float:
        """
        Dual update for λ (used during training)
        
        λ ← [λ + η(R_current - R_target)]₊
        
        Args:
            current_rate_bpf: Current bitrate (bits/frame)
            target_rate_bpf: Target bitrate (bits/frame)
        
        Returns:
            new_lambda: Updated λ
        """
        # Calculate gradient (rate error)
        grad = current_rate_bpf - target_rate_bpf
        
        # Momentum update
        self.lambda_velocity = (
            self.dual_momentum * self.lambda_velocity +
            (1 - self.dual_momentum) * grad
        )
        
        # Update λ
        self.lambda_current = max(
            self.lambda_min,
            self.lambda_current + self.dual_lr * self.lambda_velocity
        )
        
        logger.debug(f"Dual update: R={current_rate_bpf:.2f} (target={target_rate_bpf:.2f}), "
                     f"λ={self.lambda_current:.4f}")
        
        return self.lambda_current
    
    def binary_search(
        self,
        encoder_fn,
        target_rate_bpf: float,
        max_iters: Optional[int] = None,
        tolerance_bpf: Optional[float] = None,
        use_cache: bool = True,
        lambda_hint: Optional[float] = None  # Prior λ value (from first sample)
    ) -> Tuple[float, float]:
        """
        Binary search to find λ that reaches target bitrate (inference phase, supports caching)
        
        Args:
            encoder_fn: Function λ -> (indices, rate_bpf)
            target_rate_bpf: Target bitrate (bpf = bits per frame)
            max_iters: Maximum number of iterations
            tolerance_bpf: Bitrate tolerance (bpf)
            use_cache: Whether to use cache (avoid repeated encoding)
        
        Returns:
            (optimal_lambda, achieved_rate_bpf)
        """
        if max_iters is None:
            max_iters = self.config.max_binary_search_iters
        if tolerance_bpf is None:
            tolerance_bpf = self.config.rate_tolerance_bpf
        
        # Cache: avoid repeated encoding for same λ (3-5x speedup)
        # Note: cache should be cleared for each new target bitrate search
        cache = {} if use_cache else None
        
        def cached_encoder_fn(lam):
            # Quantize cache key to improve hit rate (avoid floating point precision issues)
            key = round(lam, 6)  # Keep 6 decimal places
            if cache is not None and key in cache:
                return cache[key]
            result = encoder_fn(lam)
            if cache is not None:
                cache[key] = result
            return result
        
        # If there is a prior λ (from first sample), test it first
        if lambda_hint is not None:
            logger.debug(f"  [Binary search] Received lambda_hint={lambda_hint:.4f}")
            _, hint_rate = cached_encoder_fn(lambda_hint)
            logger.debug(f"  [Binary search] hint_rate={hint_rate:.2f} bpf, target={target_rate_bpf:.2f} bpf")
            if abs(hint_rate - target_rate_bpf) < tolerance_bpf:
                # Prior λ already satisfies condition, return directly
                logger.debug(f"  [Binary search] Prior λ satisfies condition, returning directly")
                self.lambda_current = lambda_hint
                return lambda_hint, hint_rate
            # Otherwise, use hint as center to narrow search range
            lambda_low = max(self.lambda_min, lambda_hint * 0.5)
            lambda_high = min(self.lambda_max, lambda_hint * 2.0)
            logger.info(f"  [Binary search] Using hint to narrow range: [{lambda_low:.4f}, {lambda_high:.4f}] (original: [{self.lambda_min:.4f}, {self.lambda_max:.4f}])")
        else:
            lambda_low = self.lambda_min
            lambda_high = self.lambda_max
            logger.debug(f"  [Binary search] No hint, using full range: [{lambda_low:.4f}, {lambda_high:.4f}]")
        
        # Initialize best_lambda as midpoint (not lambda_current)
        # Avoid state contamination between samples
        best_lambda = (lambda_low + lambda_high) / 2
        best_rate = None
        
        for iter_idx in range(max_iters):
            # Midpoint
            lambda_mid = (lambda_low + lambda_high) / 2
            
            # Encode and measure bitrate (using cache)
            _, rate_bpf = cached_encoder_fn(lambda_mid)
            
            # Record closest match
            if best_rate is None or abs(rate_bpf - target_rate_bpf) < abs(best_rate - target_rate_bpf):
                best_lambda = lambda_mid
                best_rate = rate_bpf
            
            # Check convergence
            if abs(rate_bpf - target_rate_bpf) < tolerance_bpf:
                logger.info(f"✅ Binary search converged (iter={iter_idx+1}): "
                           f"λ={lambda_mid:.4f}, R={rate_bpf:.2f} bpf (target={target_rate_bpf:.2f})")
                self.lambda_current = lambda_mid  # Commit optimal λ to state
                return lambda_mid, rate_bpf
            
            # Update search interval
            # Note: Larger λ means lower bitrate (more SKIP)
            if rate_bpf > target_rate_bpf:
                # Bitrate too high, need to increase λ
                lambda_low = lambda_mid
            else:
                # Bitrate too low, need to decrease λ
                lambda_high = lambda_mid
            
            logger.debug(f"Binary search iter={iter_idx+1}: λ={lambda_mid:.4f}, "
                        f"R={rate_bpf:.2f} bpf (target={target_rate_bpf:.2f}), "
                        f"interval=[{lambda_low:.4f}, {lambda_high:.4f}]")
        
        logger.warning(f"⚠️ Binary search did not converge ({max_iters} iterations, acceptable): "
                      f"best λ={best_lambda:.4f}, R={best_rate:.2f} bpf (target={target_rate_bpf:.2f})")
        
        # Print cache efficiency
        if cache is not None and len(cache) > 0:
            logger.debug(f"Binary search cache: {len(cache)} λ values cached (avoiding repeated encoding)")
        
        self.lambda_current = best_lambda  # Even if not converged, commit best λ
        return best_lambda, best_rate
    
    def reset(self):
        """Reset controller state"""
        self.lambda_current = self.config.lambda_init
        self.lambda_velocity = 0.0


class RateDistortionCurve:
    """
    Rate-Distortion Curve
    Record and analyze (λ, R, D) relationship
    """
    
    def __init__(self):
        self.records = []  # [(lambda, rate_bps, distortion, info), ...]
    
    def add_point(
        self,
        lambda_val: float,
        rate_bps: float,
        distortion: float,
        info: Optional[dict] = None
    ):
        """Add a point"""
        record = {
            'lambda': lambda_val,
            'rate_bps': rate_bps,
            'distortion': distortion,
            'info': info or {}
        }
        self.records.append(record)
    
    def get_curve(self, sort_by: str = 'rate_bps'):
        """
        Get sorted curve
        
        Args:
            sort_by: 'rate_bps' or 'lambda'
        
        Returns:
            (lambdas, rates, distortions)
        """
        sorted_records = sorted(self.records, key=lambda x: x[sort_by])
        
        lambdas = np.array([r['lambda'] for r in sorted_records])
        rates = np.array([r['rate_bps'] for r in sorted_records])
        distortions = np.array([r['distortion'] for r in sorted_records])
        
        return lambdas, rates, distortions
    
    def interpolate_lambda(self, target_rate_bps: float) -> float:
        """
        Linear interpolation to find λ that reaches target bitrate
        
        Args:
            target_rate_bps: Target bitrate
        
        Returns:
            interpolated_lambda
        """
        if len(self.records) < 2:
            raise ValueError("At least 2 points required for interpolation")
        
        lambdas, rates, _ = self.get_curve(sort_by='rate_bps')
        
        # Find two points surrounding target
        if target_rate_bps <= rates[0]:
            return lambdas[0]
        if target_rate_bps >= rates[-1]:
            return lambdas[-1]
        
        # Linear interpolation
        idx = np.searchsorted(rates, target_rate_bps)
        r0, r1 = rates[idx-1], rates[idx]
        l0, l1 = lambdas[idx-1], lambdas[idx]
        
        alpha = (target_rate_bps - r0) / (r1 - r0)
        lambda_interp = l0 + alpha * (l1 - l0)
        
        return lambda_interp
    
    def save(self, filepath: str):
        """Save curve data"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.records, f, indent=2)
        logger.info(f"✅ R-D curve saved: {filepath}")
    
    def load(self, filepath: str):
        """Load curve data"""
        import json
        with open(filepath, 'r') as f:
            self.records = json.load(f)
        logger.info(f"✅ R-D curve loaded: {filepath} ({len(self.records)} points)")


def test_rate_controller():
    """Test rate controller"""
    from config import RateControlConfig
    
    config = RateControlConfig()
    controller = RateController(config)
    
    print("\n=== Testing Dual Update ===")
    target_rate = 5.0
    for step in range(10):
        # Simulate current bitrate (gradually approaching target)
        current_rate = 10.0 - step * 0.5
        lambda_new = controller.dual_update(current_rate, target_rate)
        print(f"Step {step+1}: R={current_rate:.2f}, λ={lambda_new:.4f}")
    
    print("\n=== Testing Binary Search ===")
    controller.reset()
    
    # Mock encoding function (larger λ means lower bitrate)
    def mock_encoder(lam):
        rate = 20.0 / (1 + lam)  # Inverse relationship
        return None, rate
    
    target_rate_bps = 5.0
    optimal_lambda, achieved_rate = controller.binary_search(
        mock_encoder,
        target_rate_bps,
        max_iters=10,
        tolerance_bpf=0.5
    )
    print(f"Optimal λ={optimal_lambda:.4f}, achieved bitrate={achieved_rate:.2f} bps")


if __name__ == "__main__":
    test_rate_controller()

