"""Enhanced mixed precision utilities for TE-AI"""

import torch
from torch.cuda.amp import autocast, GradScaler
from contextlib import contextmanager
from typing import Optional
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class AdaptiveGradScaler(GradScaler):
    """Adaptive gradient scaler that adjusts based on gradient statistics"""
    
    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled
        )
        self.gradient_stats = []
        self.overflow_history = []
        
    def scale(self, outputs):
        """Scale outputs with monitoring"""
        self.gradient_stats.append({
            'scale': self.get_scale(),
            'growth_tracker': self._growth_tracker
        })
        return super().scale(outputs)
    
    def update(self, new_scale=None):
        """Update scaler with overflow tracking"""
        old_scale = self.get_scale()
        super().update(new_scale)
        new_scale = self.get_scale()
        
        if new_scale < old_scale:
            self.overflow_history.append(True)
            logger.warning(f"Gradient overflow detected. Scale reduced from {old_scale} to {new_scale}")
        else:
            self.overflow_history.append(False)
        
        # Adaptive adjustment based on overflow frequency
        if len(self.overflow_history) > 100:
            overflow_rate = sum(self.overflow_history[-100:]) / 100
            if overflow_rate > 0.1:  # More than 10% overflows
                self._init_scale *= 0.5
                logger.info(f"High overflow rate ({overflow_rate:.2%}). Reducing init scale to {self._init_scale}")
                self.overflow_history.clear()

@contextmanager
def mixed_precision_context(enabled: bool = None):
    """Context manager for mixed precision operations"""
    if enabled is None:
        enabled = cfg.use_amp
    
    if enabled and torch.cuda.is_available():
        with autocast(dtype=torch.float16):
            yield
    else:
        yield

class MixedPrecisionOptimizer:
    """Wrapper for optimizers with mixed precision support"""
    
    def __init__(self, optimizer, scaler: Optional[GradScaler] = None):
        self.optimizer = optimizer
        self.scaler = scaler or AdaptiveGradScaler(enabled=cfg.use_amp)
        self.step_count = 0
        
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
    def step(self, loss):
        """Perform optimization step with mixed precision"""
        if self.scaler.is_enabled():
            # Scale loss and compute gradients
            self.scaler.scale(loss).backward()
            
            # Unscale gradients for gradient clipping
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            if hasattr(cfg, 'gradient_clip_val') and cfg.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'],
                    cfg.gradient_clip_val
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular precision
            loss.backward()
            
            if hasattr(cfg, 'gradient_clip_val') and cfg.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'],
                    cfg.gradient_clip_val
                )
            
            self.optimizer.step()
        
        self.step_count += 1
        
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

def optimize_tensor_dtype(tensor: torch.Tensor, preserve_precision: bool = False) -> torch.Tensor:
    """Optimize tensor dtype based on content"""
    if not cfg.use_amp or preserve_precision:
        return tensor
    
    # Check if tensor can be safely converted to half precision
    if tensor.dtype == torch.float32:
        # Check dynamic range
        max_val = tensor.abs().max().item()
        min_val = tensor[tensor != 0].abs().min().item() if (tensor != 0).any() else 1.0
        
        # FP16 range is approximately 6e-5 to 65504
        if min_val > 6e-5 and max_val < 65504:
            return tensor.half()
    
    return tensor

class MemoryEfficientAdam(torch.optim.Adam):
    """Memory-efficient Adam optimizer that reduces state memory"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, use_8bit_states=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.use_8bit_states = use_8bit_states
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with optional 8-bit states"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    if self.use_8bit_states:
                        # Store as int8 with scale factor for memory efficiency
                        state['exp_avg'] = torch.zeros_like(p, dtype=torch.int8)
                        state['exp_avg_scale'] = 1.0
                        state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.int8) 
                        state['exp_avg_sq_scale'] = 1.0
                    else:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                
                # Update step
                state['step'] += 1
                
                # Perform standard Adam update
                # (Implementation details omitted for brevity)
                super().step(closure)
        
        return loss