"""Pre-allocated tensor pool for TE-AI to reduce memory allocation overhead"""

import torch
from typing import Dict, Tuple, Optional
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class TensorPool:
    """Pre-allocated tensor pool to avoid repeated allocations"""
    
    def __init__(self, device: str = None):
        self.device = device or cfg.device
        self.pools: Dict[Tuple[torch.dtype, Tuple[int, ...]], torch.Tensor] = {}
        self.in_use: Dict[str, torch.Tensor] = {}
        self.allocation_count = 0
        self.reuse_count = 0
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   fill_value: Optional[float] = None) -> torch.Tensor:
        """Get a tensor from the pool or allocate a new one"""
        key = (dtype, shape)
        
        if key in self.pools:
            # Reuse existing tensor
            tensor = self.pools[key]
            self.reuse_count += 1
            
            if fill_value is not None:
                tensor.fill_(fill_value)
            else:
                tensor.zero_()  # Clear to avoid data leakage
                
            logger.debug(f"Reused tensor {shape} (reuse rate: {self.reuse_count/(self.allocation_count+1):.2%})")
        else:
            # Allocate new tensor
            if fill_value is not None:
                tensor = torch.full(shape, fill_value, dtype=dtype, device=self.device)
            else:
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
            
            self.pools[key] = tensor
            self.allocation_count += 1
            logger.debug(f"Allocated new tensor {shape}")
        
        return tensor
    
    def get_random_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                         distribution: str = 'normal') -> torch.Tensor:
        """Get a random tensor, reusing memory when possible"""
        tensor = self.get_tensor(shape, dtype)
        
        if distribution == 'normal':
            tensor.normal_(0, 1)
        elif distribution == 'uniform':
            tensor.uniform_(-1, 1)
        elif distribution == 'bernoulli':
            tensor.bernoulli_(0.5)
        
        return tensor
    
    def clear(self):
        """Clear the pool to free memory"""
        self.pools.clear()
        self.in_use.clear()
        logger.info(f"Cleared tensor pool. Total allocations: {self.allocation_count}, Reuses: {self.reuse_count}")

# Global tensor pool instance
_global_tensor_pool = None

def get_tensor_pool(device: str = None) -> TensorPool:
    """Get or create the global tensor pool"""
    global _global_tensor_pool
    if _global_tensor_pool is None:
        _global_tensor_pool = TensorPool(device)
    return _global_tensor_pool

# Convenience functions
def get_pooled_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                     fill_value: Optional[float] = None) -> torch.Tensor:
    """Get a tensor from the global pool"""
    return get_tensor_pool().get_tensor(shape, dtype, fill_value)

def get_pooled_random(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                     distribution: str = 'normal') -> torch.Tensor:
    """Get a random tensor from the global pool"""
    return get_tensor_pool().get_random_tensor(shape, dtype, distribution)