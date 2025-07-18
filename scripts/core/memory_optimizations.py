"""Memory optimization techniques for TE-AI"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict, Any
from functools import partial
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class CheckpointedGeneModule(nn.Module):
    """Wrapper for gene modules with gradient checkpointing"""
    
    def __init__(self, gene_module):
        super().__init__()
        self.gene_module = gene_module
        self.use_checkpointing = cfg.enable_activation_checkpointing
        
    def forward(self, *args, **kwargs):
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            return checkpoint(self._forward_impl, *args, **kwargs)
        else:
            return self._forward_impl(*args, **kwargs)
    
    def _forward_impl(self, *args, **kwargs):
        return self.gene_module(*args, **kwargs)

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient multi-head attention using Flash Attention principles"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Compute Q, K, V in chunks to reduce memory
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Use scaled dot-product attention with memory efficiency
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0+ efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),  # (B, H, L, D)
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        else:
            # Fallback to manual implementation
            attn_output = self._manual_attention(q, k, v, mask)
        
        return self.out_proj(attn_output)
    
    def _manual_attention(self, q, k, v, mask):
        """Manual attention computation with chunking for memory efficiency"""
        B, L, H, D = q.shape
        
        # Process in chunks to reduce memory usage
        chunk_size = min(512, L)  # Adjust based on available memory
        output_chunks = []
        
        for i in range(0, L, chunk_size):
            end_i = min(i + chunk_size, L)
            q_chunk = q[:, i:end_i].transpose(1, 2)  # (B, H, chunk, D)
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, k.transpose(1, 2).transpose(-2, -1)) / (D ** 0.5)
            
            if mask is not None:
                scores = scores.masked_fill(mask[:, i:end_i], float('-inf'))
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            chunk_output = torch.matmul(attn_weights, v.transpose(1, 2))
            output_chunks.append(chunk_output.transpose(1, 2))
        
        output = torch.cat(output_chunks, dim=1)
        return output.reshape(B, L, H * D)

class KVCacheOptimizer:
    """Optimizes memory usage by caching and reusing key-value pairs"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_counts = {}
        
    def get_or_compute(self, key: str, compute_fn, *args, **kwargs):
        """Get from cache or compute and cache"""
        if key in self.cache:
            self.access_counts[key] += 1
            return self.cache[key]
        
        # Compute value
        value = compute_fn(*args, **kwargs)
        
        # Add to cache with eviction if needed
        if len(self.cache) >= self.max_cache_size:
            # Evict least recently used
            lru_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_counts[lru_key]
        
        self.cache[key] = value
        self.access_counts[key] = 1
        
        return value
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_counts.clear()

class DynamicSparsityManager:
    """Manages dynamic sparsity in gene activations"""
    
    def __init__(self, sparsity_ratio: float = 0.9):
        self.sparsity_ratio = sparsity_ratio
        
    def apply_topk_sparsity(self, activations: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Apply top-k sparsity to activations"""
        k = int(activations.shape[dim] * (1 - self.sparsity_ratio))
        k = max(1, k)  # Ensure at least one element is kept
        
        # Get top-k values and indices
        topk_vals, topk_indices = torch.topk(activations.abs(), k, dim=dim)
        
        # Create sparse tensor
        sparse_activations = torch.zeros_like(activations)
        sparse_activations.scatter_(dim, topk_indices, 
                                   activations.gather(dim, topk_indices))
        
        return sparse_activations
    
    def apply_magnitude_pruning(self, tensor: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Prune values below threshold"""
        mask = tensor.abs() > threshold
        return tensor * mask

# Memory-efficient pooling operations
class MemoryEfficientPooling:
    """Memory-efficient pooling operations"""
    
    @staticmethod
    def chunked_mean_pool(x: torch.Tensor, chunk_size: int = 1000) -> torch.Tensor:
        """Compute mean pooling in chunks to save memory"""
        if x.shape[0] <= chunk_size:
            return x.mean(dim=0)
        
        # Process in chunks
        chunk_sums = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size]
            chunk_sums.append(chunk.sum(dim=0))
        
        total_sum = torch.stack(chunk_sums).sum(dim=0)
        return total_sum / x.shape[0]
    
    @staticmethod
    def weighted_pool(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted pooling with numerical stability"""
        # Normalize weights to prevent overflow
        weights = torch.softmax(weights, dim=0)
        return (x * weights.unsqueeze(-1)).sum(dim=0)