"""
GPU Memory and Performance Optimization Utilities
===============================================

Implements automatic batch size finding, memory optimization,
and GPU utilization maximization techniques.
"""

import torch
import contextlib
import math
import os
import logging
from typing import Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """Utilities for maximizing GPU utilization"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self._original_memory_fraction = None
        
    def setup_cuda_allocator(self, max_split_size_mb: int = 512):
        """Configure CUDA memory allocator for better performance"""
        if torch.cuda.is_available():
            # Set environment variable for CUDA allocator
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size_mb}'
            logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:{max_split_size_mb}")
    
    def set_memory_fraction(self, fraction: float = 0.95):
        """Set the fraction of GPU memory PyTorch can use"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction, self.device)
            logger.info(f"Set GPU memory fraction to {fraction}")
    
    def find_optimal_batch_size(
        self, 
        model: torch.nn.Module,
        sample_input,  # Can be torch.Tensor or torch_geometric.data.Data
        start_batch: int = 32,
        growth_factor: float = 1.5,
        max_batch: Optional[int] = None
    ) -> int:
        """
        Find the largest batch size that fits in GPU memory
        
        Args:
            model: The model to test
            sample_input: A sample input (tensor or torch_geometric Data object)
            start_batch: Starting batch size
            growth_factor: Growth factor for batch size search
            max_batch: Maximum batch size to try
            
        Returns:
            Optimal batch size
        """
        model = model.to(self.device)
        model.train()
        
        batch_size = start_batch
        optimal_batch = start_batch
        
        logger.info("Starting batch size optimization...")
        
        # Check if input is a torch_geometric Data object
        from torch_geometric.data import Data, Batch
        is_graph_data = isinstance(sample_input, Data)
        
        while True:
            if max_batch and batch_size > max_batch:
                break
                
            try:
                # Clear cache before testing
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Create batch based on input type
                if is_graph_data:
                    # For graph data, create a batch by repeating the sample
                    test_input = Batch.from_data_list([sample_input] * batch_size)
                    test_input = test_input.to(self.device)
                else:
                    # For regular tensors
                    if sample_input.dim() == 1:
                        test_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
                    else:
                        test_input = sample_input.repeat(batch_size, *[1] * (sample_input.dim() - 1))
                    
                    test_input = test_input.to(self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(test_input)
                    if isinstance(output, tuple):
                        loss = output[0].sum()
                    else:
                        loss = output.sum()
                
                # Backward pass to allocate gradients
                loss.backward()
                
                # If we get here, this batch size works
                optimal_batch = batch_size
                
                # Check memory usage
                memory_used = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                utilization = memory_used / memory_total
                
                logger.info(f"Batch size {batch_size}: Memory {memory_used:.2f}/{memory_total:.2f} GB ({utilization:.1%})")
                
                # If we're using > 90% memory, stop growing
                if utilization > 0.9:
                    logger.info(f"Reached high memory utilization at batch size {batch_size}")
                    break
                
                # Grow batch size
                batch_size = math.ceil(batch_size * growth_factor)
                
                # Clear for next iteration
                torch.cuda.empty_cache()
                model.zero_grad()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    logger.info(f"OOM at batch size {batch_size}, optimal is {optimal_batch}")
                    break
                else:
                    raise
        
        # Final cleanup
        torch.cuda.empty_cache()
        model.zero_grad()
        
        return optimal_batch
    
    def enable_mixed_precision(self) -> torch.cuda.amp.GradScaler:
        """Enable automatic mixed precision training"""
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Enabled automatic mixed precision (AMP)")
            return scaler
        return None
    
    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
            
        stats = {
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
            'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,
            'total': torch.cuda.get_device_properties(self.device).total_memory / 1024**3,
            'peak_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3,
        }
        stats['utilization'] = stats['allocated'] / stats['total']
        
        return stats
    
    @contextlib.contextmanager
    def memory_efficient_mode(self):
        """Context manager for memory-efficient operations"""
        # Store current state
        old_grad_enabled = torch.is_grad_enabled()
        
        try:
            # Enable memory-efficient settings
            torch.set_grad_enabled(False)
            torch.cuda.empty_cache()
            
            yield
            
        finally:
            # Restore state
            torch.set_grad_enabled(old_grad_enabled)
            torch.cuda.empty_cache()
    
    def optimize_model_for_inference(self, model: torch.nn.Module):
        """Optimize model for inference"""
        model.eval()
        
        # Enable cudnn benchmarking
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
        # Try to compile with torch.compile if available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
                
        return model


def parallel_batch_processing(
    func: Callable,
    batch: torch.Tensor,
    chunk_size: int = 16,
    device: torch.device = None
) -> torch.Tensor:
    """
    Process a large batch in chunks to avoid OOM
    
    Args:
        func: Function to apply to each chunk
        batch: Input batch
        chunk_size: Size of each chunk
        device: Device to use
        
    Returns:
        Concatenated results
    """
    if device is None:
        device = batch.device
        
    results = []
    
    for i in range(0, len(batch), chunk_size):
        chunk = batch[i:i + chunk_size].to(device)
        with torch.cuda.amp.autocast(enabled=True):
            result = func(chunk)
        results.append(result.cpu())  # Move to CPU to save GPU memory
        
        # Clear cache between chunks
        if i + chunk_size < len(batch):
            torch.cuda.empty_cache()
    
    # Concatenate results
    return torch.cat(results, dim=0).to(device)