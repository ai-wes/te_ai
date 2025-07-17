"""
Detailed Logger for TE-AI System
Creates comprehensive logs with function tracing, timing, and detailed state tracking
"""

import logging
import logging.handlers
import os
import sys
import time
import traceback
import functools
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import json
import torch
import numpy as np


class DetailedLogger:
    """
    Comprehensive logging system with:
    - Function entry/exit tracking
    - Execution timing
    - Parameter logging
    - Return value logging
    - Exception tracking
    - Memory usage monitoring
    - GPU utilization tracking
    """
    
    def __init__(self, run_name: Optional[str] = None, log_dir: str = "logs", 
                 max_bytes: int = 15 * 1024 * 1024,  # 50MB default
                 backup_count: int = 10):  # Keep 10 backup files
        """Initialize logger with run-specific file and size limits
        
        Args:
            run_name: Name for this run (defaults to timestamp)
            log_dir: Directory to store logs
            max_bytes: Maximum size per log file in bytes (default 50MB)
            backup_count: Number of backup files to keep (default 10)
        """
        self.start_time = time.time()
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log file for this run
        self.log_file = self.log_dir / f"te_ai_run_{self.run_name}.log"
        
        # Configure root logger
        self.logger = logging.getLogger('TE_AI')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Rotating file handler with size limit
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file, 
            mode='a',  # Append mode for rotation
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler with less detail
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter for file
        file_formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(funcName)-30s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Simple formatter for console
        console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Function call stack tracking
        self.call_stack = []
        self.function_timings = {}
        self.function_calls = {}
        
        # Log run initialization
        self.logger.info("="*80)
        self.logger.info(f"TE-AI RUN INITIALIZED: {self.run_name}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Max file size: {max_bytes / 1024 / 1024:.1f} MB")
        self.logger.info(f"Backup files: {backup_count}")
        self.logger.info("="*80)
        
        # Log system info
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system configuration and capabilities"""
        self.logger.info("SYSTEM INFORMATION:")
        self.logger.info(f"  Python version: {sys.version}")
        self.logger.info(f"  PyTorch version: {torch.__version__}")
        self.logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        self.logger.info("-"*80)
    
    def _format_value(self, value: Any, max_length: int = 100) -> str:
        """Format values for logging"""
        try:
            if isinstance(value, torch.Tensor):
                return f"Tensor(shape={list(value.shape)}, dtype={value.dtype}, device={value.device})"
            elif isinstance(value, np.ndarray):
                return f"Array(shape={value.shape}, dtype={value.dtype})"
            elif isinstance(value, (list, tuple)) and len(value) > 5:
                return f"{type(value).__name__}(len={len(value)}, first={self._format_value(value[0] if value else None)})"
            elif isinstance(value, dict) and len(value) > 5:
                return f"Dict(keys={len(value)}, sample_keys={list(value.keys())[:3]})"
            elif isinstance(value, str) and len(value) > max_length:
                return f"{value[:max_length]}..."
            elif hasattr(value, '__class__'):
                # For objects that might not have a proper __str__ method
                return f"<{value.__class__.__module__}.{value.__class__.__name__} object>"
            else:
                return str(value)
        except Exception as e:
            # Fallback for any objects that cause errors
            return f"<{type(value).__name__} object (formatting error)>"
    
    def trace(self, func: Callable) -> Callable:
        """Decorator to trace function calls with timing and parameters"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            call_id = f"{func_name}_{time.time()}"
            
            # Log function entry
            self.call_stack.append(func_name)
            indent = "  " * (len(self.call_stack) - 1)
            
            # Format arguments
            args_str = ", ".join([self._format_value(arg) for arg in args[:5]])  # Limit to first 5 args
            kwargs_str = ", ".join([f"{k}={self._format_value(v)}" for k, v in list(kwargs.items())[:5]])
            
            self.logger.debug(f"{indent}-> ENTER {func_name}")
            if args_str:
                self.logger.debug(f"{indent}  args: {args_str}")
            if kwargs_str:
                self.logger.debug(f"{indent}  kwargs: {kwargs_str}")
            
            # Track timing
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success and timing
                elapsed = time.time() - start_time
                memory_delta = (torch.cuda.memory_allocated() - start_memory) if torch.cuda.is_available() else 0
                
                self.logger.debug(f"{indent}<- EXIT {func_name} [OK] ({elapsed:.3f}s)")
                if memory_delta != 0:
                    self.logger.debug(f"{indent}  Memory Δ: {memory_delta/1e6:.1f} MB")
                
                # Log return value if not None
                if result is not None:
                    self.logger.debug(f"{indent}  returns: {self._format_value(result)}")
                
                # Update statistics
                if func_name not in self.function_timings:
                    self.function_timings[func_name] = []
                    self.function_calls[func_name] = 0
                self.function_timings[func_name].append(elapsed)
                self.function_calls[func_name] += 1
                
                return result
                
            except Exception as e:
                # Log exception
                elapsed = time.time() - start_time
                self.logger.error(f"{indent}X EXCEPTION in {func_name} after {elapsed:.3f}s")
                self.logger.error(f"{indent}  {type(e).__name__}: {str(e)}")
                self.logger.debug(f"{indent}  Traceback:\n{traceback.format_exc()}")
                raise
                
            finally:
                # Clean up call stack
                self.call_stack.pop()
        
        return wrapper
    
    def log_generation(self, generation: int, metrics: Dict[str, Any]):
        """Log generation-level metrics"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"GENERATION {generation}")
        self.logger.info(f"{'='*60}")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
            else:
                self.logger.info(f"  {key}: {self._format_value(value)}")
    
    def log_phase_transition(self, from_phase: str, to_phase: str, indicators: Dict[str, float]):
        """Log phase transition events"""
        self.logger.warning(f"\n{'!'*60}")
        self.logger.warning(f"PHASE TRANSITION: {from_phase} -> {to_phase}")
        self.logger.warning(f"{'!'*60}")
        
        for key, value in indicators.items():
            self.logger.warning(f"  {key}: {value:.4f}")
    
    def log_intervention(self, intervention_type: str, details: Dict[str, Any]):
        """Log intervention events"""
        self.logger.warning(f"\n{'*'*60}")
        self.logger.warning(f"INTERVENTION: {intervention_type}")
        self.logger.warning(f"{'*'*60}")
        
        for key, value in details.items():
            self.logger.warning(f"  {key}: {self._format_value(value)}")
    
    def log_checkpoint(self, checkpoint_path: str, generation: int, metrics: Dict[str, Any]):
        """Log checkpoint saves"""
        self.logger.info(f"\n💾 CHECKPOINT SAVED: {checkpoint_path}")
        self.logger.info(f"  Generation: {generation}")
        self.logger.info(f"  Best fitness: {metrics.get('best_fitness', 'N/A')}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance statistics summary"""
        summary = {
            'total_runtime': time.time() - self.start_time,
            'function_statistics': {}
        }
        
        for func_name, timings in self.function_timings.items():
            summary['function_statistics'][func_name] = {
                'calls': self.function_calls[func_name],
                'total_time': sum(timings),
                'avg_time': np.mean(timings),
                'min_time': min(timings),
                'max_time': max(timings)
            }
        
        return summary
    
    def finalize(self):
        """Finalize logging and write summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("RUN COMPLETED")
        self.logger.info("="*80)
        
        # Log performance summary
        summary = self.get_performance_summary()
        self.logger.info(f"Total runtime: {summary['total_runtime']:.2f}s")
        
        # Log top 10 most time-consuming functions
        self.logger.info("\nTOP 10 TIME-CONSUMING FUNCTIONS:")
        sorted_funcs = sorted(
            summary['function_statistics'].items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]
        
        for func_name, stats in sorted_funcs:
            self.logger.info(
                f"  {func_name}: {stats['calls']} calls, "
                f"{stats['total_time']:.3f}s total, "
                f"{stats['avg_time']:.3f}s avg"
            )
        
        # Save summary as JSON
        summary_file = self.log_file.with_suffix('.summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\nSummary saved to: {summary_file}")
        self.logger.info("="*80)


# Global logger instance
_logger_instance: Optional[DetailedLogger] = None


def get_logger(run_name: Optional[str] = None, 
               max_bytes: int = 50 * 1024 * 1024,
               backup_count: int = 10) -> DetailedLogger:
    """Get or create the global logger instance
    
    Args:
        run_name: Name for this run (defaults to timestamp)
        max_bytes: Maximum size per log file in bytes (default 50MB)
        backup_count: Number of backup files to keep (default 10)
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DetailedLogger(run_name, max_bytes=max_bytes, backup_count=backup_count)
    return _logger_instance


def trace(func: Callable) -> Callable:
    """Convenience decorator for function tracing"""
    logger = get_logger()
    return logger.trace(func)