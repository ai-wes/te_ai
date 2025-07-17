#!/usr/bin/env python3

import time
import torch
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.config import cfg
from torch_geometric.data import Data
import numpy as np

def create_test_antigens(num_antigens=10):
    """Create synthetic antigens for testing"""
    antigens = []
    for i in range(num_antigens):
        x = torch.randn(20, cfg.feature_dim, device=cfg.device)
        edge_index = torch.randint(0, 20, (2, 30), device=cfg.device)
        y = torch.randint(0, 2, (1,), device=cfg.device)
        
        antigen = Data(x=x, edge_index=edge_index, y=y)
        antigens.append(antigen)
    
    return antigens

def benchmark_generation_time():
    """Benchmark generation cycle performance"""
    print("🧬 Starting TE-AI Performance Benchmark")
    print(f"Device: {cfg.device}")
    print(f"GPU Batch Size: {cfg.gpu_batch_size}")
    print(f"Initial Population: {cfg.initial_population}")
    
    center = ProductionGerminalCenter()
    
    antigens = create_test_antigens(50)
    
    generation_times = []
    
    for gen in range(3):
        print(f"\n{'='*60}")
        print(f"BENCHMARK GENERATION {gen + 1}")
        print(f"{'='*60}")
        
        start_time = time.time()
        center.evolve_generation(antigens)
        end_time = time.time()
        
        gen_time = end_time - start_time
        generation_times.append(gen_time)
        
        print(f"⏱️  Generation {gen + 1} completed in {gen_time:.2f}s")
        print(f"Population size: {len(center.population)}")
        
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1e9
            print(f"GPU Memory: {memory_gb:.2f} GB")
    
    avg_time = np.mean(generation_times)
    min_time = np.min(generation_times)
    max_time = np.max(generation_times)
    
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Minimum generation time: {min_time:.2f}s")
    print(f"Maximum generation time: {max_time:.2f}s")
    print(f"Target: <30s per generation")
    
    if avg_time < 30:
        print("✅ PERFORMANCE TARGET ACHIEVED!")
    else:
        print("❌ Performance target not met")
        improvement_needed = ((avg_time - 30) / avg_time) * 100
        print(f"Need {improvement_needed:.1f}% improvement to reach target")
    
    return generation_times

if __name__ == "__main__":
    benchmark_generation_time()
