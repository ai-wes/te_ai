"""Distributed training support for TE-AI using PyTorch DDP"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from typing import Dict, List, Optional
from scripts.config import cfg
from scripts.core.production_b_cell import ProductionBCell
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class DistributedTEAI:
    """Distributed training wrapper for TE-AI populations"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Initialize process group
        self._setup_distributed()
        
    def _setup_distributed(self):
        """Initialize distributed training"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize the process group
        init_process_group(
            backend='nccl',
            rank=self.rank,
            world_size=self.world_size
        )
        torch.cuda.set_device(self.rank)
        
    def distribute_population(self, population: Dict[str, ProductionBCell]) -> Dict[str, ProductionBCell]:
        """Distribute population across GPUs"""
        # Sort cell IDs for consistent distribution
        all_cell_ids = sorted(population.keys())
        
        # Calculate cells per GPU
        cells_per_gpu = len(all_cell_ids) // self.world_size
        remainder = len(all_cell_ids) % self.world_size
        
        # Determine this GPU's cells
        start_idx = self.rank * cells_per_gpu + min(self.rank, remainder)
        end_idx = start_idx + cells_per_gpu + (1 if self.rank < remainder else 0)
        
        # Get local cells for this GPU
        local_cell_ids = all_cell_ids[start_idx:end_idx]
        local_population = {
            cell_id: population[cell_id].to(self.device) 
            for cell_id in local_cell_ids
        }
        
        logger.info(f"GPU {self.rank}: Managing {len(local_population)} cells")
        return local_population
    
    def all_gather_fitness(self, local_fitness: Dict[str, float]) -> Dict[str, float]:
        """Gather fitness scores from all GPUs"""
        # Convert to tensor for communication
        local_ids = list(local_fitness.keys())
        local_scores = torch.tensor(
            [local_fitness[cid] for cid in local_ids], 
            device=self.device
        )
        
        # Gather sizes from all ranks
        local_size = torch.tensor([len(local_ids)], device=self.device)
        all_sizes = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, local_size)
        
        # Gather fitness scores
        max_size = max(size.item() for size in all_sizes)
        padded_scores = torch.zeros(max_size, device=self.device)
        padded_scores[:len(local_scores)] = local_scores
        
        all_scores = [torch.zeros(max_size, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(all_scores, padded_scores)
        
        # Reconstruct global fitness dictionary
        global_fitness = {}
        
        # Note: This requires sharing cell IDs across ranks
        # In practice, you'd also gather the IDs
        return local_fitness  # Simplified for now
    
    def cleanup(self):
        """Clean up distributed training"""
        destroy_process_group()

def run_distributed_evolution(rank: int, world_size: int):
    """Run evolution on a single GPU in distributed mode"""
    dist_trainer = DistributedTEAI(rank, world_size)
    
    try:
        # Initialize germinal center on this GPU
        gc = ProductionGerminalCenter().to(dist_trainer.device)
        
        # Get local population subset
        local_population = dist_trainer.distribute_population(gc.population)
        
        # Run evolution with periodic synchronization
        for generation in range(cfg.num_generations):
            # Evaluate local population
            local_fitness = {}
            for cell_id, cell in local_population.items():
                # Evaluate cell (simplified)
                fitness = torch.rand(1).item()  # Replace with actual evaluation
                local_fitness[cell_id] = fitness
            
            # Synchronize fitness across GPUs every N generations
            if generation % 5 == 0:
                global_fitness = dist_trainer.all_gather_fitness(local_fitness)
                
            logger.info(f"GPU {rank}, Generation {generation}: Avg fitness {sum(local_fitness.values())/len(local_fitness):.3f}")
            
    finally:
        dist_trainer.cleanup()

# Usage:
# torchrun --nproc_per_node=4 distributed_training.py