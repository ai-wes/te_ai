



# ============================================================================
# COMPLETE OPTIMIZED TRANSPOSABLE ELEMENT AI - READY TO RUN
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from collections import defaultdict, deque
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import uuid
import random
import copy
from scipy import stats
from scripts.core.stem_gene_module import StemGeneModule, add_stem_genes_to_population
from scripts.core.production_b_cell import ProductionBCell
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.config import cfg

from scripts.core.utils.detailed_logger import get_logger, trace

logger = get_logger()

# ============================================================================
# 1. OPTIMIZED PARALLEL BATCH EVALUATION
# ============================================================================

class OptimizedBatchEvaluator:
    """
    True parallel evaluation that processes entire population in single forward pass
    """
    @trace
    def __init__(self, device='cuda'):
        self.device = device
        self._cache = {}
        
    @trace
    def evaluate_population_batch(self, population: Dict, antigens: List[Data]) -> Dict[str, float]:
        """
        Evaluate entire population in parallel with single forward pass
        """
        # Create single batch for all antigens
        antigen_batch = Batch.from_data_list([a.to(self.device) for a in antigens])
        
        # Collect all cells and prepare batch processing
        cell_ids = list(population.keys())
        cells = [population[cid] for cid in cell_ids]
        
        # Process each cell individually (safer but still optimized)
        fitness_scores = {}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                for i, (cell_id, cell) in enumerate(zip(cell_ids, cells)):
                    # Forward pass for this cell
                    affinity, cell_representation, _ = cell(antigen_batch)
                    
                    # Average affinity across antigens
                    mean_affinity = affinity.mean().item()
                    
                    # Complexity penalty
                    active_genes = len([g for g in cell.genes if g.is_active])
                    complexity_penalty = max(0, active_genes - 10) * cfg.duplication_cost
                    
                    # Diversity bonus
                    diversity_bonus = self._compute_cell_diversity(cell) * cfg.diversity_weight
                    
                    fitness = mean_affinity - complexity_penalty + diversity_bonus
                    fitness_scores[cell_id] = fitness
                    
                    # Update cell records
                    cell.fitness_history.append(fitness)
                    for gene in cell.genes:
                        if gene.is_active:
                            gene.fitness_contribution = fitness
                    
                    # Store successful responses
                    if fitness > 0.8:
                        representation_cpu = cell_representation.mean(dim=0).detach().cpu()
                        cell.store_memory(representation_cpu, fitness)
        
        print(f"   Evaluated {len(fitness_scores)} cells (optimized).")
        return fitness_scores
    
    @trace
    def _compute_cell_diversity(self, cell: ProductionBCell) -> float:
        """Compute individual cell's contribution to diversity"""
        active_genes = [g for g in cell.genes if g.is_active]
        if not active_genes:
            return 0.0
        
        # Gene type diversity
        type_counts = defaultdict(int)
        for gene in active_genes:
            type_counts[gene.gene_type] += 1
        
        # Position spread
        positions = [g.position for g in active_genes]
        position_spread = np.std(positions) if len(positions) > 1 else 0
        
        # Depth diversity
        depths = [g.compute_depth().item() for g in active_genes]
        depth_diversity = np.std(depths) if len(depths) > 1 else 0
        
        # Combined diversity score
        type_diversity = len(type_counts) / 3.0  # Normalized by max types
        
        return (type_diversity + position_spread + depth_diversity) / 3
