



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
import torch.nn.functional as F

from scripts.core.utils.detailed_logger import get_logger, trace

logger = get_logger()

# ============================================================================
# 1. OPTIMIZED PARALLEL BATCH EVALUATION
# ============================================================================

class OptimizedBatchEvaluator:
    """
    True parallel evaluation that processes entire population in single forward pass
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self._cache = {}
        
    

    def evaluate_population_batch(self, population: Dict, antigens: list, fitness_function) -> Dict[str, float]:
        """
        Evaluates the entire population, calculates raw performance metrics, and then applies
        a specialized fitness function to get the final score for each cell.
        """
        if not population or not antigens:
            return {}

        antigen_batch = Batch.from_data_list(antigens).to(self.device)
        true_score = antigen_batch.y.to(self.device).squeeze()

        fitness_scores = {}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                for cell_id, cell in population.items():
                    # --- 1. Measure Raw Performance ---
                    start_time = time.perf_counter()
                    predicted_score, _, _ = cell(antigen_batch)
                    end_time = time.perf_counter()
                    
                    # Calculate raw metrics
                    inference_time_per_molecule = (end_time - start_time) / len(antigens)
                    
                    # Calculate raw fitness based on MSE loss
                    loss = F.mse_loss(predicted_score.squeeze(), true_score)
                    raw_fitness = 1.0 / (1.0 + loss.item())

                    # --- 2. Apply Specialized Fitness Function ---
                    evaluation_results = {
                        'raw_fitness': raw_fitness,
                        'inference_time': inference_time_per_molecule
                    }
                    specialized_fitness = fitness_function(cell, evaluation_results)

                    # --- 3. Apply Penalties/Bonuses ---
                    active_genes = len([g for g in cell.genes if g.is_active])
                    complexity_penalty = max(0, active_genes - 10) * cfg.duplication_cost
                    
                    # Note: A diversity bonus is complex in this context and might be better handled
                    # at the selection stage. For now, we focus on the core fitness.
                    
                    final_fitness = specialized_fitness - complexity_penalty
                    fitness_scores[cell_id] = final_fitness
                    
                    # Update cell's internal history
                    cell.fitness_history.append(final_fitness)

        logger.info(f"   Evaluated {len(fitness_scores)} cells.")
        return fitness_scores
    
        
    
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
