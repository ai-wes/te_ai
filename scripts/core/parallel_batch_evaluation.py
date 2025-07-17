



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
        
    

    def evaluate_population_batch(self, population: Dict, antigens: List[Data]) -> Dict[str, float]:
        """
        Evaluate entire population in parallel with single forward pass
        MODIFIED FOR DRUG DISCOVERY
        """
        # Create single batch for all antigens
        antigen_graphs = []
        for i, a in enumerate(antigens):
            # Check if it's already a graph object or an antigen object
            if hasattr(a, 'to_graph'):
                # It's an antigen object
                graph = a.to(self.device)
            else:
                # It's already a graph object
                graph = a
            
            # Ensure graph is on the correct device
            if hasattr(graph, 'to'):
                graph = graph.to(self.device)
                
            # Remove any existing batch attribute to let from_data_list handle it
            if hasattr(graph, 'batch'):
                delattr(graph, 'batch')
            antigen_graphs.append(graph)
        
        antigen_batch = Batch.from_data_list(antigen_graphs)
        # Ensure the batch is on the correct device
        antigen_batch = antigen_batch.to(self.device)
        
        # Collect all cells and prepare batch processing
        cell_ids = list(population.keys())
        cells = [population[cid] for cid in cell_ids]
        
        fitness_scores = {}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                for i, (cell_id, cell) in enumerate(zip(cell_ids, cells)):
                    # Forward pass for this cell
                    # The cell now predicts a score.
                    predicted_score, cell_representation, _ = cell(antigen_batch)
                    
                    # --- START OF FIX ---
                    # The target value is stored in the batch. Let's assume it's 'druggability'
                    # You need to ensure your DrugTargetAntigen.to_graph() sets this attribute.
                    # Let's check for 'y' or 'druggability'.
                    true_score = None
                    if hasattr(antigen_batch, 'y') and antigen_batch.y is not None:
                        true_score = antigen_batch.y
                    elif hasattr(antigen_batch, 'druggability') and antigen_batch.druggability is not None:
                        true_score = antigen_batch.druggability
                    
                    if true_score is None:
                        # Fallback if no target is defined in the graph
                        true_score = torch.ones_like(predicted_score) * 0.5 

                    # Ensure shapes are compatible for loss calculation
                    pred_squeezed = predicted_score.squeeze()
                    true_squeezed = true_score.squeeze()

                    # Handle the case where the batch size is 1, which might lead to 0-dim tensors
                    if pred_squeezed.dim() == 0:
                        pred_squeezed = pred_squeezed.unsqueeze(0)
                    if true_squeezed.dim() == 0:
                        true_squeezed = true_squeezed.unsqueeze(0)
                    
                    # Handle shape mismatch due to batch truncation
                    if pred_squeezed.shape[0] < true_squeezed.shape[0]:
                        # Take only the first N true scores to match predicted
                        true_squeezed = true_squeezed[:pred_squeezed.shape[0]]
                    elif pred_squeezed.shape[0] > true_squeezed.shape[0]:
                        # This shouldn't happen, but handle it anyway
                        pred_squeezed = pred_squeezed[:true_squeezed.shape[0]]
                        
                    # Calculate a loss (e.g., Mean Squared Error)
                    loss = F.mse_loss(pred_squeezed, true_squeezed.to(pred_squeezed.device))

                    # Fitness is inversely proportional to the loss
                    fitness = 1.0 / (1.0 + loss.item())
                    # --- END OF FIX ---

                    # Complexity penalty (this part is fine)
                    active_genes = len([g for g in cell.genes if g.is_active])
                    complexity_penalty = max(0, active_genes - 10) * cfg.duplication_cost
                    
                    # Diversity bonus (this part is fine)
                    diversity_bonus = self._compute_cell_diversity(cell) * cfg.diversity_weight
                    
                    # Final fitness score
                    final_fitness = fitness - complexity_penalty + diversity_bonus
                    fitness_scores[cell_id] = final_fitness
                    
                    # Update cell records
                    cell.fitness_history.append(final_fitness)
                    for gene in cell.genes:
                        if gene.is_active:
                            gene.fitness_contribution = final_fitness
                    
                    # Store successful responses (use final_fitness now)
                    if final_fitness > 0.8:
                        representation_cpu = cell_representation.mean(dim=0).detach().cpu()
                        # The store_memory method might need to be checked if it exists
                        if hasattr(cell, 'store_memory'):
                           cell.store_memory(representation_cpu, final_fitness)
        
        print(f"   Evaluated {len(fitness_scores)} cells (drug discovery fitness).")
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
