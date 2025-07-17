



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
        
        # Extract target scores from batch
        true_score = None
        if hasattr(antigen_batch, 'y') and antigen_batch.y is not None:
            true_score = antigen_batch.y
        elif hasattr(antigen_batch, 'druggability') and antigen_batch.druggability is not None:
            true_score = antigen_batch.druggability
        
        if true_score is None:
            true_score = torch.ones(antigen_batch.num_graphs, device=self.device) * 0.5
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                # Process cells in batches for better GPU utilization
                batch_size = min(32, len(cells))  # Process up to 32 cells at once
                
                for batch_start in range(0, len(cells), batch_size):
                    batch_end = min(batch_start + batch_size, len(cells))
                    batch_cells = cells[batch_start:batch_end]
                    batch_cell_ids = cell_ids[batch_start:batch_end]
                    
                    # Collect predictions from this batch of cells
                    batch_predictions = []
                    batch_representations = []
                    
                    for cell in batch_cells:
                        predicted_score, cell_representation, _ = cell(antigen_batch)
                        batch_predictions.append(predicted_score)
                        batch_representations.append(cell_representation)
                    
                    # Stack predictions for vectorized loss computation
                    if batch_predictions:
                        pred_stack = torch.stack([p.squeeze() for p in batch_predictions])
                        repr_stack = torch.stack(batch_representations)
                        
                        # Ensure true_score matches prediction shape
                        true_squeezed = true_score.squeeze()
                        if true_squeezed.dim() == 0:
                            true_squeezed = true_squeezed.unsqueeze(0)
                        
                        # Expand true_score for each cell in batch
                        true_expanded = true_squeezed.unsqueeze(0).expand(pred_stack.shape[0], -1)
                        
                        # Handle shape mismatch
                        min_size = min(pred_stack.shape[1], true_expanded.shape[1])
                        pred_stack = pred_stack[:, :min_size]
                        true_expanded = true_expanded[:, :min_size]
                        
                        # Vectorized loss computation
                        losses = F.mse_loss(pred_stack, true_expanded, reduction='none').mean(dim=1)
                        fitnesses = 1.0 / (1.0 + losses)
                        
                        # Process results for each cell
                        for i, (cell_id, cell, fitness) in enumerate(zip(batch_cell_ids, batch_cells, fitnesses)):
                            # Complexity penalty
                            active_genes = len([g for g in cell.genes if g.is_active])
                            complexity_penalty = max(0, active_genes - 10) * cfg.duplication_cost
                            
                            # Diversity bonus
                            diversity_bonus = self._compute_cell_diversity(cell) * cfg.diversity_weight
                            
                            # Final fitness score
                            final_fitness = fitness.item() - complexity_penalty + diversity_bonus
                            fitness_scores[cell_id] = final_fitness
                            
                            # Update cell records
                            cell.fitness_history.append(final_fitness)
                            for gene in cell.genes:
                                if gene.is_active:
                                    gene.fitness_contribution = final_fitness
                            
                            # Store successful responses
                            if final_fitness > 0.8:
                                representation_cpu = repr_stack[i].mean(dim=0).detach().cpu()
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
