


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
# Import from your main file
from scripts.depricated.transposable_immune_ai_production_complete import (
    ProductionBCell,  ContinuousDepthGeneModule, 
     QuantumGeneModule
)

from config import cfg


# ============================================================================
# 2. OPTIMIZED CLONE OPERATION
# ============================================================================

class FastClonePool:
    """
    Pre-allocated cell pool for fast cloning without CPU transfers
    """
    def __init__(self, pool_size=100, device='cuda'):
        self.pool_size = pool_size
        self.device = device
        self.available_cells = deque()
        self.in_use = set()
        
# In class FastClonePool:
    def fast_clone(self, parent: ProductionBCell) -> ProductionBCell:
        """
        Ultra-fast cloning that avoids CPU transfers and correctly handles gene types.
        """
        try:
            child_genes = []
            
            with torch.no_grad():
                for gene in parent.genes:
                    if gene.is_active:
                        gene_state = gene.state_dict()
                        
                        # ============================================================================
                        # FIX: Instantiate the correct gene type (Quantum or Continuous)
                        # ============================================================================
                        if isinstance(gene, QuantumGeneModule):
                            new_gene = QuantumGeneModule(gene.gene_type, gene.variant_id)
                        else:
                            new_gene = ContinuousDepthGeneModule(gene.gene_type, gene.variant_id)
                        # ============================================================================
                        
                        new_gene.to(self.device)
                        
                        try:
                            # Load the state dict. strict=False allows it to ignore
                            # parameters that might not exist in one version vs another.
                            new_gene.load_state_dict(gene_state, strict=False)
                        except Exception as e_load:
                            # Fallback for safety
                            print(f"Warning: Could not load state dict for gene {gene.gene_id}. Error: {e_load}. Copying manually.")
                            for key, value in gene_state.items():
                                if key in new_gene.state_dict():
                                    target_param = new_gene.state_dict()[key]
                                    if value.shape == target_param.shape:
                                        new_gene.state_dict()[key].copy_(value)
                        
                        # Copy non-parameter attributes
                        new_gene.position = gene.position
                        new_gene.is_active = gene.is_active
                        new_gene.is_inverted = gene.is_inverted
                        new_gene.fitness_contribution = gene.fitness_contribution
                        new_gene.chromatin_accessibility = gene.chromatin_accessibility
                        new_gene.is_cold = gene.is_cold
                        new_gene.activation_ema = gene.activation_ema
                        
                        new_gene.transposition_history = copy.deepcopy(gene.transposition_history)
                        
                        # Epigenetic inheritance
                        if hasattr(gene, 'methylation_state') and hasattr(new_gene, 'methylation_state'):
                            if gene.methylation_state.shape == new_gene.methylation_state.shape:
                                new_gene.methylation_state.data.copy_(gene.methylation_state.data * cfg.methylation_inheritance)
                        
                        if hasattr(gene, 'histone_modifications') and hasattr(new_gene, 'histone_modifications'):
                            if gene.histone_modifications.shape == new_gene.histone_modifications.shape:
                                new_gene.histone_modifications.data.copy_(gene.histone_modifications.data * cfg.methylation_inheritance)
                        
                        if random.random() < 0.05:
                            transposed_child, _ = new_gene.transpose(0.1, 0.5)
                            if transposed_child:
                                child_genes.append(transposed_child)
                        
                        child_genes.append(new_gene)
            
            child = ProductionBCell(child_genes).to(self.device)
            
            child.lineage = parent.lineage + [parent.cell_id]
            child.generation = parent.generation + 1
            
            with torch.no_grad():
                parent_matrix = parent.gene_regulatory_matrix.data
                child_matrix = child.gene_regulatory_matrix.data
                
                if parent_matrix.shape == child_matrix.shape:
                    child.gene_regulatory_matrix.data.copy_(parent_matrix * 0.9 + torch.randn_like(child_matrix) * 0.1)
                else:
                    child.gene_regulatory_matrix.data.copy_(torch.randn_like(child_matrix) * 0.1)
            
            self._fast_mutate(child)
            
            return child
            
        except Exception as e:
            print(f"Fast clone failed: {e}, using fallback method")
            return parent.clone()
        
        
        
        
    def _fast_mutate(self, cell):
        """Optimized mutation"""
        with torch.no_grad():
            for param in cell.parameters():
                if random.random() < cfg.mutation_rate:
                    mutation = torch.randn_like(param) * cfg.mutation_rate
                    param.data += mutation
