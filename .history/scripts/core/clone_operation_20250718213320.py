


import torch
from collections import  deque
import random
import copy
from scipy import stats
from typing import List, Optional
# Import from your main file
from scripts.core.production_b_cell import ProductionBCell
from scripts.core.stem_gene_module import StemGeneModule
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger, trace

logger = get_logger()


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
        self.clone_count = 0
        self.batch_clone_count = 0
        
# In class FastClonePool:
    
    def fast_clone(self, parent: ProductionBCell) -> ProductionBCell:
        """
        Ultra-fast cloning that avoids CPU transfers and correctly handles gene types.
        """
        # *** CRITICAL FIX: Get the device from the parent cell ***
        device = next(parent.parameters()).device
        
        try:
            child_genes = []
            
            with torch.no_grad():
                for gene in parent.genes:
                    if gene.is_active:
                        gene_state = gene.state_dict()
                        
                        # ============================================================================
                        # FIX: Instantiate the correct gene type (Quantum or Continuous)
                        # ============================================================================
                        used_clone_method = False
                        if isinstance(gene, QuantumGeneModule):
                            new_gene = QuantumGeneModule(gene.gene_type, gene.variant_id)
                        elif isinstance(gene, StemGeneModule):
                            new_gene = StemGeneModule(gene.gene_types)
                        else:
                            # Try to use the gene's own clone method if available
                            if hasattr(gene, 'clone'):
                                new_gene = gene.clone()
                                used_clone_method = True
                            else:
                                # Fallback to deepcopy for unknown gene types
                                new_gene = copy.deepcopy(gene)
                                used_clone_method = True
                        # ============================================================================
                        
                        # *** CRITICAL FIX: Ensure the new gene is on the correct device ***
                        new_gene.to(device)
                        
                        # Only load state dict if we didn't use clone method
                        if not used_clone_method:
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
                        
                        # Copy non-parameter attributes (only if we didn't use clone method)
                        if not used_clone_method:
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
            
            # Create the child and ensure it's on the correct device
            child = ProductionBCell(child_genes)
            child = child.to(device)  # <--- THE FIX: move the whole cell to device
            
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
            logger.error(f"Fast clone failed: {e}, using fallback method", exc_info=True)
            # Fallback must also be on the correct device
            return parent.clone().to(device)
        
        
        
        
    
    def _fast_mutate(self, cell):
        logger.debug("Entering FastClonePool._fast_mutate")
        """Optimized mutation"""
        with torch.no_grad():
            for param in cell.parameters():
                if random.random() < cfg.mutation_rate:
                    mutation = torch.randn_like(param) * cfg.mutation_rate
                    param.data += mutation
    
    def batch_clone(self, parents: List[ProductionBCell], num_clones_per_parent: int = 1) -> List[ProductionBCell]:
        """
        Clone multiple parents in parallel for maximum efficiency
        """
        self.batch_clone_count += 1
        all_children = []
        
        with torch.no_grad():
            for parent in parents:
                for _ in range(num_clones_per_parent):
                    try:
                        child = self.fast_clone(parent)
                        all_children.append(child)
                    except Exception as e:
                        logger.warning(f"Batch clone failed for parent: {e}")
                        # Fallback to regular clone
                        child = parent.clone()
                        all_children.append(child)
        
        logger.info(f"Batch cloned {len(all_children)} cells from {len(parents)} parents")
        return all_children
