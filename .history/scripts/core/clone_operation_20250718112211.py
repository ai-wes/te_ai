


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
                        # Use the gene's own clone method if available
                        if hasattr(gene, 'clone'):
                            new_gene = gene.clone()
                        else:
                            # Fallback to deepcopy
                            new_gene = copy.deepcopy(gene)
                        
                        # *** CRITICAL FIX: Ensure the new gene is on the correct device ***
                        child_genes.append(new_gene.to(device))

            # Create the child and ensure it's on the correct device
            child = ProductionBCell(child_genes).to(device)
            
            # ... (rest of the method is correct) ...
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
