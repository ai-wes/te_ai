
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
    
    def fast_clone(self, parent_cell: ProductionBCell) -> ProductionBCell:
        """
        Creates a fast clone, ensuring it is placed on the correct target device.
        """
        # The gene's own clone method is now device-aware
        cloned_genes = [gene.clone() for gene in parent_cell.genes]
        
        # *** THE FIX: Pass the correct device to the new cell's constructor ***
        child_cell = ProductionBCell(cloned_genes, device=self.device)
        child_cell = child_cell.to(self.device)  # Final guarantee
        
        child_cell.lineage = parent_cell.lineage + [parent_cell.cell_id]
        
        return child_cell
        
        
        
        
    
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
