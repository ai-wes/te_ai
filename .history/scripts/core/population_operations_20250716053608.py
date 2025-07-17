

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






# ============================================================================
# 3. VECTORIZED POPULATION OPERATIONS
# ============================================================================

class VectorizedPopulationOps:
    """
    Vectorized operations for population-wide computations
    """
    
    @staticmethod
    def compute_population_diversity_vectorized(population: Dict) -> Dict[str, float]:
        """
        Fully vectorized diversity computation
        """
        # Extract all gene information at once
        all_gene_types = []
        all_positions = []
        
        for cell in population.values():
            for gene in cell.genes:
                if gene.is_active:
                    all_gene_types.append(f"{gene.gene_type}{gene.variant_id}")
                    all_positions.append(gene.position)
        
        if not all_gene_types:
            return {'shannon_index': 0, 'simpson_index': 0, 'position_entropy': 0, 'gene_richness': 0}
        
        # Vectorized counting
        unique_genes, counts = np.unique(all_gene_types, return_counts=True)
        total = len(all_gene_types)
        
        # Vectorized probability computation
        probs = counts / total
        
        # Vectorized entropy calculations
        shannon = -np.sum(probs * np.log(probs + 1e-10))
        simpson = 1 - np.sum(probs ** 2)
        
        # Vectorized position entropy
        positions = np.array(all_positions)
        hist, _ = np.histogram(positions, bins=20, range=(0, 1))
        hist_prob = hist / hist.sum()
        position_entropy = -np.sum(hist_prob[hist_prob > 0] * np.log(hist_prob[hist_prob > 0]))
        
        return {
            'shannon_index': shannon,
            'simpson_index': simpson,
            'position_entropy': position_entropy,
            'gene_richness': len(unique_genes)
        }
