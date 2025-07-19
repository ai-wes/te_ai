
#core/breeder_gene.py
"""
Intelligent Gene Recombination Engine
=====================================

Implements the "Perfect Spawn" concept. Instead of random crossover,
this module analyzes parent genes and intelligently combines their
best components to create a superior offspring gene.
"""

import torch
import random
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class BreederGene:
    """
    Analyzes parent genes and recombines their best sub-modules
    to create a new, potentially superior, "perfect spawn" gene.
    """

    def analyze_component_quality(self, gene: ContinuousDepthGeneModule) -> float:
        """
        Analyzes the quality of a gene's components using cheap proxies.
        Here, we use the activation EMA as a proxy for a component's
        effectiveness and stability.
        
        Returns:
            A dictionary of quality scores for each major component.
        """
        # A simple but effective proxy: a component that is consistently active
        # and produces high-norm outputs is likely a strong feature extractor.
        return getattr(gene, 'activation_ema', 0.0)

    def recombine(
        self,
        parent_gene_1: ContinuousDepthGeneModule,
        parent_gene_2: ContinuousDepthGeneModule
    ) -> ContinuousDepthGeneModule:
        """
        Creates a new child gene by selecting the best components from two parents.

        Args:
            parent_gene_1: The first parent gene.
            parent_gene_2: The second parent gene.

        Returns:
            A new child gene created from the best parts of the parents.
        """
        device = next(parent_gene_1.parameters()).device

        # 1. Analyze parents to find their best components
        p1_quality = self.analyze_component_quality(parent_gene_1)
        p2_quality = self.analyze_component_quality(parent_gene_2)

        # 2. Create the "perfect spawn" - a new, blank gene
        child_gene = ContinuousDepthGeneModule(
            gene_type='Hybrid',
            variant_id=parent_gene_1.variant_id
        ).to(device)

        # 3. Inherit the best components by copying their state_dicts
        # We compare the overall quality of the genes for this demonstration.
        # A more advanced version could analyze sub-components like encoders vs. ODEs.
        if p1_quality >= p2_quality:
            best_parent = parent_gene_1
            other_parent = parent_gene_2
            logger.info("   [Breeder] Child inherits primary traits from Parent 1.")
        else:
            best_parent = parent_gene_2
            other_parent = parent_gene_1
            logger.info("   [Breeder] Child inherits primary traits from Parent 2.")

        # Copy the entire state from the best parent
        child_gene.load_state_dict(best_parent.state_dict())

        # Introduce a small amount of genetic material from the other parent for diversity
        with torch.no_grad():
            for child_param, other_param in zip(child_gene.parameters(), other_parent.parameters()):
                # Blend a small fraction (e.g., 10%) of the other parent's weights
                child_param.data.mul_(0.9).add_(other_param.data, alpha=0.1)

        logger.info(f"   ✨ Perfect Spawn Created: A new Hybrid gene was bred from parents {parent_gene_1.gene_id[:8]} and {parent_gene_2.gene_id[:8]}.")
        return child_gene