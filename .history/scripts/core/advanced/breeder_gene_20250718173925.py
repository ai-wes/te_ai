# scripts/core/breeder_gene.py

import torch
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class BreederGene:
    """
    Analyzes parent genes and intelligently recombines them to create a "Perfect Spawn."
    This version correctly handles complex, evolved architectures.
    """

    def analyze_component_quality(self, gene: ContinuousDepthGeneModule) -> float:
        """
        Analyzes the quality of a gene using its activation EMA as a proxy.
        """
        return getattr(gene, 'activation_ema', 0.0)

    def recombine(
        self,
        parent_gene_1: ContinuousDepthGeneModule,
        parent_gene_2: ContinuousDepthGeneModule
    ) -> ContinuousDepthGeneModule:
        """
        Creates a new child gene by cloning the best parent and then blending
        in traits from the other parent.
        """
        p1_quality = self.analyze_component_quality(parent_gene_1)
        p2_quality = self.analyze_component_quality(parent_gene_2)

        if p1_quality >= p2_quality:
            best_parent = parent_gene_1
            other_parent = parent_gene_2
            logger.info("   [Breeder] Child inherits primary architecture from Parent 1.")
        else:
            best_parent = parent_gene_2
            other_parent = parent_gene_1
            logger.info("   [Breeder] Child inherits primary architecture from Parent 2.")

        # --- THE FIX ---
        # 1. Create the child as a perfect clone of the best parent.
        # This ensures the architecture and all state_dict keys match perfectly.
        # We use a simple state_dict copy method for cloning.
        child_gene = type(best_parent)() # Create a new instance of the same class
        child_gene.load_state_dict(best_parent.state_dict())
        
        # Manually copy non-parameter attributes that aren't in the state_dict
        for attr in ['gene_type', 'variant_id', 'position', 'is_active']:
            if hasattr(best_parent, attr):
                setattr(child_gene, attr, getattr(best_parent, attr))

        # 2. Now, blend in a small amount of genetic material from the other parent.
        # This introduces diversity without breaking the architecture.
        with torch.no_grad():
            for child_param, other_param in zip(child_gene.parameters(), other_parent.parameters()):
                # Ensure we don't go out of bounds if parents have different numbers of parameters
                if other_param is None:
                    break
                # Blend 10% of the other parent's weights
                child_param.data.mul_(0.9).add_(other_param.data, alpha=0.1)

        logger.info(f"   ✨ Perfect Spawn Created: A new Hybrid gene was bred from parents {parent_gene_1.gene_id[:8]} and {parent_gene_2.gene_id[:8]}.")
        return child_gene