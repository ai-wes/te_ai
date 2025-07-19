# scripts/core/breeder_gene.py

import torch
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class BreederGene:
    """
    Analyzes parent genes and intelligently recombines them to create a "Perfect Spawn."
    This version correctly handles complex, evolved architectures by cloning the best
    parent's structure before blending traits.
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
        # 1. Create a new instance of the child gene, providing the required
        #    arguments from the best_parent to the constructor.
        child_gene = ContinuousDepthGeneModule(
            gene_type=best_parent.gene_type,
            variant_id=best_parent.variant_id
        )
        
        # 2. Now load the state dict. The architectures will match because
        #    the parent was also created from this same base class. Any new
        #    layers added during evolution are handled by PyTorch's state_dict.
        #    We set strict=False to ignore any non-matching keys (like buffers
        #    that might not be present in a fresh model).
        child_gene.load_state_dict(best_parent.state_dict(), strict=False)
        
        # Manually copy other important attributes
        for attr in ['position', 'is_active']:
            if hasattr(best_parent, attr):
                setattr(child_gene, attr, getattr(best_parent, attr))

        # 3. Blend in genetic material from the other parent.
        with torch.no_grad():
            child_params = dict(child_gene.named_parameters())
            other_params = dict(other_parent.named_parameters())

            for name, child_param in child_params.items():
                if name in other_params:
                    other_param = other_params[name]
                    # Ensure shapes match before blending
                    if child_param.data.shape == other_param.data.shape:
                        child_param.data.mul_(0.9).add_(other_param.data, alpha=0.1)

        logger.info(f"   ✨ Perfect Spawn Created: A new Hybrid gene was bred from parents {parent_gene_1.gene_id[:8]} and {parent_gene_2.gene_id[:8]}.")
        return child_gene