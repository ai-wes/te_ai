# scripts/core/breeder_gene.py

import torch
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class BreederGene:
    """
    Analyzes parent genes and intelligently recombines them to create a "Perfect Spawn."
    This version is fully device-aware to prevent cross-device errors.
    """

    def analyze_component_quality(self, gene: ContinuousDepthGeneModule) -> float:
        """Analyze the quality of a gene component."""
        return getattr(gene, 'activation_ema', 0.0)

    def recombine(
        self,
        parent_gene_1: ContinuousDepthGeneModule,
        parent_gene_2: ContinuousDepthGeneModule,
        target_device: torch.device
    ) -> ContinuousDepthGeneModule:
        """
        Creates a new child gene on the target_device by cloning the best parent
        and then blending in traits from the other parent.
        """
        # Move both parents to the target device
        p1 = parent_gene_1.to(target_device)
        p2 = parent_gene_2.to(target_device)
        p1_quality = self.analyze_component_quality(p1)
        p2_quality = self.analyze_component_quality(p2)

        if p1_quality >= p2_quality:
            best_parent = p1
            other_parent = p2
            logger.info(f"   [Breeder] Child inherits primary architecture from Parent 1.")
        else:
            best_parent = p2
            other_parent = p1
            logger.info(f"   [Breeder] Child inherits primary architecture from Parent 2.")

        # Create the child gene directly on the target device
        child_gene = ContinuousDepthGeneModule(
            gene_type=best_parent.gene_type,
            variant_id=best_parent.variant_id
        ).to(target_device)
        child_gene.load_state_dict(best_parent.state_dict(), strict=False)

        # Copy attributes
        for attr in ['position', 'is_active']:
            if hasattr(best_parent, attr):
                setattr(child_gene, attr, getattr(best_parent, attr))

        # Blend parameters, now guaranteed to be on the same device
        with torch.no_grad():
            for child_param, other_param in zip(child_gene.parameters(), other_parent.parameters()):
                if child_param.data.shape == other_param.data.shape:
                    child_param.data.mul_(0.9).add_(other_param.data, alpha=0.1)

        # FINAL GUARANTEE: Move the child gene to the target device
        child_gene = child_gene.to(target_device)

        logger.info(f"   ✨ Perfect Spawn Created: A new Hybrid gene was bred.")
        return child_gene