"""
Optimized Drug Discovery Germinal Center
========================================

This file shows how to properly use the optimized components while maintaining
drug discovery specific functionality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
import copy
from collections import defaultdict

from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.production_b_cell import ProductionBCell
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

from .drug_discovery_genes import BindingPocketGene, PharmacophoreGene, AllostericGene
from .drug_target_antigen import DrugTargetAntigen

logger = get_logger()


class DrugDiscoveryBCellOptimized(ProductionBCell):
    """
    B-Cell specialized for drug discovery with enhanced tracking
    of binding profiles and specificity.
    """
    
    def __init__(self, cell_id: str, initial_genes: List = None):
        super().__init__(cell_id, initial_genes)
        
        # Drug discovery specific tracking
        self.binding_profiles = defaultdict(list)
        self.specificity_scores = defaultdict(float)
        self.off_target_hits = []
        self.druggability_contributions = []
        
    def compute_druggability_bonus(self, target_id: str) -> float:
        """
        Compute a druggability bonus for this cell based on its
        drug discovery specific properties.
        """
        # Base bonus from binding profile
        if target_id in self.binding_profiles:
            profile = self.binding_profiles[target_id][-1] if self.binding_profiles[target_id] else {}
            
            # Bonus for pocket targeting
            pocket_bonus = len(profile.get('pockets_targeted', [])) * 0.05
            
            # Bonus for allosteric sites
            allosteric_bonus = 0.1 if profile.get('allosteric', False) else 0.0
            
            # Specificity bonus
            specificity = self.specificity_scores.get(target_id, 1.0)
            
            return (pocket_bonus + allosteric_bonus) * specificity
        
        return 0.0


class DrugDiscoveryGerminalCenterOptimized(ProductionGerminalCenter):
    """
    Optimized Germinal Center for drug discovery that properly uses
    all performance optimizations while maintaining drug-specific capabilities.
    """
    
    def __init__(
        self,
        population_size: int = None,
        enable_stem_cells: bool = True,
        enable_drug_genes: bool = True,
        enable_quantum_dreams: bool = True
    ):
        super().__init__()
        
        # Drug discovery specific attributes
        self.target_library = {}
        self.druggability_rankings = defaultdict(list)
        self.mutation_resistance_data = defaultdict(dict)
        self.selectivity_matrix = None
        
        # Override population size if specified
        if population_size:
            cfg.initial_population = population_size
            cfg.max_population = population_size * 4
            
        # Enable drug discovery genes
        if enable_drug_genes:
            self._add_drug_discovery_genes()
            
    def _add_drug_discovery_genes(self):
        """Add specialized drug discovery genes to initial population"""
        logger.info("Adding specialized drug discovery genes to population...")
        
        # Convert cells to DrugDiscoveryBCellOptimized
        for cell_id, cell in list(self.population.items()):
            drug_cell = DrugDiscoveryBCellOptimized(cell_id, cell.genes)
            drug_cell.__dict__.update(cell.__dict__)
            self.population[cell_id] = drug_cell
        
        # Add drug genes to 30% of cells
        cells_to_modify = list(self.population.values())[:int(len(self.population) * 0.3)]
        
        for i, cell in enumerate(cells_to_modify):
            if i % 3 == 0:
                pocket_gene = BindingPocketGene(
                    variant_id=100 + i,
                    position=0.15 + np.random.randn() * 0.05,
                    input_dim=cfg.feature_dim,
                    hidden_dim=cfg.hidden_dim,
                    output_dim=cfg.hidden_dim
                ).to(self.device)
                cell.genes.append(pocket_gene)
                
            elif i % 3 == 1:
                pharma_gene = PharmacophoreGene(
                    variant_id=200 + i,
                    position=0.45 + np.random.randn() * 0.05,
                    input_dim=cfg.feature_dim
                ).to(self.device)
                cell.genes.append(pharma_gene)
                
            else:
                allo_gene = AllostericGene(
                    variant_id=300 + i,
                    position=0.5 + np.random.randn() * 0.05,
                    input_dim=cfg.feature_dim
                ).to(self.device)
                cell.genes.append(allo_gene)
    
    def _evaluate_population_parallel(self, antigens: List[Data]) -> Dict[str, float]:
        """
        OPTIMIZED: Uses batch evaluator while adding drug discovery specific logic.
        
        This method shows the proper pattern:
        1. Use the optimized batch evaluator for base fitness
        2. Add domain-specific modifications
        3. Maintain performance benefits
        """
        # Step 1: Get base fitness scores using the optimized evaluator
        base_fitness_scores = self.batch_evaluator.evaluate_population_batch(
            self.population, 
            antigens
        )
        
        # Step 2: Apply drug discovery specific modifications
        drug_fitness_scores = {}
        
        # Extract target information from antigens if available
        target_ids = []
        for antigen in antigens:
            if hasattr(antigen, 'target_id'):
                target_ids.append(antigen.target_id)
            else:
                target_ids.append('unknown')
        
        # Apply druggability bonuses
        for cell_id, base_fitness in base_fitness_scores.items():
            cell = self.population[cell_id]
            
            # Convert to DrugDiscoveryBCell if needed
            if not isinstance(cell, DrugDiscoveryBCellOptimized):
                drug_cell = DrugDiscoveryBCellOptimized(cell.cell_id, cell.genes)
                drug_cell.__dict__.update(cell.__dict__)
                self.population[cell_id] = drug_cell
                cell = drug_cell
            
            # Compute drug-specific bonus
            druggability_bonus = 0.0
            for target_id in set(target_ids):
                if target_id != 'unknown':
                    druggability_bonus += cell.compute_druggability_bonus(target_id)
            
            # Normalize by number of targets
            if target_ids:
                druggability_bonus /= len(set(target_ids))
            
            # Combine base fitness with drug-specific bonus
            drug_fitness_scores[cell_id] = base_fitness + druggability_bonus * 0.2  # 20% weight
        
        logger.info(f"   Evaluated {len(drug_fitness_scores)} cells with drug-specific optimizations")
        return drug_fitness_scores
    
    def screen_drug_targets(
        self,
        targets: List[DrugTargetAntigen],
        generations_per_target: int = 5,
        test_mutations: bool = True
    ) -> Dict[str, Dict]:
        """
        Screen multiple drug targets using optimized evaluation.
        """
        logger.info(f"\nScreening {len(targets)} drug targets with optimized evaluation...")
        results = {}
        
        for target_idx, target in enumerate(targets):
            target_id = target.protein_structure.pdb_id or f"target_{target_idx}"
            logger.info(f"\nProcessing target {target_id} ({target_idx+1}/{len(targets)})")
            
            # Store target
            self.target_library[target_id] = target
            
            # Convert to batch of antigens with target_id
            target_graphs = []
            for _ in range(cfg.batch_size):
                graph = target.to_graph()
                graph.target_id = target_id  # Add target ID for tracking
                target_graphs.append(graph)
            
            # Evolve population on this target using optimized evaluation
            initial_fitness = None
            for gen in range(generations_per_target):
                # The parent's evolve_generation will use our optimized _evaluate_population_parallel
                self.evolve_generation(target_graphs)
                
                # Track initial fitness
                if initial_fitness is None:
                    initial_fitness = self.fitness_landscape[-1]['mean_fitness']
            
            # Rest of the analysis remains the same...
            final_fitness = self.fitness_landscape[-1]['mean_fitness']
            improvement = final_fitness - initial_fitness
            
            # Get top binders using optimized operations
            top_cells = self._get_top_drug_binders_optimized(target, n=10)
            
            # Test mutations if requested
            mutation_scores = {}
            if test_mutations:
                mutation_scores = self._test_mutation_resistance_optimized(target, top_cells)
                self.mutation_resistance_data[target_id] = mutation_scores
            
            # Store results
            results[target_id] = {
                'target': target,
                'druggability_score': self._compute_target_druggability(
                    target, top_cells, improvement, mutation_scores
                ),
                'initial_fitness': initial_fitness,
                'final_fitness': final_fitness,
                'improvement': improvement,
                'top_binders': top_cells,
                'mutation_resistance': mutation_scores,
                'population_diversity': self.diversity_metrics[-1] if self.diversity_metrics else {}
            }
            
            self.druggability_rankings[target_id].append(results[target_id]['druggability_score'])
        
        return results
    
    def _get_top_drug_binders_optimized(
        self,
        target: DrugTargetAntigen,
        n: int = 10
    ) -> List[Dict]:
        """
        Get top binding cells using batch processing for efficiency.
        """
        # Create batch of target
        target_batch = [target.to_graph() for _ in range(len(self.population))]
        
        # Use batch evaluator to get all affinities at once
        affinities = self.batch_evaluator.evaluate_population_batch(
            self.population,
            target_batch[:len(self.population)]  # Match population size
        )
        
        # Sort and get top N
        sorted_cells = sorted(affinities.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Format results
        results = []
        for cell_id, affinity in sorted_cells:
            cell = self.population[cell_id]
            results.append({
                'cell_id': cell_id,
                'affinity': affinity,
                'gene_signature': self._get_gene_signature(cell),
                'binding_profile': cell.binding_profiles.get(target.protein_structure.pdb_id, [])
            })
        
        return results
    
    def _test_mutation_resistance_optimized(
        self,
        target: DrugTargetAntigen,
        top_cells: List[Dict]
    ) -> Dict:
        """
        Test mutation resistance using batch evaluation for efficiency.
        """
        resistance_scores = {}
        
        # Create batch of mutated targets
        mutated_targets = []
        for _ in range(5):  # 5 mutations per cell
            mutated_targets.append(target.apply_disease_mutations().to_graph())
        
        # Test top 5 cells
        for cell_info in top_cells[:5]:
            cell_id = cell_info['cell_id']
            original_affinity = cell_info['affinity']
            
            # Evaluate against all mutations at once
            cell_subset = {cell_id: self.population[cell_id]}
            mutation_affinities = self.batch_evaluator.evaluate_population_batch(
                cell_subset,
                mutated_targets
            )
            
            # Calculate resistance
            mutated_affinity = mutation_affinities[cell_id]
            resistance = min(1.0, mutated_affinity / (original_affinity + 1e-6))
            
            resistance_scores[cell_id] = {
                'mean_resistance': resistance,
                'original_affinity': original_affinity,
                'mutated_affinity': mutated_affinity
            }
        
        return resistance_scores
    
    def _get_gene_signature(self, cell: ProductionBCell) -> str:
        """Get a signature representing the cell's gene composition"""
        active_genes = [g for g in cell.genes if g.is_active]
        signature_parts = []
        
        for gene in active_genes:
            gene_type = gene.gene_type
            variant = gene.variant_id
            
            # Check for drug discovery genes
            if isinstance(gene, BindingPocketGene):
                gene_type = "BP"
            elif isinstance(gene, PharmacophoreGene):
                gene_type = "PH"
            elif isinstance(gene, AllostericGene):
                gene_type = "AL"
                
            signature_parts.append(f"{gene_type}{variant}")
            
        return "-".join(sorted(signature_parts))
    
    def _compute_target_druggability(
        self,
        target: DrugTargetAntigen,
        top_cells: List[Dict],
        fitness_improvement: float,
        mutation_scores: Dict
    ) -> float:
        """Compute comprehensive druggability score"""
        
        # Base druggability from target
        base_druggability = target.global_druggability
        
        # Achievable affinity
        max_affinity = top_cells[0]['affinity'] if top_cells else 0.0
        affinity_score = min(1.0, max_affinity)
        
        # Convergence score
        convergence_score = min(1.0, fitness_improvement * 10)
        
        # Diversity of binders
        if len(top_cells) > 1:
            unique_signatures = len(set(cell['gene_signature'] for cell in top_cells[:5]))
            diversity_score = unique_signatures / 5.0
        else:
            diversity_score = 0.0
        
        # Mutation resistance
        if mutation_scores:
            resistances = [s['mean_resistance'] for s in mutation_scores.values()]
            resistance_score = np.mean(resistances)
        else:
            resistance_score = 0.5
        
        # Combine scores
        druggability = (
            base_druggability * 0.2 +
            affinity_score * 0.3 +
            convergence_score * 0.2 +
            diversity_score * 0.15 +
            resistance_score * 0.15
        )
        
        return druggability