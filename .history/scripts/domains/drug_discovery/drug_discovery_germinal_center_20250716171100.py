"""
Drug Discovery Germinal Center
==============================

Extends ProductionGerminalCenter with drug discovery specific features
while maintaining all the complex evolution, transposition, and 
population dynamics of the original system.
"""

import torch
import numpy as np
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
import copy
from collections import defaultdict

from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.production_b_cell import ProductionBCell
from scripts.core.utils.telemetry import write_visualization_state, set_germinal_center
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger
from scripts.core.dreamer import DreamConsolidationEngine, QuantumDreamConsolidationEngine
from scripts.core.quantum_gene import QuantumGeneModule

from .drug_discovery_genes import BindingPocketGene, PharmacophoreGene, AllostericGene
from .drug_target_antigen import DrugTargetAntigen

logger = get_logger()


class DrugDiscoveryBCell(ProductionBCell):
    """
    B-Cell specialized for drug discovery with enhanced tracking
    of binding profiles and specificity.
    """
    
    def __init__(self, cell_id: str, initial_genes: List = None):
        super().__init__(cell_id, initial_genes)
        
        # Drug discovery specific tracking
        self.binding_profiles = defaultdict(list)  # target_id -> binding history
        self.specificity_scores = defaultdict(float)  # target_id -> specificity
        self.off_target_hits = []
        self.druggability_contributions = []
        
    def process_drug_target(self, target: DrugTargetAntigen) -> Tuple[torch.Tensor, Dict]:
        """
        Process a drug target antigen with enhanced tracking.
        """
        # Convert to graph
        target_graph = target.to_graph()
        
        # Process through standard forward pass
        affinity, representation, metadata = self.forward(target_graph)
        
        # Track binding profile
        target_id = target.protein_structure.pdb_id or target.antigen_type
        self.binding_profiles[target_id].append({
            'affinity': affinity.item(),
            'pockets_targeted': self._extract_pocket_targeting(metadata),
            'pharmacophores': metadata.get('pharmacophore_features', {}),
            'allosteric': metadata.get('found_allosteric', False)
        })
        
        # Compute specificity (how much better than average)
        all_affinities = [a.item() for a in self.fitness_history[-10:]]
        if all_affinities:
            specificity = affinity.item() / (np.mean(all_affinities) + 1e-6)
            self.specificity_scores[target_id] = specificity
        
        return affinity, {
            'representation': representation,
            'metadata': metadata,
            'specificity': self.specificity_scores[target_id],
            'binding_profile': self.binding_profiles[target_id][-1]
        }
    
    def _extract_pocket_targeting(self, metadata: Dict) -> List[str]:
        """Extract which pockets are being targeted based on gene activations"""
        targeted_pockets = []
        
        # Check each gene's contribution
        for gene_meta in metadata.get('gene_metadata', []):
            if gene_meta.get('pocket_focus', False):
                # Gene is focusing on pockets
                attention = gene_meta.get('pocket_attention', None)
                if attention is not None:
                    # Find top attended pockets
                    top_pockets = torch.topk(attention, k=min(3, attention.shape[-1]))
                    targeted_pockets.extend([f"pocket_{i}" for i in top_pockets.indices.tolist()])
                    
        return list(set(targeted_pockets))
    
    def compute_druggability_score(self, target: DrugTargetAntigen) -> float:
        """
        Compute comprehensive druggability score for a target.
        """
        # Get binding affinity
        affinity, profile = self.process_drug_target(target)
        
        # Base score from affinity
        base_score = affinity.item()
        
        # Bonus for pocket targeting
        pocket_bonus = len(profile['binding_profile']['pockets_targeted']) * 0.1
        
        # Bonus for finding allosteric sites
        allosteric_bonus = 0.2 if profile['binding_profile']['allosteric'] else 0.0
        
        # Penalty for low specificity
        specificity_factor = profile['specificity']
        
        # Combine scores
        druggability = (base_score + pocket_bonus + allosteric_bonus) * specificity_factor
        
        # Track contribution
        self.druggability_contributions.append({
            'target': target.antigen_type,
            'score': druggability,
            'components': {
                'affinity': base_score,
                'pockets': pocket_bonus,
                'allosteric': allosteric_bonus,
                'specificity': specificity_factor
            }
        })
        
        return druggability


class DrugDiscoveryGerminalCenter(ProductionGerminalCenter):
    """
    Germinal Center optimized for drug discovery workflows.
    Maintains all complex features while adding drug-specific capabilities.
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
        self.target_library = {}  # Store processed targets
        self.druggability_rankings = defaultdict(list)
        self.mutation_resistance_data = defaultdict(dict)
        self.selectivity_matrix = None  # Cross-target selectivity
        
        # Quantum dream state for drug discovery
        if enable_quantum_dreams:
            self.drug_dream_engine = QuantumDreamConsolidationEngine(input_dim=cfg.hidden_dim,
            )
            # Override parent's dream engine
            self.dream_engine = self.drug_dream_engine
        
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
        
        # Add to 30% of cells
        cells_to_modify = list(self.population.values())[:int(len(self.population) * 0.3)]
        
        for i, cell in enumerate(cells_to_modify):
            if i % 3 == 0:
                # Add binding pocket gene
                pocket_gene = BindingPocketGene(
                    variant_id=100 + i,
                    position=0.15 + np.random.randn() * 0.05
                )
                cell.genes.append(pocket_gene)
                logger.info(f"Added BindingPocketGene to cell {cell.cell_id}")
                
            elif i % 3 == 1:
                # Add pharmacophore gene
                pharma_gene = PharmacophoreGene(
                    variant_id=200 + i,
                    position=0.45 + np.random.randn() * 0.05
                )
                cell.genes.append(pharma_gene)
                logger.info(f"Added PharmacophoreGene to cell {cell.cell_id}")
                
            else:
                # Add allosteric gene
                allo_gene = AllostericGene(
                    variant_id=300 + i,
                    position=0.5 + np.random.randn() * 0.05
                )
                cell.genes.append(allo_gene)
                logger.info(f"Added AllostericGene to cell {cell.cell_id}")
    
    def screen_drug_targets(
        self,
        targets: List[DrugTargetAntigen],
        generations_per_target: int = 5,
        test_mutations: bool = True
    ) -> Dict[str, Dict]:
        """
        Screen multiple drug targets and rank by druggability.
        """
        logger.info(f"\nScreening {len(targets)} drug targets...")
        results = {}
        
        for target_idx, target in enumerate(targets):
            target_id = target.protein_structure.pdb_id or f"target_{target_idx}"
            logger.info(f"\nProcessing target {target_id} ({target_idx+1}/{len(targets)})")
            
            # Store target
            self.target_library[target_id] = target
            
            # Convert to batch of antigens
            target_batch = [target.to_graph() for _ in range(cfg.batch_size)]
            
            # Evolve population on this target
            initial_fitness = None
            for gen in range(generations_per_target):
                # Write visualization state
                set_germinal_center(self)
                if self.population:
                    first_cell_id = list(self.population.keys())[0]
                    first_cell = self.population[first_cell_id]
                    if hasattr(first_cell, 'architecture_modifier'):
                        write_visualization_state(first_cell_id, first_cell.architecture_modifier)
                
                # Evolve
                self.evolve_generation(target_batch)
                
                # Track initial fitness
                if initial_fitness is None:
                    initial_fitness = self.fitness_landscape[-1]['mean_fitness']
                    
            # Analyze results
            final_fitness = self.fitness_landscape[-1]['mean_fitness']
            improvement = final_fitness - initial_fitness
            
            # Get top binders
            top_cells = self._get_top_drug_binders(target, n=10)
            
            # Test mutation resistance if requested
            mutation_scores = {}
            if test_mutations:
                logger.info(f"Testing mutation resistance for {target_id}...")
                mutation_scores = self._test_mutation_resistance(target, top_cells)
                self.mutation_resistance_data[target_id] = mutation_scores
            
            # Compute comprehensive druggability
            druggability_score = self._compute_target_druggability(
                target, top_cells, improvement, mutation_scores
            )
            
            # Store results
            results[target_id] = {
                'target': target,
                'druggability_score': druggability_score,
                'initial_fitness': initial_fitness,
                'final_fitness': final_fitness,
                'improvement': improvement,
                'convergence_speed': generations_per_target / (improvement + 1e-6),
                'top_binders': [
                    {
                        'cell_id': cell.cell_id,
                        'affinity': affinity,
                        'gene_signature': self._get_gene_signature(cell),
                        'binding_profile': cell.binding_profiles.get(target_id, [])
                    }
                    for cell, affinity in top_cells
                ],
                'mutation_resistance': mutation_scores,
                'population_diversity': self.diversity_metrics[-1] if self.diversity_metrics else {}
            }
            
            # Update rankings
            self.druggability_rankings[target_id].append(druggability_score)
            
        # Compute selectivity matrix
        self._compute_selectivity_matrix(results)
        
        return results
    
    def _get_top_drug_binders(
        self,
        target: DrugTargetAntigen,
        n: int = 10
    ) -> List[Tuple[DrugDiscoveryBCell, float]]:
        """Get top binding cells for a target"""
        affinities = []
        
        for cell in self.population.values():
            # Ensure it's a DrugDiscoveryBCell
            if not isinstance(cell, DrugDiscoveryBCell):
                # Convert if needed
                drug_cell = DrugDiscoveryBCell(cell.cell_id, cell.genes)
                drug_cell.__dict__.update(cell.__dict__)
                self.population[cell.cell_id] = drug_cell
                cell = drug_cell
            
            # Process target
            affinity, _ = cell.process_drug_target(target)
            affinities.append((cell, affinity.item()))
        
        # Sort by affinity
        affinities.sort(key=lambda x: x[1], reverse=True)
        return affinities[:n]
    
    def _test_mutation_resistance(
        self,
        target: DrugTargetAntigen,
        top_cells: List[Tuple[DrugDiscoveryBCell, float]]
    ) -> Dict:
        """Test how well top binders handle target mutations"""
        resistance_scores = {}
        
        for cell, original_affinity in top_cells[:5]:  # Test top 5
            cell_scores = []
            
            # Test multiple mutations
            for i in range(5):
                # Apply disease mutations
                mutated_target = target.apply_disease_mutations()
                
                # Test binding
                mutated_affinity, _ = cell.process_drug_target(mutated_target)
                
                # Calculate resistance (maintain affinity despite mutations)
                resistance = min(1.0, mutated_affinity.item() / (original_affinity + 1e-6))
                cell_scores.append(resistance)
                
            resistance_scores[cell.cell_id] = {
                'mean_resistance': np.mean(cell_scores),
                'min_resistance': np.min(cell_scores),
                'std_resistance': np.std(cell_scores)
            }
            
        return resistance_scores
    
    def _compute_target_druggability(
        self,
        target: DrugTargetAntigen,
        top_cells: List[Tuple[DrugDiscoveryBCell, float]],
        fitness_improvement: float,
        mutation_scores: Dict
    ) -> float:
        """Compute comprehensive druggability score"""
        
        # Base druggability from target properties
        base_druggability = target.global_druggability
        
        # Achievable affinity component
        max_affinity = top_cells[0][1] if top_cells else 0.0
        affinity_score = min(1.0, max_affinity)
        
        # Convergence component (how quickly we found good binders)
        convergence_score = min(1.0, fitness_improvement * 10)
        
        # Diversity of successful binders
        if len(top_cells) > 1:
            gene_signatures = [self._get_gene_signature(cell) for cell, _ in top_cells[:5]]
            unique_signatures = len(set(gene_signatures))
            diversity_score = unique_signatures / 5.0
        else:
            diversity_score = 0.0
            
        # Mutation resistance
        if mutation_scores:
            resistances = [s['mean_resistance'] for s in mutation_scores.values()]
            resistance_score = np.mean(resistances) if resistances else 0.0
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
    
    def _compute_selectivity_matrix(self, results: Dict):
        """Compute cross-target selectivity for all screened targets"""
        target_ids = list(results.keys())
        n_targets = len(target_ids)
        
        if n_targets < 2:
            return
            
        # Initialize selectivity matrix
        self.selectivity_matrix = np.zeros((n_targets, n_targets))
        
        for i, target_id_i in enumerate(target_ids):
            for j, target_id_j in enumerate(target_ids):
                if i == j:
                    self.selectivity_matrix[i, j] = 1.0
                    continue
                    
                # Get top binders for each target
                binders_i = results[target_id_i]['top_binders']
                binders_j = results[target_id_j]['top_binders']
                
                # Check overlap
                cells_i = set(b['cell_id'] for b in binders_i[:5])
                cells_j = set(b['cell_id'] for b in binders_j[:5])
                
                overlap = len(cells_i.intersection(cells_j))
                selectivity = 1.0 - (overlap / 5.0)
                
                self.selectivity_matrix[i, j] = selectivity
                
    def generate_druggability_report(self, results: Dict) -> Dict:
        """Generate comprehensive druggability report"""
        report = {
            'summary': {
                'targets_screened': len(results),
                'total_generations': self.generation,
                'population_size': len(self.population),
                'mean_druggability': np.mean([r['druggability_score'] for r in results.values()])
            },
            'rankings': [],
            'detailed_results': {}
        }
        
        # Sort targets by druggability
        sorted_targets = sorted(
            results.items(),
            key=lambda x: x[1]['druggability_score'],
            reverse=True
        )
        
        for rank, (target_id, result) in enumerate(sorted_targets, 1):
            target = result['target']
            
            report['rankings'].append({
                'rank': rank,
                'target_id': target_id,
                'protein': target.protein_structure.sequence[:50] + "...",
                'druggability_score': result['druggability_score'],
                'best_affinity': result['top_binders'][0]['affinity'] if result['top_binders'] else 0,
                'num_pockets': len(target.binding_pockets),
                'mutation_resistance': np.mean([
                    s['mean_resistance'] 
                    for s in result['mutation_resistance'].values()
                ]) if result['mutation_resistance'] else None
            })
            
            report['detailed_results'][target_id] = {
                'druggability_components': {
                    'base_druggability': target.global_druggability,
                    'achieved_affinity': result['final_fitness'],
                    'convergence_speed': result['convergence_speed'],
                    'binder_diversity': len(set(b['gene_signature'] for b in result['top_binders'][:5])),
                    'mutation_resistance': result['mutation_resistance']
                },
                'top_binder_profiles': result['top_binders'][:3],
                'population_metrics': result['population_diversity']
            }
            
        # Add selectivity analysis if available
        if self.selectivity_matrix is not None:
            report['selectivity_analysis'] = {
                'matrix': self.selectivity_matrix.tolist(),
                'mean_selectivity': np.mean(self.selectivity_matrix[np.triu_indices_from(self.selectivity_matrix, k=1)])
            }
            
        return report