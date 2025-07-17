"""
TE-AI Cultivation Pipeline
=========================

Implements continuous evolutionary training for TE-AI on drug discovery tasks.
Unlike traditional training, TE-AI is "cultivated" through exposure to data.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Iterator, Tuple
import time
from collections import deque
import deepchem as dc
from torch_geometric.data import Data

from scripts.domains.drug_discovery.drug_discovery_germinal_center import DrugDiscoveryGerminalCenter
from scripts.domains.drug_discovery.deepchem_converter import DeepChemToTEAI, prepare_deepchem_benchmarks
from scripts.core.utils.detailed_logger import get_logger
from scripts.config import cfg

logger = get_logger()


class TEAICultivator:
    """
    Cultivates TE-AI populations through continuous evolution.
    
    Unlike traditional training:
    - No fixed epochs - continuous generations
    - Population evolves its own architecture
    - Tracks emergence of specialized genes
    - Monitors phase transitions
    """
    
    def __init__(
        self,
        population_size: int = 128,
        enable_drug_genes: bool = True,
        track_emergence: bool = True
    ):
        self.population_size = population_size
        self.enable_drug_genes = enable_drug_genes
        self.track_emergence = track_emergence
        
        # Initialize germinal center
        self.germinal_center = DrugDiscoveryGerminalCenter(
            population_size=population_size,
            enable_drug_genes=enable_drug_genes
        )
        
        # Tracking
        self.generation = 0
        self.emergence_log = []
        self.performance_history = deque(maxlen=100)
        self.gene_diversity_history = deque(maxlen=100)
        self.architecture_snapshots = {}
        
    def create_data_stream(
        self,
        antigens: List,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Iterator[List]:
        """
        Create continuous data stream for cultivation.
        
        Args:
            antigens: List of antigens to stream
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of antigens
        """
        indices = np.arange(len(antigens))
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
                
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = [antigens[idx] for idx in batch_indices]
                yield batch
                
    def cultivate(
        self,
        train_antigens: List,
        valid_antigens: List,
        generations: int = 100,
        batch_size: int = 32,
        target_performance: float = 0.95,
        checkpoint_interval: int = 10
    ) -> Dict:
        """
        Cultivate TE-AI population through evolutionary exposure.
        
        Args:
            train_antigens: Training antigens
            valid_antigens: Validation antigens
            generations: Number of generations to evolve
            batch_size: Batch size for evolution
            target_performance: Target performance to achieve
            checkpoint_interval: How often to save checkpoints
            
        Returns:
            Cultivation results and statistics
        """
        logger.info(f"Starting TE-AI cultivation for {generations} generations")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Training samples: {len(train_antigens)}")
        logger.info(f"Validation samples: {len(valid_antigens)}")
        
        # Create data streams
        train_stream = self.create_data_stream(train_antigens, batch_size)
        
        # Cultivation loop
        start_time = time.time()
        best_performance = 0
        generations_to_target = None
        
        for generation in range(generations):
            self.generation = generation
            
            # Get next batch
            antigen_batch = next(train_stream)
            
            # Convert to graphs
            antigen_graphs = []
            for antigen in antigen_batch:
                graph = antigen.to_graph()
                antigen_graphs.append(graph)
            
            # Evolve one generation
            stats = self.germinal_center.evolve_generation(antigen_graphs)
            
            # Track performance
            if stats:
                current_fitness = stats.get('best_fitness', 0)
                self.performance_history.append(current_fitness)
                
                # Check if target reached
                if current_fitness > target_performance and generations_to_target is None:
                    generations_to_target = generation
                    logger.info(f"🎯 Target performance {target_performance} reached at generation {generation}!")
                
                # Update best
                if current_fitness > best_performance:
                    best_performance = current_fitness
                    
                # Track emergence
                if self.track_emergence:
                    self._track_gene_emergence(stats)
                    
            # Validation check every 10 generations
            if generation % 10 == 0:
                val_performance = self._validate(valid_antigens[:100])  # Quick validation
                logger.info(f"Generation {generation}: Train={current_fitness:.4f}, Val={val_performance:.4f}")
                
                # Track architecture
                if self.track_emergence:
                    self._snapshot_architecture(generation)
                    
            # Checkpoint
            if generation % checkpoint_interval == 0:
                self._save_checkpoint(generation)
                
        # Final statistics
        cultivation_time = time.time() - start_time
        
        results = {
            'total_generations': generations,
            'cultivation_time': cultivation_time,
            'best_performance': best_performance,
            'generations_to_target': generations_to_target,
            'final_population_size': len(self.germinal_center.population),
            'performance_history': list(self.performance_history),
            'emergence_log': self.emergence_log,
            'gene_diversity': self._calculate_gene_diversity(),
            'architecture_evolution': self.architecture_snapshots
        }
        
        logger.info(f"\nCultivation Complete!")
        logger.info(f"  Time: {cultivation_time:.2f}s")
        logger.info(f"  Best Performance: {best_performance:.4f}")
        logger.info(f"  Final Population: {results['final_population_size']} cells")
        
        return results
    
    def _track_gene_emergence(self, stats: Dict):
        """Track emergence of new gene types and capabilities"""
        
        # Count gene types in population
        gene_counts = {}
        for cell in self.germinal_center.population.values():
            for gene in cell.genes:
                gene_type = gene.__class__.__name__
                gene_counts[gene_type] = gene_counts.get(gene_type, 0) + 1
                
        # Check for new emergent genes
        for gene_type, count in gene_counts.items():
            if gene_type not in ['V', 'D', 'J']:  # Not basic genes
                emergence_event = {
                    'generation': self.generation,
                    'gene_type': gene_type,
                    'count': count,
                    'fitness': stats.get('best_fitness', 0)
                }
                
                # Check if this is first emergence
                if not any(e['gene_type'] == gene_type for e in self.emergence_log):
                    logger.info(f"✨ NEW GENE EMERGED: {gene_type} at generation {self.generation}!")
                    emergence_event['first_emergence'] = True
                    
                self.emergence_log.append(emergence_event)
                
        # Track diversity
        self.gene_diversity_history.append(len(gene_counts))
        
    def _snapshot_architecture(self, generation: int):
        """Take a snapshot of population architecture"""
        
        snapshot = {
            'generation': generation,
            'population_size': len(self.germinal_center.population),
            'gene_types': {},
            'avg_genes_per_cell': 0,
            'specialized_cells': 0
        }
        
        total_genes = 0
        for cell in self.germinal_center.population.values():
            total_genes += len(cell.genes)
            
            # Count specialized drug discovery genes
            drug_genes = sum(1 for g in cell.genes 
                           if 'Binding' in g.__class__.__name__ or 
                              'Pharmaco' in g.__class__.__name__ or
                              'Allosteric' in g.__class__.__name__)
            if drug_genes > 0:
                snapshot['specialized_cells'] += 1
                
            # Track gene type distribution
            for gene in cell.genes:
                gene_type = gene.__class__.__name__
                snapshot['gene_types'][gene_type] = snapshot['gene_types'].get(gene_type, 0) + 1
                
        snapshot['avg_genes_per_cell'] = total_genes / len(self.germinal_center.population)
        
        self.architecture_snapshots[generation] = snapshot
        
    def _validate(self, valid_antigens: List) -> float:
        """Quick validation on subset of data"""
        
        # Convert to graphs
        graphs = []
        for antigen in valid_antigens:
            graph = antigen.to_graph()
            graphs.append(graph)
            
        # Evaluate population
        fitness_scores = self.germinal_center.batch_evaluator.evaluate_population_batch(
            self.germinal_center.population,
            graphs
        )
        
        # Return best fitness
        return max(fitness_scores.values()) if fitness_scores else 0.0
        
    def _calculate_gene_diversity(self) -> Dict:
        """Calculate diversity metrics for final population"""
        
        gene_types = set()
        gene_positions = []
        gene_depths = []
        
        for cell in self.germinal_center.population.values():
            for gene in cell.genes:
                gene_types.add(gene.__class__.__name__)
                
                if hasattr(gene, 'position'):
                    gene_positions.append(gene.position)
                    
                if hasattr(gene, 'log_depth'):
                    gene_depths.append(gene.log_depth.item())
                    
        return {
            'unique_gene_types': len(gene_types),
            'gene_types': list(gene_types),
            'position_diversity': np.std(gene_positions) if gene_positions else 0,
            'depth_diversity': np.std(gene_depths) if gene_depths else 0
        }
        
    def _save_checkpoint(self, generation: int):
        """Save checkpoint of current population"""
        
        checkpoint_path = f"cultivation_checkpoints/gen_{generation}.pt"
        logger.info(f"Saving checkpoint at generation {generation}")
        
        # Save population state
        torch.save({
            'generation': generation,
            'population_state': self.germinal_center.state_dict(),
            'performance_history': list(self.performance_history),
            'emergence_log': self.emergence_log
        }, checkpoint_path)


def run_cultivation_experiment(
    dataset_name: str = 'bbbp',
    generations: int = 100,
    population_size: int = 128
) -> Dict:
    """
    Run a complete cultivation experiment on a dataset.
    
    Args:
        dataset_name: MoleculeNet dataset name
        generations: Number of generations to evolve
        population_size: Size of TE-AI population
        
    Returns:
        Experiment results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TE-AI CULTIVATION EXPERIMENT: {dataset_name.upper()}")
    logger.info(f"{'='*60}\n")
    
    # Load data
    converter = DeepChemToTEAI()
    train_antigens, valid_antigens, test_antigens = converter.convert_molnet_dataset(dataset_name)
    
    # Create cultivator
    cultivator = TEAICultivator(
        population_size=population_size,
        enable_drug_genes=True,
        track_emergence=True
    )
    
    # Cultivate
    results = cultivator.cultivate(
        train_antigens,
        valid_antigens,
        generations=generations,
        target_performance=0.95
    )
    
    # Final evaluation on test set
    logger.info("\nFinal test set evaluation...")
    test_performance = cultivator._validate(test_antigens[:200])
    results['test_performance'] = test_performance
    
    logger.info(f"\nFinal Test Performance: {test_performance:.4f}")
    
    # Emergence summary
    if results['emergence_log']:
        logger.info("\nGene Emergence Summary:")
        unique_genes = set(e['gene_type'] for e in results['emergence_log'])
        for gene in unique_genes:
            first_gen = next(e['generation'] for e in results['emergence_log'] 
                           if e['gene_type'] == gene)
            logger.info(f"  {gene}: First emerged at generation {first_gen}")
            
    return results


if __name__ == "__main__":
    # Run cultivation experiment
    results = run_cultivation_experiment(
        dataset_name='bbbp',
        generations=50,
        population_size=64
    )
    
    # Display results
    print(f"\nCultivation Results:")
    print(f"  Best Performance: {results['best_performance']:.4f}")
    print(f"  Test Performance: {results['test_performance']:.4f}")
    print(f"  Time to Target: {results['generations_to_target']} generations")
    print(f"  Gene Types Emerged: {results['gene_diversity']['unique_gene_types']}")
    print(f"  Final Population: {results['final_population_size']} cells")