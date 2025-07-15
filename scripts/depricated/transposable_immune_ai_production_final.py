import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from torch_geometric.utils import to_undirected, add_self_loops
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import copy
import uuid
import json
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
import os
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hashlib


# ============================================================================
# Complete Production Population Manager
# ============================================================================

class ProductionGerminalCenter:
    """Production-ready population manager with all features"""
    
    def __init__(self):
        self.population: Dict[str, ProductionBCell] = {}
        self.generation = 0
        self.current_stress = 0.0
        
        # History tracking
        self.fitness_landscape = []
        self.diversity_metrics = []
        self.transposition_events = []
        self.phase_transitions = []
        
        # Advanced systems
        self.dream_engine = DreamConsolidationEngine()
        self.phase_detector = PhaseTransitionDetector()
        self.plasmid_pool = deque(maxlen=200)
        
        # Scheduled tasks
        self.scheduled_tasks = []
        
        # Performance optimization
        self.gpu_cache = {}
        self.parallel_executor = ThreadPoolExecutor(max_workers=CFG.num_workers)
        
        # Initialize population
        self._initialize_population()
        
        # Mixed precision training
        if CFG.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def _initialize_population(self):
        """Create initial diverse population"""
        print(f"\n🧬 Initializing production population with {CFG.initial_population} cells...")
        
        for i in range(CFG.initial_population):
            cell = self._create_random_cell()
            self.population[cell.cell_id] = cell
            
            if (i + 1) % 50 == 0:
                print(f"   Created {i + 1}/{CFG.initial_population} cells...")
        
        print(f"✅ Population initialized with {len(self.population)} cells")
    
    def _create_random_cell(self) -> ProductionBCell:
        """Create cell with random gene configuration"""
        genes = []
        
        # V genes (variable region)
        num_v = random.randint(1, 4)
        for _ in range(num_v):
            gene = ContinuousDepthGeneModule('V', random.randint(1, 100))
            gene.position = np.clip(np.random.normal(0.15, 0.1), 0, 0.3)
            gene.log_depth.data = torch.tensor(np.random.normal(0, 0.3))
            genes.append(gene)
        
        # D genes (diversity region)
        num_d = random.randint(1, 3)
        for _ in range(num_d):
            gene = ContinuousDepthGeneModule('D', random.randint(1, 50))
            gene.position = np.clip(np.random.normal(0.45, 0.1), 0.3, 0.6)
            gene.log_depth.data = torch.tensor(np.random.normal(0, 0.2))
            genes.append(gene)
        
        # J genes (joining region)
        num_j = random.randint(1, 2)
        for _ in range(num_j):
            gene = ContinuousDepthGeneModule('J', random.randint(1, 10))
            gene.position = np.clip(np.random.normal(0.8, 0.1), 0.6, 1.0)
            gene.log_depth.data = torch.tensor(np.random.normal(0, 0.2))
            genes.append(gene)
        
        return ProductionBCell(genes).to(CFG.device)
    
    def _add_random_individuals(self, count: int):
        """Add new random individuals to population"""
        for _ in range(count):
            if len(self.population) < CFG.max_population:
                cell = self._create_random_cell()
                self.population[cell.cell_id] = cell
    
    def evolve_generation(self, antigens: List[Data]):
        """Complete evolution cycle with all systems"""
        generation_start = time.time()
        self.generation += 1
        
        print(f"\n{'='*80}")
        print(f"GENERATION {self.generation}")
        print(f"{'='*80}")
        
        # Phase 1: Parallel fitness evaluation
        print("\n📊 Phase 1: Fitness Evaluation")
        fitness_scores = self._evaluate_population_parallel(antigens)
        
        # Phase 2: Compute metrics and detect stress
        print("\n📈 Phase 2: Metrics and Stress Detection")
        metrics = self._compute_comprehensive_metrics(fitness_scores)
        self.current_stress = self._detect_population_stress(metrics)
        
        # Phase 3: Phase transition detection and intervention
        print("\n🔍 Phase 3: Phase Transition Analysis")
        population_state = self._get_population_state()
        intervention = self.phase_detector.update(metrics, population_state)
        
        if intervention:
            intervention(self)
        
        # Phase 4: Stress response
        if self.current_stress > CFG.stress_threshold:
            print(f"\n⚠️  Phase 4: High Stress Response (stress={self.current_stress:.3f})")
            self._execute_stress_response()
        
        # Phase 5: Selection and reproduction
        print("\n🧬 Phase 5: Selection and Reproduction")
        self._selection_and_reproduction(fitness_scores)
        
        # Phase 6: Dream consolidation (periodic)
        if self.generation % 10 == 0:
            print("\n💤 Phase 6: Dream Consolidation")
            self._execute_dream_phase()
        
        # Phase 7: Record and visualize
        self._record_generation_data(metrics, time.time() - generation_start)
        
        # Execute scheduled tasks
        self._execute_scheduled_tasks()
        
        print(f"\n⏱️  Generation completed in {time.time() - generation_start:.2f}s")
    
    def _evaluate_population_parallel(self, antigens: List[Data]) -> Dict[str, float]:
        """True parallel GPU evaluation of entire population"""
        antigen_batch = Batch.from_data_list([a.to(CFG.device) for a in antigens])
        fitness_scores = {}
        
        # Process in GPU batches
        cell_ids = list(self.population.keys())
        num_batches = (len(cell_ids) + CFG.gpu_batch_size - 1) // CFG.gpu_batch_size
        
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * CFG.gpu_batch_size
                end_idx = min((batch_idx + 1) * CFG.gpu_batch_size, len(cell_ids))
                batch_cell_ids = cell_ids[start_idx:end_idx]
                
                # Create parallel batch
                parallel_batch = create_parallel_batch(self.population, batch_cell_ids)
                
                # Process entire batch at once
                affinities, hiddens = parallel_batch(antigen_batch)
                
                # Compute fitness for each cell
                for i, cell_id in enumerate(batch_cell_ids):
                    cell = self.population[cell_id]
                    
                    # Average affinity across antigens
                    cell_affinities = affinities[i]
                    mean_affinity = cell_affinities.mean().item()
                    
                    # Compute fitness with complexity penalty
                    active_genes = len([g for g in cell.genes if g.is_active])
                    complexity_penalty = max(0, active_genes - 10) * CFG.duplication_cost
                    
                    # Diversity bonus
                    diversity_bonus = self._compute_cell_diversity(cell) * CFG.diversity_weight
                    
                    fitness = mean_affinity - complexity_penalty + diversity_bonus
                    fitness_scores[cell_id] = fitness
                    
                    # Update cell records
                    cell.fitness_history.append(fitness)
                    
                    # Update gene fitness contributions
                    for gene in cell.genes:
                        if gene.is_active:
                            gene.fitness_contribution = fitness
                    
                    # Store successful responses
                    if fitness > 0.8:
                        cell.store_memory(hiddens[i].mean(dim=0), fitness)
        
        print(f"   Evaluated {len(fitness_scores)} cells in {num_batches} GPU batches")
        return fitness_scores
    
    def _compute_comprehensive_metrics(self, fitness_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute all population metrics"""
        fitness_values = list(fitness_scores.values())
        
        # Basic statistics
        metrics = {
            'mean_fitness': np.mean(fitness_values),
            'max_fitness': np.max(fitness_values),
            'min_fitness': np.min(fitness_values),
            'fitness_variance': np.var(fitness_values),
            'fitness_skewness': stats.skew(fitness_values),
            'fitness_kurtosis': stats.kurtosis(fitness_values)
        }
        
        # Diversity metrics
        diversity = self._compute_population_diversity()
        metrics.update(diversity)
        
        # Gene statistics
        all_genes = []
        gene_depths = []
        for cell in self.population.values():
            for gene in cell.genes:
                if gene.is_active:
                    all_genes.append(f"{gene.gene_type}{gene.variant_id}")
                    gene_depths.append(gene.compute_depth().item())
        
        metrics['total_active_genes'] = len(all_genes)
        metrics['unique_genes'] = len(set(all_genes))
        metrics['mean_gene_depth'] = np.mean(gene_depths) if gene_depths else 1.0
        metrics['gene_depth_variance'] = np.var(gene_depths) if gene_depths else 0.0
        
        # Transposition rate
        recent_transpositions = [
            e for e in self.transposition_events 
            if e['generation'] >= self.generation - 10
        ]
        metrics['transposition_rate'] = len(recent_transpositions) / max(len(self.population), 1)
        
        # Phase state
        metrics['phase_state'] = self.phase_detector.current_phase
        
        print(f"   Metrics: fitness={metrics['mean_fitness']:.3f}±{metrics['fitness_variance']:.3f}, "
              f"diversity={metrics['shannon_index']:.3f}, genes={metrics['unique_genes']}")
        
        return metrics
    
    def _compute_population_diversity(self) -> Dict[str, float]:
        """Compute multiple diversity metrics"""
        # Gene position distribution
        all_positions = []
        gene_type_counts = defaultdict(int)
        
        for cell in self.population.values():
            for gene in cell.genes:
                if gene.is_active:
                    all_positions.append(gene.position)
                    gene_type_counts[f"{gene.gene_type}{gene.variant_id}"] += 1
        
        # Shannon diversity index
        total_genes = sum(gene_type_counts.values())
        shannon = 0
        for count in gene_type_counts.values():
            if count > 0:
                p = count / total_genes
                shannon -= p * np.log(p)
        
        # Simpson's diversity index
        simpson = 1 - sum((n/total_genes)**2 for n in gene_type_counts.values())
        
        # Positional entropy
        if all_positions:
            hist, _ = np.histogram(all_positions, bins=20, range=(0, 1))
            hist = hist / hist.sum()
            position_entropy = -sum(p * np.log(p + 1e-10) for p in hist if p > 0)
        else:
            position_entropy = 0
        
        return {
            'shannon_index': shannon,
            'simpson_index': simpson,
            'position_entropy': position_entropy,
            'gene_richness': len(gene_type_counts)
        }
    
    def _compute_cell_diversity(self, cell: ProductionBCell) -> float:
        """Compute individual cell's contribution to diversity"""
        active_genes = [g for g in cell.genes if g.is_active]
        if not active_genes:
            return 0.0
        
        # Gene type diversity
        type_counts = defaultdict(int)
        for gene in active_genes:
            type_counts[gene.gene_type] += 1
        
        # Position spread
        positions = [g.position for g in active_genes]
        position_spread = np.std(positions) if len(positions) > 1 else 0
        
        # Depth diversity
        depths = [g.compute_depth().item() for g in active_genes]
        depth_diversity = np.std(depths) if len(depths) > 1 else 0
        
        # Combined diversity score
        type_diversity = len(type_counts) / 3.0  # Normalized by max types
        
        return (type_diversity + position_spread + depth_diversity) / 3
    
    def _detect_population_stress(self, metrics: Dict[str, float]) -> float:
        """Sophisticated stress detection"""
        stress_factors = []
        
        # Factor 1: Fitness stagnation
        if len(self.fitness_landscape) >= CFG.stress_window:
            recent_fitness = [f['mean_fitness'] for f in self.fitness_landscape[-CFG.stress_window:]]
            fitness_trend = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]
            stagnation_stress = max(0, -fitness_trend * 100)
            stress_factors.append(stagnation_stress)
        
        # Factor 2: Low diversity
        diversity_stress = max(0, CFG.shannon_entropy_target - metrics['shannon_index'])
        stress_factors.append(diversity_stress)
        
        # Factor 3: High fitness variance (population struggling)
        variance_stress = min(1.0, metrics['fitness_variance'] * 2)
        stress_factors.append(variance_stress)
        
        # Factor 4: Phase state
        phase_stress = {
            'stable': 0.0,
            'critical_slowing': 0.5,
            'bifurcation': 0.7,
            'chaos': 0.9,
            'collapse': 1.0
        }.get(metrics['phase_state'], 0.5)
        stress_factors.append(phase_stress)
        
        # Combine factors
        stress = np.mean(stress_factors)
        
        print(f"   Stress factors: stagnation={stress_factors[0]:.2f}, "
              f"diversity={diversity_stress:.2f}, variance={variance_stress:.2f}, "
              f"phase={phase_stress:.2f}")
        print(f"   Combined stress: {stress:.3f}")
        
        return stress
    
    def _get_population_state(self) -> Dict:
        """Get comprehensive population state for phase detection"""
        fitness_values = [
            cell.fitness_history[-1] if cell.fitness_history else 0
            for cell in self.population.values()
        ]
        
        gene_positions = []
        for cell in self.population.values():
            for gene in cell.genes:
                if gene.is_active:
                    gene_positions.append((gene.position, gene.fitness_contribution))
        
        return {
            'fitness_distribution': fitness_values,
            'gene_positions': gene_positions,
            'population_size': len(self.population),
            'generation': self.generation
        }
    
    def _execute_stress_response(self):
        """Comprehensive stress response"""
        print("   Executing stress response protocols:")
        
        # 1. Increase transposition rate
        print("   • Triggering transposition cascade")
        transposition_count = 0
        for cell in list(self.population.values()):
            cell.undergo_transposition(self.current_stress, 
                                     self.diversity_metrics[-1]['shannon_index'] if self.diversity_metrics else 0.5)
            transposition_count += 1
            
            # Record events
            for gene in cell.genes:
                if gene.transposition_history:
                    self.transposition_events.append({
                        'generation': self.generation,
                        'cell_id': cell.cell_id,
                        'gene_id': gene.gene_id,
                        'event': gene.transposition_history[-1]
                    })
        
        print(f"     - {transposition_count} cells underwent transposition")
        
        # 2. Epigenetic stress response
        print("   • Applying epigenetic modifications")
        epigenetic_count = 0
        for cell in list(self.population.values())[:100]:  # Limit for efficiency
            for gene in cell.genes:
                if gene.is_active and random.random() < CFG.methylation_rate * self.current_stress:
                    sites = torch.randint(0, CFG.hidden_dim, (5,))
                    gene.add_methylation(sites, self.current_stress * 0.5)
                    epigenetic_count += 1
        
        print(f"     - {epigenetic_count} genes methylated")
        
        # 3. Horizontal gene transfer
        print("   • Facilitating horizontal gene transfer")
        transfer_count = self._execute_horizontal_transfer()
        print(f"     - {transfer_count} successful gene transfers")
        
        # 4. Inject diversity if critically low
        if self.diversity_metrics and self.diversity_metrics[-1]['shannon_index'] < 0.5:
            print("   • Injecting new diverse individuals")
            self._add_random_individuals(50)
    
    def _execute_horizontal_transfer(self) -> int:
        """Execute horizontal gene transfer between cells"""
        transfer_count = 0
        
        # Extract plasmids from successful cells
        fitness_threshold = np.percentile(
            [c.fitness_history[-1] if c.fitness_history else 0 
             for c in self.population.values()], 
            70
        )
        
        # Donor cells release plasmids
        for cell in self.population.values():
            if cell.fitness_history and cell.fitness_history[-1] > fitness_threshold:
                plasmid = cell.extract_plasmid()
                if plasmid:
                    self.plasmid_pool.append(plasmid)
        
        # Recipient cells integrate plasmids
        if self.plasmid_pool:
            recipient_cells = random.sample(
                list(self.population.values()),
                min(100, len(self.population))
            )
            
            for cell in recipient_cells:
                if random.random() < CFG.horizontal_transfer_prob * (1 + self.current_stress):
                    # Select compatible plasmid
                    plasmid = random.choice(list(self.plasmid_pool))
                    if cell.integrate_plasmid(plasmid):
                        transfer_count += 1
        
        return transfer_count
    
    def _selection_and_reproduction(self, fitness_scores: Dict[str, float]):
        """Natural selection with multiple strategies"""
        sorted_cells = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Elite preservation
        num_elite = max(5, int(len(sorted_cells) * 0.05))
        elite_ids = [cid for cid, _ in sorted_cells[:num_elite]]
        print(f"   Preserving {num_elite} elite individuals")
        
        # Tournament selection for diversity
        num_survivors = int(len(sorted_cells) * (1 - CFG.selection_pressure))
        survivors = self._tournament_selection(fitness_scores, num_survivors)
        
        # Reproduction
        new_population = {}
        
        # Keep elites
        for elite_id in elite_ids:
            new_population[elite_id] = self.population[elite_id]
        
        # Elite reproduction
        for elite_id in elite_ids:
            parent = self.population[elite_id]
            num_offspring = random.randint(2, 4)
            
            for _ in range(num_offspring):
                if len(new_population) >= CFG.max_population:
                    break
                    
                child = parent.clone()
                new_population[child.cell_id] = child
        
        # General reproduction
        for parent_id in survivors:
            if parent_id not in new_population and len(new_population) < CFG.max_population:
                parent = self.population[parent_id]
                new_population[parent_id] = parent
                
                # Offspring based on fitness
                fitness = fitness_scores[parent_id]
                num_offspring = int(1 + fitness * 2)
                
                for _ in range(num_offspring):
                    if len(new_population) >= CFG.max_population:
                        break
                        
                    child = parent.clone()
                    new_population[child.cell_id] = child
        
        # Crossover between top performers
        if CFG.crossover_rate > 0 and len(elite_ids) >= 2:
            crossover_count = int(len(new_population) * CFG.crossover_rate)
            for _ in range(crossover_count):
                if len(new_population) >= CFG.max_population:
                    break
                    
                parent1 = self.population[random.choice(elite_ids)]
                parent2 = self.population[random.choice(elite_ids)]
                
                if parent1.cell_id != parent2.cell_id:
                    child = self._crossover(parent1, parent2)
                    new_population[child.cell_id] = child
        
        self.population = new_population
        print(f"   New population size: {len(self.population)}")
    
    def _tournament_selection(self, fitness_scores: Dict[str, float], 
                            num_survivors: int) -> List[str]:
        """Tournament selection for diversity"""
        survivors = []
        tournament_size = 5
        
        cell_ids = list(fitness_scores.keys())
        
        while len(survivors) < num_survivors:
            # Random tournament
            tournament = random.sample(cell_ids, min(tournament_size, len(cell_ids)))
            
            # Winner based on fitness and diversity
            best_id = None
            best_score = -float('inf')
            
            for cid in tournament:
                fitness = fitness_scores[cid]
                diversity = self._compute_cell_diversity(self.population[cid])
                combined_score = fitness + diversity * CFG.niche_pressure
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_id = cid
            
            if best_id and best_id not in survivors:
                survivors.append(best_id)
        
        return survivors
    
    def _crossover(self, parent1: ProductionBCell, 
                  parent2: ProductionBCell) -> ProductionBCell:
        """Crossover between two cells"""
        # Combine genes from both parents
        all_genes = []
        
        # Take random subset from each parent
        p1_genes = [g for g in parent1.genes if g.is_active]
        p2_genes = [g for g in parent2.genes if g.is_active]
        
        num_from_p1 = random.randint(1, max(1, len(p1_genes) - 1))
        num_from_p2 = random.randint(1, max(1, len(p2_genes) - 1))
        
        # Select genes
        if p1_genes:
            selected_p1 = random.sample(p1_genes, min(num_from_p1, len(p1_genes)))
            all_genes.extend([copy.deepcopy(g) for g in selected_p1])
        
        if p2_genes:
            selected_p2 = random.sample(p2_genes, min(num_from_p2, len(p2_genes)))
            all_genes.extend([copy.deepcopy(g) for g in selected_p2])
        
        # Create child
        child = ProductionBCell(all_genes[:CFG.max_genes_per_clone])
        child.lineage = [parent1.cell_id, parent2.cell_id]
        
        # Combine regulatory matrices
        with torch.no_grad():
            child.gene_regulatory_matrix.data = \
                (parent1.gene_regulatory_matrix.data + parent2.gene_regulatory_matrix.data) / 2 + \
                torch.randn_like(child.gene_regulatory_matrix) * 0.1
        
        return child.to(CFG.device)
    
    def _execute_dream_phase(self):
        """Execute dream consolidation"""
        # Record experiences
        for cell in random.sample(list(self.population.values()), 
                                min(100, len(self.population))):
            if cell.fitness_history and cell.genes:
                # Get representative gene state
                gene_states = []
                for gene in cell.genes[:5]:
                    if gene.is_active and hasattr(gene, 'output_projection'):
                        weight = gene.output_projection[0].weight
                        gene_states.append(weight.mean(dim=0))
                
                if gene_states:
                    combined_state = torch.stack(gene_states).mean(dim=0)
                    self.dream_engine.episodic_memory.store(
                        combined_state,
                        'gene_expression',
                        cell.fitness_history[-1],
                        combined_state,
                        {'stress': self.current_stress, 'generation': self.generation}
                    )
        
        # Run dream consolidation
        self.dream_engine.dream_phase(self.population, CFG.dream_cycles_per_generation)
    
    def _record_generation_data(self, metrics: Dict[str, float], generation_time: float):
        """Record comprehensive generation data"""
        # Update histories
        self.fitness_landscape.append({
            'generation': self.generation,
            'time': datetime.now().isoformat(),
            'generation_time': generation_time,
            **metrics
        })
        
        self.diversity_metrics.append({
            'generation': self.generation,
            **{k: v for k, v in metrics.items() if 'diversity' in k or 'index' in k}
        })
        
        # Save checkpoint periodically
        if self.generation % CFG.checkpoint_interval == 0:
            self._save_checkpoint()
    
    def _execute_scheduled_tasks(self):
        """Execute any scheduled tasks"""
        completed_tasks = []
        
        for task in self.scheduled_tasks:
            if task['generation'] <= self.generation:
                task['action']()
                completed_tasks.append(task)
        
        # Remove completed tasks
        for task in completed_tasks:
            self.scheduled_tasks.remove(task)
    
    def _save_checkpoint(self):
        """Save population checkpoint"""
        checkpoint_path = os.path.join(CFG.save_dir, f'checkpoint_gen_{self.generation}.pt')
        
        checkpoint = {
            'generation': self.generation,
            'config': CFG.__dict__,
            'population_size': len(self.population),
            'fitness_landscape': self.fitness_landscape[-100:],  # Last 100 generations
            'diversity_metrics': self.diversity_metrics[-100:],
            'current_stress': self.current_stress,
            'phase_state': self.phase_detector.current_phase
        }
        
        # Save subset of best cells
        sorted_cells = sorted(
            self.population.items(),
            key=lambda x: x[1].fitness_history[-1] if x[1].fitness_history else 0,
            reverse=True
        )
        
        best_cells = {}
        for cid, cell in sorted_cells[:10]:
            best_cells[cid] = {
                'gene_count': len([g for g in cell.genes if g.is_active]),
                'fitness': cell.fitness_history[-1] if cell.fitness_history else 0,
                'generation': cell.generation,
                'lineage': cell.lineage[-10:]  # Last 10 ancestors
            }
        
        checkpoint['best_cells'] = best_cells
        
        torch.save(checkpoint, checkpoint_path)
        print(f"   💾 Saved checkpoint to {checkpoint_path}")

# ============================================================================
# Main Simulation Function
# ============================================================================

def run_production_simulation():
    """Run complete production simulation with all features"""
    print("\n" + "="*80)
    print("🧬 TRANSPOSABLE ELEMENT AI - PRODUCTION SYSTEM v1.0")
    print("="*80)
    print(f"\n⚙️  Configuration:")
    print(f"   Device: {CFG.device}")
    print(f"   Population: {CFG.initial_population} → {CFG.max_population}")
    print(f"   Epochs: {CFG.epochs}")
    print(f"   GPU Batch Size: {CFG.gpu_batch_size}")
    print(f"   ODE Solver: {CFG.ode_solver}")
    print(f"   Dream Cycles: {CFG.dream_cycles_per_generation}")
    
    # Initialize population
    germinal_center = ProductionGerminalCenter()
    
    # Viral evolution timeline
    viral_timeline = [
        (0, [], "Wild Type"),
        (50, [(0, 5)], "Alpha Variant"),
        (100, [(0, 5), (1, 12)], "Beta Variant"),
        (150, [(0, 5), (1, 12), (2, 18)], "Delta Variant"),
        (200, [(0, 3), (0, 7), (1, 12), (1, 15), (2, 18)], "Omicron Variant"),
        (250, [(0, 1), (0, 3), (0, 5), (0, 7), (0, 9), 
               (1, 12), (1, 15), (2, 17), (2, 18)], "Escape Variant"),
        (300, [(i, j) for i in range(3) for j in range(0, 20, 2)], "Hypermutated Variant")
    ]
    
    current_variant_idx = 0
    simulation_start = time.time()
    
    # Main evolution loop
    for epoch in range(CFG.epochs):
        # Check for viral mutation
        if current_variant_idx < len(viral_timeline) - 1:
            if epoch >= viral_timeline[current_variant_idx + 1][0]:
                current_variant_idx += 1
                _, mutations, variant_name = viral_timeline[current_variant_idx]
                print(f"\n🦠 VIRUS MUTATED TO {variant_name.upper()}!")
                print(f"   Mutation sites: {mutations}")
                
                # Spike stress for major variants
                if 'Omicron' in variant_name or 'Escape' in variant_name:
                    germinal_center.current_stress = 1.0
        
        # Generate realistic antigens
        _, mutations, variant_name = viral_timeline[current_variant_idx]
        antigens = []
        
        for i in range(CFG.batch_size):
            # Mix of conformations
            antigen = generate_realistic_antigen(
                variant_type=variant_name.lower().split()[0],
                mutations=mutations
            )
            antigens.append(antigen)
        
        # Evolve population
        germinal_center.evolve_generation(antigens)
        
        # Periodic visualization
        if epoch % CFG.plot_interval == 0:
            visualize_production_state(germinal_center, epoch)
        
        # Progress report
        if epoch % 10 == 0:
            elapsed = time.time() - simulation_start
            eta = (elapsed / (epoch + 1)) * (CFG.epochs - epoch - 1)
            print(f"\n📊 Progress: {epoch+1}/{CFG.epochs} ({(epoch+1)/CFG.epochs*100:.1f}%)")
            print(f"   Elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m")
    
    # Final analysis and visualization
    print("\n" + "="*80)
    print("🏁 SIMULATION COMPLETE")
    print("="*80)
    
    final_analysis(germinal_center, time.time() - simulation_start)
    
    return germinal_center

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_production_state(center: ProductionGerminalCenter, epoch: int):
    """Create comprehensive visualization of current state"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Gene topology with depth
    ax1 = plt.subplot(3, 3, 1)
    visualize_gene_topology_3d(center, ax1)
    
    # 2. Fitness landscape
    ax2 = plt.subplot(3, 3, 2)
    plot_fitness_landscape(center, ax2)
    
    # 3. Phase space
    ax3 = plt.subplot(3, 3, 3)
    plot_phase_space(center, ax3)
    
    # 4. Diversity metrics
    ax4 = plt.subplot(3, 3, 4)
    plot_diversity_metrics(center, ax4)
    
    # 5. Gene expression heatmap
    ax5 = plt.subplot(3, 3, 5)
    plot_gene_expression_heatmap(center, ax5)
    
    # 6. Transposition events
    ax6 = plt.subplot(3, 3, 6)
    plot_transposition_timeline(center, ax6)
    
    # 7. Population structure
    ax7 = plt.subplot(3, 3, 7)
    plot_population_structure(center, ax7)
    
    # 8. Epigenetic landscape
    ax8 = plt.subplot(3, 3, 8)
    plot_epigenetic_landscape(center, ax8)
    
    # 9. Performance metrics
    ax9 = plt.subplot(3, 3, 9)
    plot_performance_summary(center, ax9)
    
    plt.suptitle(f'Transposable Element AI - Generation {center.generation}', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(CFG.save_dir, f'state_gen_{epoch:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_gene_topology_3d(center: ProductionGerminalCenter, ax):
    """3D visualization of gene arrangements"""
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(3, 3, 1, projection='3d')
    
    # Sample cells
    sample_cells = list(center.population.values())[:20]
    
    for i, cell in enumerate(sample_cells):
        for gene in cell.genes:
            if gene.is_active:
                x = gene.position
                y = i / len(sample_cells)
                z = gene.compute_depth().item()
                
                color = {'V': 'red', 'D': 'green', 'J': 'blue'}.get(gene.gene_type, 'gray')
                size = 50 * (1 + gene.fitness_contribution)
                
                ax.scatter(x, y, z, c=color, s=size, alpha=0.6)
    
    ax.set_xlabel('Genomic Position')
    ax.set_ylabel('Cell Index')
    ax.set_zlabel('Neural Depth')
    ax.set_title('3D Gene Topology')

def plot_fitness_landscape(center: ProductionGerminalCenter, ax):
    """Plot fitness evolution with phase transitions"""
    if not center.fitness_landscape:
        return
    
    generations = [d['generation'] for d in center.fitness_landscape]
    mean_fitness = [d['mean_fitness'] for d in center.fitness_landscape]
    max_fitness = [d['max_fitness'] for d in center.fitness_landscape]
    
    ax.plot(generations, mean_fitness, 'b-', label='Mean', linewidth=2)
    ax.plot(generations, max_fitness, 'g--', label='Max', linewidth=2)
    
    # Mark phase transitions
    for transition in center.phase_detector.transition_history:
        gen = transition['metrics'].get('generation', 0)
        ax.axvline(x=gen, color='red', alpha=0.3, linestyle=':')
        ax.text(gen, ax.get_ylim()[1], transition['to_phase'][:4], 
               rotation=90, va='top', fontsize=8)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_phase_space(center: ProductionGerminalCenter, ax):
    """Plot phase space diagram"""
    phase_data = center.phase_detector.get_phase_diagram_data()
    
    if phase_data and 'autocorrelation' in phase_data and 'variance' in phase_data:
        ax.scatter(phase_data['autocorrelation'], phase_data['variance'],
                  c=phase_data['phase_colors'], s=50, alpha=0.6)
        
        # Add phase boundaries
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Autocorrelation')
        ax.set_ylabel('Variance')
        ax.set_title('Phase Space')
        
        # Add phase labels
        ax.text(0.2, 0.05, 'Stable', fontsize=10, ha='center')
        ax.text(0.9, 0.05, 'Critical', fontsize=10, ha='center')
        ax.text(0.9, 0.3, 'Bifurcation', fontsize=10, ha='center')

def plot_diversity_metrics(center: ProductionGerminalCenter, ax):
    """Plot diversity evolution"""
    if not center.diversity_metrics:
        return
    
    generations = [d['generation'] for d in center.diversity_metrics]
    shannon = [d.get('shannon_index', 0) for d in center.diversity_metrics]
    simpson = [d.get('simpson_index', 0) for d in center.diversity_metrics]
    
    ax.plot(generations, shannon, 'purple', label='Shannon', linewidth=2)
    ax.plot(generations, simpson, 'orange', label='Simpson', linewidth=2)
    ax.axhline(y=CFG.shannon_entropy_target, color='red', linestyle='--', 
              alpha=0.5, label='Target')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity Index')
    ax.set_title('Population Diversity')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_gene_expression_heatmap(center: ProductionGerminalCenter, ax):
    """Heatmap of gene expression patterns"""
    # Sample cells and genes
    sample_size = min(30, len(center.population))
    sample_cells = list(center.population.values())[:sample_size]
    
    expression_matrix = []
    
    for cell in sample_cells:
        cell_expression = []
        for gene in cell.genes[:10]:  # First 10 genes
            if gene.is_active:
                # Use chromatin accessibility as proxy for expression
                expression = gene.chromatin_accessibility * (1 - gene.methylation_level)
                cell_expression.append(expression)
            else:
                cell_expression.append(0)
        
        # Pad to fixed size
        while len(cell_expression) < 10:
            cell_expression.append(0)
        
        expression_matrix.append(cell_expression)
    
    if expression_matrix:
        im = ax.imshow(expression_matrix, aspect='auto', cmap='RdYlBu_r')
        ax.set_xlabel('Gene Index')
        ax.set_ylabel('Cell Index')
        ax.set_title('Gene Expression Heatmap')
        plt.colorbar(im, ax=ax, fraction=0.046)

def plot_transposition_timeline(center: ProductionGerminalCenter, ax):
    """Timeline of transposition events"""
    if not center.transposition_events:
        return
    
    # Count events by type and generation
    event_counts = defaultdict(lambda: defaultdict(int))
    
    for event in center.transposition_events[-1000:]:  # Last 1000 events
        gen = event['generation']
        action = event['event']['action']
        event_counts[action][gen] += 1
    
    # Plot stacked area chart
    generations = sorted(set(g for counts in event_counts.values() for g in counts))
    
    jump_counts = [event_counts['jump'].get(g, 0) for g in generations]
    dup_counts = [event_counts['duplicate'].get(g, 0) for g in generations]
    inv_counts = [event_counts['invert'].get(g, 0) for g in generations]
    del_counts = [event_counts['delete'].get(g, 0) for g in generations]
    
    ax.stackplot(generations, jump_counts, dup_counts, inv_counts, del_counts,
                labels=['Jump', 'Duplicate', 'Invert', 'Delete'],
                colors=['blue', 'green', 'orange', 'red'],
                alpha=0.7)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Transposition Events')
    ax.set_title('Transposition Timeline')
    ax.legend(loc='upper left')

def plot_population_structure(center: ProductionGerminalCenter, ax):
    """Population structure visualization"""
    # Compute population statistics
    fitness_values = []
    gene_counts = []
    lineage_depths = []
    
    for cell in center.population.values():
        if cell.fitness_history:
            fitness_values.append(cell.fitness_history[-1])
        else:
            fitness_values.append(0)
        
        gene_counts.append(len([g for g in cell.genes if g.is_active]))
        lineage_depths.append(len(cell.lineage))
    
    # Create 2D histogram
    if fitness_values and gene_counts:
        h = ax.hist2d(fitness_values, gene_counts, bins=20, cmap='YlOrRd')
        plt.colorbar(h[3], ax=ax)
        
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Active Gene Count')
        ax.set_title('Population Structure')

def plot_epigenetic_landscape(center: ProductionGerminalCenter, ax):
    """Epigenetic modification landscape"""
    methylation_levels = []
    histone_states = []
    
    # Sample genes
    for cell in list(center.population.values())[:50]:
        for gene in cell.genes:
            if gene.is_active:
                methylation = gene.methylation_state.mean().item()
                methylation_levels.append(methylation)
                
                # Compute histone state
                h3k4me3 = torch.sigmoid(gene.histone_modifications[0]).item()
                h3k27me3 = torch.sigmoid(gene.histone_modifications[1]).item()
                histone_state = h3k4me3 - h3k27me3  # Active - repressive
                histone_states.append(histone_state)
    
    if methylation_levels and histone_states:
        ax.scatter(methylation_levels, histone_states, alpha=0.5, s=30)
        ax.set_xlabel('Methylation Level')
        ax.set_ylabel('Histone State (Active - Repressive)')
        ax.set_title('Epigenetic Landscape')
        ax.grid(True, alpha=0.3)

def plot_performance_summary(center: ProductionGerminalCenter, ax):
    """Summary performance metrics"""
    ax.axis('off')
    
    # Compute summary statistics
    current_gen = center.generation
    
    if center.fitness_landscape:
        current_fitness = center.fitness_landscape[-1]['mean_fitness']
        max_fitness_ever = max(d['max_fitness'] for d in center.fitness_landscape)
    else:
        current_fitness = 0
        max_fitness_ever = 0
    
    total_transpositions = len(center.transposition_events)
    
    if center.diversity_metrics:
        current_diversity = center.diversity_metrics[-1]['shannon_index']
    else:
        current_diversity = 0
    
    current_phase = center.phase_detector.current_phase
    population_size = len(center.population)
    
    # Create text summary
    summary_text = f"""
    PERFORMANCE SUMMARY
    ==================
    
    Generation: {current_gen}
    Population Size: {population_size}
    
    Fitness:
      Current Mean: {current_fitness:.4f}
      Best Ever: {max_fitness_ever:.4f}
    
    Diversity:
      Shannon Index: {current_diversity:.4f}
      Phase State: {current_phase}
    
    Evolution:
      Total Transpositions: {total_transpositions}
      Stress Level: {center.current_stress:.3f}
    
    System Health:
      GPU Utilization: {get_gpu_utilization():.1f}%
      Memory Usage: {get_memory_usage():.1f}%
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace')

def get_gpu_utilization():
    """Get current GPU utilization"""
    if torch.cuda.is_available():
        return torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 50.0
    return 0.0

def get_memory_usage():
    """Get current memory usage"""
    import psutil
    return psutil.virtual_memory().percent

def final_analysis(center: ProductionGerminalCenter, total_time: float):
    """Comprehensive final analysis"""
    print(f"\n📊 Final Analysis:")
    print(f"   Total runtime: {total_time/3600:.2f} hours")
    print(f"   Generations completed: {center.generation}")
    print(f"   Final population size: {len(center.population)}")
    
    if center.fitness_landscape:
        final_fitness = center.fitness_landscape[-1]['mean_fitness']
        max_fitness = max(d['max_fitness'] for d in center.fitness_landscape)
        print(f"   Final mean fitness: {final_fitness:.4f}")
        print(f"   Best fitness achieved: {max_fitness:.4f}")
    
    print(f"\n🧬 Evolutionary Statistics:")
    print(f"   Total transposition events: {len(center.transposition_events)}")
    
    # Count event types
    event_types = defaultdict(int)
    for event in center.transposition_events:
        event_types[event['event']['action']] += 1
    
    for action, count in event_types.items():
        print(f"   - {action}: {count}")
    
    if center.diversity_metrics:
        final_diversity = center.diversity_metrics[-1]
        print(f"\n🌈 Final Diversity:")
        print(f"   Shannon Index: {final_diversity['shannon_index']:.4f}")
        print(f"   Gene Richness: {final_diversity['gene_richness']}")
    
    print(f"\n🔄 Phase Transitions:")
    print(f"   Total transitions: {len(center.phase_detector.transition_history)}")
    for transition in center.phase_detector.transition_history[-5:]:
        print(f"   - Gen {transition['metrics'].get('generation', 0)}: "
              f"{transition['from_phase']} → {transition['to_phase']}")
    
    # Save final results
    results_path = os.path.join(CFG.save_dir, 'final_results.json')
    results = {
        'config': CFG.__dict__,
        'runtime_hours': total_time / 3600,
        'generations': center.generation,
        'final_population_size': len(center.population),
        'fitness_landscape': center.fitness_landscape,
        'diversity_metrics': center.diversity_metrics,
        'phase_transitions': [
            {
                'generation': t['metrics'].get('generation', 0),
                'from_phase': t['from_phase'],
                'to_phase': t['to_phase']
            }
            for t in center.phase_detector.transition_history
        ],
        'event_counts': dict(event_types)
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {results_path}")
    
    # Generate final visualizations
    print("\n🎨 Generating final visualizations...")
    visualize_production_state(center, center.generation)
    
    # Create summary plot
    create_summary_figure(center)
    
    print("\n✅ Simulation complete!")

def create_summary_figure(center: ProductionGerminalCenter):
    """Create comprehensive summary figure"""
    fig = plt.figure(figsize=(24, 16))
    
    # Main fitness plot
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    plot_fitness_landscape(center, ax1)
    
    # Phase diagram
    ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    plot_phase_space(center, ax2)
    
    # Diversity
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
    plot_diversity_metrics(center, ax3)
    
    # Transpositions
    ax4 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
    plot_transposition_timeline(center, ax4)
    
    # Gene expression
    ax5 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    plot_gene_expression_heatmap(center, ax5)
    
    # Summary text
    ax6 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    plot_performance_summary(center, ax6)
    
    plt.suptitle('Transposable Element AI - Complete Evolution Summary', fontsize=20)
    plt.tight_layout()
    
    save_path = os.path.join(CFG.save_dir, 'evolution_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Summary figure saved to {save_path}")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Set up environment
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
    
    # Run production simulation
    germinal_center = run_production_simulation()
    
    print("\n🎉 Production simulation completed successfully!")
    print(f"   Results directory: {CFG.save_dir}")
    print(f"   Final checkpoint: checkpoint_gen_{germinal_center.generation}.pt")

# ============================================================================
# END OF PRODUCTION IMPLEMENTATION
# ============================================================================