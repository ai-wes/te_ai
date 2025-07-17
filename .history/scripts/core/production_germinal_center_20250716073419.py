
import matplotlib
matplotlib.use('Agg')

import os
import torch
from torch_geometric.data import Data, Batch
import numpy as np
import random
import copy
import uuid
from collections import defaultdict, deque
from typing import List, Dict
import os
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy import stats
import hashlib
from scripts.config import cfg
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.core.production_b_cell import ProductionBCell
from scripts.core.phase_transition_detector import PhaseTransitionDetector
from scripts.core.dreamer import DreamConsolidationEngine
from scripts.core.utils.telemetry import TermColors
from scripts.core.ode import ContinuousDepthGeneModule
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()



# Complete ProductionGerminalCenter
# ============================================================================

class ProductionGerminalCenter:
    """Production-ready population manager with all features"""
    
    def __init__(self):
        from scripts.core.parallel_batch_evaluation import OptimizedBatchEvaluator
        from scripts.core.clone_operation import FastClonePool
        from scripts.core.population_operations import VectorizedPopulationOps
        
        self.population: Dict[str, ProductionBCell] = {}
        self.generation = 0
        self.current_stress = 0.0
        
        # History tracking
        self.fitness_landscape = []
        self.diversity_metrics = []
        self.transposition_events = []
        self.phase_transitions = []
        
        # Advanced systems
        self.dream_engine = DreamConsolidationEngine().to(cfg.device)
        self.phase_detector = PhaseTransitionDetector()
        self.plasmid_pool = deque(maxlen=200)
        
        # Scheduled tasks
        self.scheduled_tasks = []
        
        # Performance optimization
        self.gpu_cache = {}
        self.parallel_executor = ThreadPoolExecutor(max_workers=cfg.num_workers)


        self.mutation_log = deque(maxlen=500)
        self.input_batch_history = deque(maxlen=500)
        
        
        # Initialize population
        self._initialize_population()
        
        # Mixed precision training
        if cfg.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()


        self.max_mutation_tokens = 50.0 # B
        self.mutation_tokens = self.max_mutation_tokens
        self.token_refill_rate = self.max_mutation_tokens / 60.0 # Per generation (proxy for per minute)
        self.mutation_costs = {'jump': 1, 'duplicate': 2, 'invert': 1, 'delete': -0.5} # Deletion gives back tokens

        self._parallel_batch_cache = None
        self._cached_cell_ids_hash = None
        # Add optimized components
        self.batch_evaluator = OptimizedBatchEvaluator(cfg.device)
        self.clone_pool = FastClonePool(device=cfg.device)
        self.vectorized_ops = VectorizedPopulationOps()
        self.cached_phase_detector = PhaseTransitionDetector()
        
        # Set global reference for visualization
        import scripts.depricated.transposable_immune_ai_production_complete as prod
        prod._current_germinal_center = self
        

    
    def _initialize_population(self):
        logger.debug("Entering ProductionGerminalCenter._initialize_population")
        """Create initial diverse population with proper stem cell representation"""
        print(f"\n🧬 Initializing production population with {cfg.initial_population} cells...")
        
        # Create dedicated stem cells (20% of initial population)
        num_stem_cells = int(cfg.initial_population * 0.2)
        print(f"   Creating {num_stem_cells} dedicated stem cells...")
        
        for i in range(num_stem_cells):
            cell = self._create_stem_cell()
            self.population[cell.cell_id] = cell
        
        # Create remaining cells with mixed types
        remaining_cells = cfg.initial_population - num_stem_cells
        print(f"   Creating {remaining_cells} mixed-type cells...")
        
        for i in range(remaining_cells):
            cell = self._create_random_cell()
            self.population[cell.cell_id] = cell
            
            if (i + 1) % 50 == 0:
                print(f"   Created {i + 1}/{remaining_cells} mixed cells...")
        
        print(f"✅✅  Population initialized with {len(self.population)} cells ({num_stem_cells} stem cells + {remaining_cells} mixed)  ✅✅ ")
    
    
    def _create_random_cell(self) -> ProductionBCell:
        """Create cell with random gene configuration"""
        genes = []
        
        # For early generations (0-2), create more stem cells
        is_early_generation = self.generation <= 2
        stem_cell_probability = 0.6 if is_early_generation else 0.3
        
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
        
        # --- NEW: Small chance to add a Quantum Gene ---
        if random.random() < 0.1: # 10% chance for a new cell to have a quantum gene
            print("   ✨✨ A Quantum Gene has emerged!  ✨✨")
            q_gene = QuantumGeneModule('Q', random.randint(1, 5))
            q_gene.position = random.random() # Place it anywhere
            genes.append(q_gene)
        
        # --- NEW: Stem cells initialization ---
        # Higher chance for stem genes in early generations (representing pluripotent potential)
        if random.random() < stem_cell_probability:
            num_s = random.randint(1, 3)  # 1-3 stem genes
            for _ in range(num_s):
                try:
                    # Try to use StemGeneModule if available, otherwise use ContinuousDepthGeneModule
                    try:
                        from stem_gene_module import StemGeneModule
                        s_gene = StemGeneModule('S', random.randint(1, 20))
                    except ImportError:
                        s_gene = ContinuousDepthGeneModule('S', random.randint(1, 20))
                    s_gene.position = np.clip(np.random.normal(0.5, 0.2), 0, 1.0)  # Can be anywhere
                    s_gene.log_depth.data = torch.tensor(np.random.normal(0, 0.15))  # Higher stability
                    genes.append(s_gene)
                except:
                    # Fallback to ContinuousDepthGeneModule if StemGeneModule fails
                    s_gene = ContinuousDepthGeneModule('S', random.randint(1, 20))
                    s_gene.position = np.clip(np.random.normal(0.5, 0.2), 0, 1.0)
                    s_gene.log_depth.data = torch.tensor(np.random.normal(0, 0.15))
                    genes.append(s_gene)
        
        return ProductionBCell(genes).to(cfg.device)
    
    
    def _create_stem_cell(self) -> ProductionBCell:
        """Create a dedicated stem cell with majority stem genes"""
        genes = []
        
        # Stem cells start with more S genes (2-4)
        num_s = random.randint(2, 4)
        for _ in range(num_s):
            try:
                try:
                    from stem_gene_module import StemGeneModule
                    s_gene = StemGeneModule('S', random.randint(1, 20))
                except ImportError:
                    s_gene = ContinuousDepthGeneModule('S', random.randint(1, 20))
                s_gene.position = np.clip(np.random.normal(0.5, 0.2), 0, 1.0)
                s_gene.log_depth.data = torch.tensor(np.random.normal(0, 0.1))  # More stable
                genes.append(s_gene)
            except:
                s_gene = ContinuousDepthGeneModule('S', random.randint(1, 20))
                s_gene.position = np.clip(np.random.normal(0.5, 0.2), 0, 1.0)
                s_gene.log_depth.data = torch.tensor(np.random.normal(0, 0.1))
                genes.append(s_gene)
        
        # Add a few other gene types for versatility (but fewer than S genes)
        for gene_type in ['V', 'D', 'J']:
            if random.random() < 0.7:  # 70% chance for each type
                if gene_type == 'V':
                    gene = ContinuousDepthGeneModule('V', random.randint(1, 100))
                    gene.position = np.clip(np.random.normal(0.15, 0.1), 0, 0.3)
                elif gene_type == 'D':
                    gene = ContinuousDepthGeneModule('D', random.randint(1, 50))
                    gene.position = np.clip(np.random.normal(0.45, 0.1), 0.3, 0.6)
                else:  # J
                    gene = ContinuousDepthGeneModule('J', random.randint(1, 10))
                    gene.position = np.clip(np.random.normal(0.8, 0.1), 0.6, 1.0)
                
                gene.log_depth.data = torch.tensor(np.random.normal(0, 0.2))
                genes.append(gene)
        
        return ProductionBCell(genes).to(cfg.device)
    
    
        
    
    def _add_random_individuals(self, count: int):
        logger.debug("Entering ProductionGerminalCenter._add_random_individuals")
        """Add new random individuals to population"""
        for _ in range(count):
            if len(self.population) < cfg.max_population:
                cell = self._create_random_cell()
                self.population[cell.cell_id] = cell
                
                
    
    
    def evolve_generation(self, antigens: List[Data]):
        logger.debug("Entering ProductionGerminalCenter.evolve_generation")
        """Complete evolution cycle with all systems"""
        generation_start = time.time()
        self.generation += 1
        
        print(f"\n{'='*80}")
        print(f"GENERATION {self.generation}")
        print(f"{'='*80}")
        
        # --- Store input history for replay/HGT ---
        self.input_batch_history.append([a.to('cpu') for a in antigens])
        
        print("\n📊 Phase 1: Fitness Evaluation")
        fitness_scores = self._evaluate_population_parallel(antigens)
        
        # Phase 2: Compute metrics and detect stress
        print("\n📈 Phase 2: Metrics and Stress Detection")
        metrics = self._compute_comprehensive_metrics(fitness_scores)
        self.current_stress = self._detect_population_stress(metrics)
        
        # --- CHANGE 1: FORCE STRESS AT GENERATION 3 ---
        if self.generation == 3:
            print("\n🔥 DEBUG: Forcing maximum stress at Generation 3.")
            self.current_stress = 1.0
        print(f"   Current stress level: {self.current_stress:.3f}")
                
        # Phase 3: Phase transition detection and intervention
        print("\n🔍🔍  Phase 3: Phase Transition Analysis  🔍🔍")
        population_state = self._get_population_state()
        intervention = self.phase_detector.update(metrics, population_state)
        
        if intervention:
            intervention(self)
        
        # Phase 4: Stress response
        if self.current_stress > cfg.stress_threshold:
            print(f"\n⚠️⚠️   Phase 4: High Stress Response (stress={self.current_stress:.3f})  ⚠️ ⚠️ ")
            self._execute_stress_response()
        
        # # The code seems to be a comment in a Python script, indicating that it is part of Phase 5
        # which involves selection and reproduction. It is likely describing a specific phase or
        # step in a larger program or project.
        #Phase 5: Selection and reproduction
        print("\n🧬🧬 Phase 5: Selection and Reproduction 🧬🧬")
        self._selection_and_reproduction(fitness_scores)
    
        if self.generation % 10 == 0: # Every 15 generations, try to entangle
            print("\n🌀🌀 Entanglement Phase  🌀🌀")
            for cell in self.population.values():
                if hasattr(cell, 'attempt_entanglement'):
                    cell.attempt_entanglement()
                    
                    
        # Phase 6: Dream consolidation (periodic)
        if self.generation % 5 == 0:
            print("\n💤 Phase 6: Dream Consolidation")
            self._execute_dream_phase()
        
        # Phase 7: Record and visualize
        self._record_generation_data(metrics, time.time() - generation_start)
        
        # Execute scheduled tasks
        self._execute_scheduled_tasks()
        
        # --- FINAL STEP: Memory Cleanup and Logging ---
        
        # Optional: Explicitly clear large variables from this generation's scope
        del fitness_scores, metrics, population_state
        
        # Run cleanup every few generations to balance performance and memory
        if self.generation % 2 == 0: 
            import gc
            
            # Suggest to Python's garbage collector to run
            gc.collect()
            
            # Tell PyTorch to release all unused cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Log memory usage AFTER cleanup
                mem_after_cleanup = torch.cuda.memory_allocated() / 1e9
                print(f"   - Cleared CUDA memory cache. Usage now: {mem_after_cleanup:.2f} GB")

        # Final generation summary log
        gen_time = time.time() - generation_start
        print(f"\n⏱️  Generation {self.generation} completed in {gen_time:.2f}s. Population: {len(self.population)}")
        
        
        
            
# In the ProductionGerminalCenter class:

    
    def _evaluate_population_parallel(self, antigens: List[Data]) -> Dict[str, float]:
        """
        True parallel GPU evaluation of the population.
        MODIFIED: Processes the population in batches to manage memory, but evaluates
                  each cell independently within the batch to prevent cross-talk and
                  ensure accurate, distinct fitness scores.
        """
        antigen_batch = Batch.from_data_list([a.to(cfg.device) for a in antigens])
        fitness_scores = {}
        
        cell_ids = list(self.population.keys())
        num_batches = (len(cell_ids) + cfg.gpu_batch_size - 1) // cfg.gpu_batch_size
        
        with torch.no_grad(): # No gradients needed for fitness evaluation
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * cfg.gpu_batch_size
                    end_idx = min((batch_idx + 1) * cfg.gpu_batch_size, len(cell_ids))
                    batch_cell_ids = cell_ids[start_idx:end_idx]

                    for cell_id in batch_cell_ids:
                        cell = self.population[cell_id]
                        
                        # Each cell processes the entire batch of antigens independently.
                        # This is parallel at the antigen level, which is highly efficient on GPU.
                        affinity, cell_representation, _ = cell(antigen_batch)
                        
                        # Average affinity across the antigen batch
                        mean_affinity = affinity.mean().item()
                        
                        # Compute fitness with complexity penalty
                        active_genes = len([g for g in cell.genes if g.is_active])
                        complexity_penalty = max(0, active_genes - 10) * cfg.duplication_cost
                        
                        # Diversity bonus
                        diversity_bonus = self._compute_cell_diversity(cell) * cfg.diversity_weight
                        
                        fitness = mean_affinity - complexity_penalty + diversity_bonus
                        fitness_scores[cell_id] = fitness
                        
                        # Update cell records
                        cell.fitness_history.append(fitness)
                        for gene in cell.genes:
                            if gene.is_active:
                                gene.fitness_contribution = fitness
                        
                        # Store successful responses
                        if fitness > 0.8:
                            # Move the representation to CPU before storing to prevent leaks
                            representation_cpu = cell_representation.mean(dim=0).detach().cpu()
                            cell.store_memory(representation_cpu, fitness)

        print(f"   Evaluated {len(fitness_scores)} cells in {num_batches} batches.")
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
        
        # Stem gene metrics
        stem_metrics = self._compute_stem_gene_metrics()
        metrics.update(stem_metrics)
        
        # --- MODIFIED PRINT STATEMENT ---
        print(
            f"   {TermColors.BOLD}📊 Metrics:{TermColors.RESET} "
            f"{TermColors.CYAN}💪 Fitness: {metrics['mean_fitness']:.3f} ± {metrics['fitness_variance']:.3f}{TermColors.RESET}, "
            f"{TermColors.MAGENTA}🌿 Diversity: {metrics['shannon_index']:.3f}{TermColors.RESET}, "
            f"{TermColors.YELLOW}🧬 Genes: {metrics['unique_genes']}{TermColors.RESET}"
        )
        
        # Print stem gene metrics if present
        if metrics.get('stem_gene_count', 0) > 0:
            print(
                f"   {TermColors.BOLD}🌱 Stem Genes:{TermColors.RESET} "
                f"Count: {metrics['stem_gene_count']}, "
                f"Differentiations: {metrics['differentiation_events']}, "
                f"Avg Commitment: {metrics['avg_commitment']:.3f}"
            )
        
        return metrics

    

    # In ProductionGerminalCenter class:

    
    def _compute_population_diversity(self) -> Dict[str, float]:
        """Compute multiple diversity metrics.
        MODIFIED: Optimized with vectorized operations.
        """
        active_genes = [
            (f"{gene.gene_type}{gene.variant_id}", gene.position)
            for cell in self.population.values()
            for gene in cell.genes if gene.is_active
        ]
        
        if not active_genes:
            return {'shannon_index': 0, 'simpson_index': 0, 'position_entropy': 0, 'gene_richness': 0}

        gene_types, all_positions = zip(*active_genes)
        
        # Use collections.Counter for fast counting
        from collections import Counter
        gene_type_counts = Counter(gene_types)
        total_genes = len(gene_types)
        
        # Vectorized Shannon and Simpson indices
        counts_array = np.array(list(gene_type_counts.values()))
        probabilities = counts_array / total_genes
        
        shannon = -np.sum(probabilities * np.log(probabilities))
        simpson = 1 - np.sum(probabilities**2)
        
        # Vectorized positional entropy
        hist, _ = np.histogram(all_positions, bins=20, range=(0, 1))
        hist_prob = hist / hist.sum()
        position_entropy = -np.sum(hist_prob[hist_prob > 0] * np.log(hist_prob[hist_prob > 0]))
        
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
    
    
    def _compute_stem_gene_metrics(self) -> Dict[str, float]:
        """Compute metrics for stem genes in the population"""
        stem_gene_count = 0
        differentiation_events = 0
        commitment_levels = []
        emergency_differentiations = 0
        stem_fitness_contributions = []

        for cell in self.population.values():
            for gene in cell.genes:
                # Check if this is a StemGeneModule
                if hasattr(gene, 'commitment_level'):
                    stem_gene_count += 1
                    commitment_levels.append(gene.commitment_level)

                    # Count differentiation events
                    if hasattr(gene, 'differentiation_history'):
                        differentiation_events += len(gene.differentiation_history)

                    # Count emergency differentiations
                    if hasattr(gene, 'emergency_differentiation_count'):
                        emergency_differentiations += gene.emergency_differentiation_count

                    # Measure contribution to fitness
                    if hasattr(cell, 'fitness_history') and len(cell.fitness_history) > 0:
                        stem_fitness_contributions.append(cell.fitness_history[-1])

        avg_commitment = np.mean(commitment_levels) if commitment_levels else 0.0
        stem_contribution = np.mean(stem_fitness_contributions) if stem_fitness_contributions else 0.0

        return {
            'stem_gene_count': stem_gene_count,
            'differentiation_events': differentiation_events,
            'avg_commitment': avg_commitment,
            'emergency_differentiations': emergency_differentiations,
            'stem_contribution_to_fitness': stem_contribution
        }
    
# In the ProductionGerminalCenter class:

    
    def _detect_population_stress(self, metrics: Dict[str, float]) -> float:
        """Sophisticated stress detection
        MODIFIED: Made more sensitive to fitness drops and amplified.
        """
        # --- MODIFICATION: Remove the hair-trigger ---
        # The hair-trigger is too sensitive for this phase of evolution.
        # We will now rely on the more robust trend analysis.
        # if len(self.fitness_landscape) > 1:
        #     current_fitness = self.fitness_landscape[-1]['mean_fitness']
        #     previous_fitness = self.fitness_landscape[-2]['mean_fitness']
        #     if current_fitness < previous_fitness:
        #         print("   Hair-trigger stress detected due to fitness drop!")
        #         return 1.0

        stress_factors = []
        # Factor 1: Tightened stagnation detection - earlier "panic" transposition
        STAGNATION_WINDOW = 5        # Reduced from cfg.stress_window
        STAGNATION_DELTA = 0.002     # Trigger if mean fitness improves < 0.2%
        
        if len(self.fitness_landscape) >= STAGNATION_WINDOW:
            recent_fitness = [f['mean_fitness'] for f in self.fitness_landscape[-STAGNATION_WINDOW:]]
            mean_fitness = recent_fitness[-1]
            
            # Check for improvement over stagnation window
            delta = mean_fitness - self.fitness_landscape[-STAGNATION_WINDOW]['mean_fitness']
            
            stagnation_stress = 0.0
            if abs(delta) < STAGNATION_DELTA:
                stagnation_stress = 1.0  # Full stress for stagnation
                print(f"   Stagnation detected: Δfitness={delta:.4f} < {STAGNATION_DELTA}. Triggering panic transposition!")
            else:
                # Also check for sustained decline
                slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(recent_fitness)), recent_fitness)
                if slope < 0 and p_value < 0.05:
                    stagnation_stress = max(0, -slope * 100)
                    print(f"   Sustained fitness decline detected (p={p_value:.3f}). Stress: {stagnation_stress:.2f}")

            stress_factors.append(stagnation_stress)
        # Factor 2: Low diversity
        diversity_stress = max(0, cfg.shannon_entropy_target - metrics['shannon_index'])
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
        stress = np.mean(stress_factors) if stress_factors else 0.0
        
        # HACK APPLIED: Amplify the calculated stress
        stress = min(1.0, stress * 2.0)
        
        # --- CORRECTED PRINTING LOGIC ---
        if len(stress_factors) == 4:
            # This happens after generation cfg.stress_window
            print(f"   Stress factors: stagnation={stress_factors[0]:.2f}, "
                  f"diversity={stress_factors[1]:.2f}, "
                  f"variance={stress_factors[2]:.2f}, "
                  f"phase={stress_factors[3]:.2f}")
        elif len(stress_factors) == 3:
            # This happens in early generations (no stagnation factor)
            print(f"   Stress factors: stagnation=N/A, "
                  f"diversity={stress_factors[0]:.2f}, "
                  f"variance={stress_factors[1]:.2f}, "
                  f"phase={stress_factors[2]:.2f}")
        else:
            # Fallback for any other case
            print(f"   Stress factors: {stress_factors}")

        print(f"   Combined amplified stress: {stress:.3f}")
        
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
        logger.debug("Entering ProductionGerminalCenter._execute_stress_response")
        """
        Comprehensive stress response, incorporating a token budget for mutations,
        epigenetic modifications, and horizontal gene transfer.
        
        Mitigations Integrated:
        - Mitigation #3 (Mutation-Budget Token Bucket): Controls the rate of mutations.
        - Mitigation #5 (Deterministic Graph-Diff): Logs all successful mutations.
        """
        print("   Executing stress response protocols:")
        
        # 1. Increase transposition rate (with token budget)
        # ========================================================================
        print("   • Triggering transposition cascade (budgeted)")
        
        # Refill tokens at the start of the stress response
        self.mutation_tokens = min(self.max_mutation_tokens, self.mutation_tokens + self.token_refill_rate)
        
        transposition_attempts = 0
        transposition_success = 0
        
        # Use the last known diversity metric, or a default if history is empty
        diversity_metric = self.diversity_metrics[-1]['shannon_index'] if self.diversity_metrics else 0.5
        
        for cell in list(self.population.values()):
            new_genes = []
            for gene in list(cell.genes):
                if not gene.is_active:
                    continue
                
                # The transpose method now returns the action taken
                child, action = gene.transpose(self.current_stress, diversity_metric)
                
                if action:
                    transposition_attempts += 1
                    cost = self.mutation_costs.get(action, 1)
                    
                    # Check if there are enough tokens in the budget
                    if self.mutation_tokens >= cost:
                        self.mutation_tokens -= cost
                        transposition_success += 1
                        
                        # Log the successful, budgeted mutation (Mitigation #5)
                        log_entry = {
                            'id': str(uuid.uuid4()),
                            'generation': self.generation,
                            'cell_id': cell.cell_id,
                            'gene_id': gene.gene_id,
                            'action': action,
                            'stress_level': self.current_stress,
                            'parent_gene_hash': hashlib.sha256(str(gene.state_dict()).encode()).hexdigest(),
                        }
                        self.mutation_log.append(log_entry)
                        
                        # If a duplication occurred, add the new gene
                        if child:
                            new_genes.append(child)
                    # else: Token budget exceeded, mutation is blocked.
            
            # Add any newly created (and paid for) genes to the cell
            for new_gene in new_genes:
                if len(cell.genes) < cfg.max_genes_per_clone:
                    cell.genes.append(new_gene)
        
        print(f"     - {transposition_success}/{transposition_attempts} mutations executed. Tokens remaining: {self.mutation_tokens:.1f}")

        # 2. Epigenetic stress response
        # ========================================================================
        print("   • Applying epigenetic modifications")
        epigenetic_count = 0
        # Limit to a subset for efficiency
        for cell in list(self.population.values())[:100]:
            for gene in cell.genes:
                if gene.is_active and random.random() < cfg.methylation_rate * self.current_stress:
                    sites = torch.randint(0, cfg.hidden_dim, (5,), device=gene.methylation_state.device)
                    gene.add_methylation(sites, self.current_stress * 0.5)
                    epigenetic_count += 1
        
        print(f"     - {epigenetic_count} genes epigenetically modified.")

        # 3. Horizontal gene transfer
        # ========================================================================
        print("   • Facilitating horizontal gene transfer")
        
        # Fetch the last input batch to use for signature calculation (Mitigation #4)
        if self.input_batch_history:
            calibration_antigens = [a.to(cfg.device) for a in self.input_batch_history[-1]]
            calibration_batch = Batch.from_data_list(calibration_antigens)
            transfer_count = self._execute_horizontal_transfer(calibration_batch)
            print(f"     - {transfer_count} successful gene transfers.")
        else:
            print("     - Skipping HGT (no input history for calibration).")

        # 4. Inject diversity if critically low
        # ========================================================================
        if self.diversity_metrics and self.diversity_metrics[-1]['shannon_index'] < 0.5:
            print("   • Injecting new diverse individuals due to low diversity.")
            self._add_random_individuals(50)
            
        




    
    def _execute_horizontal_transfer(self, calibration_batch: Data) -> int:
        """
        Execute horizontal gene transfer between cells, with signature-based compatibility checks.
        
        Mitigations Integrated:
        - Mitigation #4 (Feature-Signature Handshake): Ensures compatibility before gene transfer.
        """
        transfer_count = 0
        
        # --- 1. Extract Plasmids from High-Fitness Donors ---
        
        # Determine the fitness threshold (e.g., top 30% of the population)
        all_fitness_scores = [
            cell.fitness_history[-1] for cell in self.population.values() if cell.fitness_history
        ]
        if not all_fitness_scores:
            return 0 # Cannot proceed without fitness scores
            
        fitness_threshold = np.percentile(all_fitness_scores, 70)
        
        # Donor cells release plasmids into the shared pool
        for cell in self.population.values():
            if cell.fitness_history and cell.fitness_history[-1] > fitness_threshold:
                # The extract_plasmid method needs to be updated to add the signature
                
                # First, ensure the cell has a signature
                if not hasattr(cell, '_signature_cache') or cell._signature_cache is None:
                    cell.get_signature(calibration_batch) # This will compute and cache it
                
                # Now, extract the plasmid
                plasmid = cell.extract_plasmid()
                if plasmid:
                    # Add the signature to the plasmid for the handshake
                    plasmid['signature'] = cell._signature_cache
                    self.plasmid_pool.append(plasmid)

        if not self.plasmid_pool:
            return 0 # No plasmids were created

        # --- 2. Recipient Cells Attempt to Integrate Plasmids ---
        
        # Select a random subset of the population to be potential recipients
        recipient_cells = random.sample(
            list(self.population.values()),
            min(100, len(self.population)) # Limit to 100 attempts for efficiency
        )
        
        for cell in recipient_cells:
            # Check if this cell will attempt to take up a plasmid
            if random.random() < cfg.horizontal_transfer_prob * (1 + self.current_stress):
                # Pick a random plasmid from the pool
                plasmid_to_integrate = random.choice(list(self.plasmid_pool))
                
                # The integrate_plasmid method performs the handshake
                if cell.integrate_plasmid(plasmid_to_integrate, calibration_batch):
                    transfer_count += 1
                    print(f"   - Successful HGT: Cell {cell.cell_id[:8]} integrated plasmid from {plasmid_to_integrate['donor_cell'][:8]}")

        return transfer_count


    
    def _selection_and_reproduction(self, fitness_scores: Dict[str, float]):
        logger.debug("Entering ProductionGerminalCenter._selection_and_reproduction")
        """
        Natural selection with multiple strategies.
        MODIFIED: Uses a memory-efficient 'recycling' strategy to prevent OOM errors.
        """
        if not fitness_scores:
            return

        sorted_cells = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Identify survivors and those to be eliminated
        num_survivors = int(len(sorted_cells) * (1 - cfg.selection_pressure))
        survivor_ids = {cid for cid, _ in sorted_cells[:num_survivors]}
        eliminated_ids = [cid for cid, _ in sorted_cells[num_survivors:]]
        
        # Identify parents for the next generation (the top survivors)
        parents = [self.population[cid] for cid in survivor_ids]
        if not parents: # Edge case if all cells are eliminated
            parents = [self.population[sorted_cells[0][0]]]

        print(f"   Selection complete: {len(survivor_ids)} survivors, {len(eliminated_ids)} to be recycled.")

        # Recycle eliminated cells into children of survivors
        recycled_count = 0
        for i, cell_id_to_recycle in enumerate(eliminated_ids):
            # Pick a parent cyclically from the survivor pool
            parent = parents[i % len(parents)]
            
            # Get the cell object to be recycled
            recycled_cell = self.population[cell_id_to_recycle]
            
            # Use the in-place recycle method
            recycled_cell.recycle_as_child(parent)
            recycled_count += 1

        print(f"   Recycled {recycled_count} cells as new offspring.")
        
        # The population dictionary itself remains the same size, but its contents are updated.
        # We just need to handle the case where the population needs to grow.
        current_pop_size = len(self.population)
        while current_pop_size < cfg.max_population and current_pop_size < len(fitness_scores) * 1.5:
            parent = random.choice(parents)
            child = parent.clone() # Use the old clone method just for population growth
            self.population[child.cell_id] = child
            current_pop_size += 1
            if current_pop_size >= cfg.max_population:
                break
        
        print(f"   New population size: {len(self.population)}")
        
        # Clear the cache to force a rebuild with the new (recycled) cell states
        self._parallel_batch_cache = None
        self._cached_cell_ids_hash = None
        print("   - Parallel batch cache cleared.")
    
        
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
                combined_score = fitness + diversity * cfg.niche_pressure
                
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
        child = ProductionBCell(all_genes[:cfg.max_genes_per_clone])
        # Move the child to the correct device BEFORE operating on its parameters.
        child = child.to(cfg.device)
        
        child.lineage = [parent1.cell_id, parent2.cell_id]
        
        # Combine regulatory matrices
        with torch.no_grad():
            # Now all tensors in this operation are on the same device (cfg.device)
            child.gene_regulatory_matrix.data = \
                (parent1.gene_regulatory_matrix.data + parent2.gene_regulatory_matrix.data) / 2 + \
                torch.randn_like(child.gene_regulatory_matrix) * 0.1
        
        # The .to(cfg.device) at the end is now redundant but harmless. We can remove it.
        return child
    
    
    def _execute_dream_phase(self):
        logger.debug("Entering ProductionGerminalCenter._execute_dream_phase")
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
        self.dream_engine.dream_phase(self.population, cfg.dream_cycles_per_generation)
    
    
    def _record_generation_data(self, metrics: Dict[str, float], generation_time: float):
        logger.debug("Entering ProductionGerminalCenter._record_generation_data")
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
        if self.generation % cfg.checkpoint_interval == 0:
            self._save_checkpoint()
    
    
    def _execute_scheduled_tasks(self):
        logger.debug("Entering ProductionGerminalCenter._execute_scheduled_tasks")
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
        logger.debug("Entering ProductionGerminalCenter._save_checkpoint")
        """Save population checkpoint"""
        checkpoint_path = os.path.join(cfg.save_dir, f'checkpoint_gen_{self.generation}.pt')
        
        checkpoint = {
            'generation': self.generation,
            'config': cfg.__dict__,
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
        
        # Save detailed architecture for the #1 elite cell
        if sorted_cells:
            elite_cell_id, elite_cell = sorted_cells[0]
            if hasattr(elite_cell, 'architecture_modifier'):
                mod = elite_cell.architecture_modifier
                checkpoint['elite_architecture'] = {
                    'cell_id': elite_cell_id,
                    'lineage': getattr(elite_cell, 'lineage', []),
                    'dynamic_modules': list(mod.dynamic_modules.keys()),
                    'connections': {k: list(v) for k, v in mod.module_connections.items()},
                    'dna': getattr(mod, 'architecture_dna', 'unknown'),
                    'quantum_genes': sum(1 for gene in elite_cell.genes if isinstance(gene, QuantumGeneModule))
                }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"   💾 Saved checkpoint to {checkpoint_path}")



    
    def evolve_generation(self, antigens: List[Data]):
        """Optimized evolution with minimal overhead"""
        generation_start = time.time()
        
        # Get logger instance
        self.generation += 1
        
        print(f"\n{'='*80}")
        print(f"GENERATION {self.generation}")
        print(f"{'='*80}")
        
        # Store input history for replay/HGT
        self.input_batch_history.append([a.to('cpu') for a in antigens])
        
        # Phase 1: Fast fitness evaluation
        print("\n📊 Phase 1: Fitness Evaluation (Optimized)")
        fitness_scores = self.batch_evaluator.evaluate_population_batch(
            self.population, antigens
        )
        
        # Phase 2: Compute metrics and detect stress
        print("\n📈 Phase 2: Metrics and Stress Detection")
        metrics = self._compute_comprehensive_metrics(fitness_scores)
        self.current_stress = self._detect_population_stress(metrics)
        
        # Force stress at generation 3 (from original)
        if self.generation == 3:
            print("\n🔥 DEBUG: Forcing maximum stress at Generation 3.")
            self.current_stress = 1.0
        print(f"   Current stress level: {self.current_stress:.3f}")
        
        # Phase 3: Phase transition detection
        print("\n🔍🔍  Phase 3: Phase Transition Analysis  🔍🔍")
        population_state = self._get_population_state()
        intervention_func = self.phase_detector.update(metrics, population_state)
        
        if intervention_func:
            intervention_func(self)
        
        # Phase 4: Stress response
        if self.current_stress > cfg.stress_threshold:
            print(f"\n⚠️⚠️   Phase 4: High Stress Response (stress={self.current_stress:.3f})  ⚠️ ⚠️ ")
            self._execute_stress_response()
        
        # Phase 5: Fast selection and reproduction
        print("\n🧬🧬 Phase 5: Selection and Reproduction (Optimized) 🧬🧬")
        self._selection_and_reproduction_fast(fitness_scores)
        
        # Phase 6: Dream consolidation (periodic)
        if self.generation % 5 == 0:
            print("\n💤 Phase 6: Dream Consolidation")
            self._execute_dream_phase()
        
        # Record and visualize
        self._record_generation_data(metrics, time.time() - generation_start)
        
        # Execute scheduled tasks
        self._execute_scheduled_tasks()
        
        # Memory cleanup
        if self.generation % 2 == 0:
            self._cleanup_memory()
        
        gen_time = time.time() - generation_start
        print(f"\n⏱️  Generation {self.generation} completed in {gen_time:.2f}s. Population: {len(self.population)}")
    
    
    def _selection_and_reproduction_fast(self, fitness_scores: Dict[str, float]):
        logger.debug("Entering ProductionGerminalCenter._selection_and_reproduction_fast")
        """Optimized selection using fast cloning"""
        if not fitness_scores:
            return
        
        sorted_cells = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Identify survivors and those to be eliminated
        num_survivors = int(len(sorted_cells) * (1 - cfg.selection_pressure))
        survivor_ids = {cid for cid, _ in sorted_cells[:num_survivors]}
        eliminated_ids = [cid for cid, _ in sorted_cells[num_survivors:]]
        
        # Get parent cells
        parents = [self.population[cid] for cid in survivor_ids]
        if not parents:
            parents = [self.population[sorted_cells[0][0]]]
        
        print(f"   Selection complete: {len(survivor_ids)} survivors, {len(eliminated_ids)} to be replaced.")
        
        # Replace eliminated cells with fast clones
        replaced_count = 0
        for i, cell_id_to_replace in enumerate(eliminated_ids):
            parent = parents[i % len(parents)]
            
            # Use fast clone
            child = self.clone_pool.fast_clone(parent)
            
            # Replace in population
            del self.population[cell_id_to_replace]
            self.population[child.cell_id] = child
            replaced_count += 1
        
        print(f"   Replaced {replaced_count} cells with optimized cloning.")
        
        # Handle population growth if needed
        current_pop_size = len(self.population)
        while current_pop_size < cfg.max_population and current_pop_size < len(fitness_scores) * 1.5:
            parent = random.choice(parents)
            child = self.clone_pool.fast_clone(parent)
            self.population[child.cell_id] = child
            current_pop_size += 1
            if current_pop_size >= cfg.max_population:
                break
        
        print(f"   New population size: {len(self.population)}")
        
        # Clear cache
        self._parallel_batch_cache = None
        self._cached_cell_ids_hash = None
        print("   - Parallel batch cache cleared.")
    
    
    def _cleanup_memory(self):
        logger.debug("Entering ProductionGerminalCenter._cleanup_memory")
        """Aggressive memory cleanup"""
        # Clear old history
        for cell in self.population.values():
            if len(cell.fitness_history) > 50:
                cell.fitness_history = deque(list(cell.fitness_history)[-50:], maxlen=100)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_usage = torch.cuda.memory_allocated() / 1e9
            print(f"   - Cleared CUDA memory cache. Usage now: {mem_usage:.2f} GB")





    
    def _create_random_cell_with_quantum(self) -> ProductionBCell:
        """Create cell with random gene configuration, including quantum genes"""
        genes = []
        
        # V genes (variable region)
        num_v = random.randint(1, 4)
        for _ in range(num_v):
            # 20% chance of quantum gene
            if random.random() < 0.2:
                gene = QuantumGeneModule('V', random.randint(1, 100))
                print("   ✨✨ A Quantum Gene has emerged!  ✨✨")
            else:
                gene = ContinuousDepthGeneModule('V', random.randint(1, 100))
            
            gene.position = np.clip(np.random.normal(0.15, 0.1), 0, 0.3)
            gene.log_depth.data = torch.tensor(np.random.normal(0, 0.3))
            genes.append(gene)
        
        # D genes (diversity region)
        num_d = random.randint(1, 3)
        for _ in range(num_d):
            # 20% chance of quantum gene
            if random.random() < 0.2:
                gene = QuantumGeneModule('D', random.randint(1, 50))
                print("   ✨✨ A Quantum Gene has emerged!  ✨✨")
            else:
                gene = ContinuousDepthGeneModule('D', random.randint(1, 50))
            
            gene.position = np.clip(np.random.normal(0.45, 0.1), 0.3, 0.6)
            gene.log_depth.data = torch.tensor(np.random.normal(0, 0.2))
            genes.append(gene)
        
        # J genes (joining region)
        num_j = random.randint(1, 2)
        for _ in range(num_j):
            # 20% chance of quantum gene
            if random.random() < 0.2:
                gene = QuantumGeneModule('J', random.randint(1, 10))
                print("   ✨✨ A Quantum Gene has emerged!  ✨✨")
            else:
                gene = ContinuousDepthGeneModule('J', random.randint(1, 10))
            
            gene.position = np.clip(np.random.normal(0.8, 0.1), 0.6, 1.0)
            gene.log_depth.data = torch.tensor(np.random.normal(0, 0.2))
            genes.append(gene)
        
        return ProductionBCell(genes).to(cfg.device)

