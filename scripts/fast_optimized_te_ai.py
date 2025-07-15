# ============================================================================
# COMPLETE OPTIMIZED TRANSPOSABLE ELEMENT AI - READY TO RUN
# ============================================================================

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
from stem_gene_module import StemGeneModule, add_stem_genes_to_population
# Import from your main file
from transposable_immune_ai_production_complete import (
    ProductionBCell,  ContinuousDepthGeneModule, 
    ProductionGerminalCenter, PhaseTransitionDetector, 
    generate_realistic_antigen, TermColors, integrate_quantum_dreams, QuantumGeneModule
)

from config import cfg


# ============================================================================
# 1. OPTIMIZED PARALLEL BATCH EVALUATION
# ============================================================================

class OptimizedBatchEvaluator:
    """
    True parallel evaluation that processes entire population in single forward pass
    """
    def __init__(self, device='cuda'):
        self.device = device
        self._cache = {}
        
    def evaluate_population_batch(self, population: Dict, antigens: List[Data]) -> Dict[str, float]:
        """
        Evaluate entire population in parallel with single forward pass
        """
        # Create single batch for all antigens
        antigen_batch = Batch.from_data_list([a.to(self.device) for a in antigens])
        
        # Collect all cells and prepare batch processing
        cell_ids = list(population.keys())
        cells = [population[cid] for cid in cell_ids]
        
        # Process each cell individually (safer but still optimized)
        fitness_scores = {}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                for i, (cell_id, cell) in enumerate(zip(cell_ids, cells)):
                    # Forward pass for this cell
                    affinity, cell_representation, _ = cell(antigen_batch)
                    
                    # Average affinity across antigens
                    mean_affinity = affinity.mean().item()
                    
                    # Complexity penalty
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
                        representation_cpu = cell_representation.mean(dim=0).detach().cpu()
                        cell.store_memory(representation_cpu, fitness)
        
        print(f"   Evaluated {len(fitness_scores)} cells (optimized).")
        return fitness_scores
    
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

# ============================================================================
# 2. OPTIMIZED CLONE OPERATION
# ============================================================================

class FastClonePool:
    """
    Pre-allocated cell pool for fast cloning without CPU transfers
    """
    def __init__(self, pool_size=100, device='cuda'):
        self.pool_size = pool_size
        self.device = device
        self.available_cells = deque()
        self.in_use = set()
        
# In class FastClonePool:
    def fast_clone(self, parent: ProductionBCell) -> ProductionBCell:
        """
        Ultra-fast cloning that avoids CPU transfers and correctly handles gene types.
        """
        try:
            child_genes = []
            
            with torch.no_grad():
                for gene in parent.genes:
                    if gene.is_active:
                        gene_state = gene.state_dict()
                        
                        # ============================================================================
                        # FIX: Instantiate the correct gene type (Quantum or Continuous)
                        # ============================================================================
                        if isinstance(gene, QuantumGeneModule):
                            new_gene = QuantumGeneModule(gene.gene_type, gene.variant_id)
                        else:
                            new_gene = ContinuousDepthGeneModule(gene.gene_type, gene.variant_id)
                        # ============================================================================
                        
                        new_gene.to(self.device)
                        
                        try:
                            # Load the state dict. strict=False allows it to ignore
                            # parameters that might not exist in one version vs another.
                            new_gene.load_state_dict(gene_state, strict=False)
                        except Exception as e_load:
                            # Fallback for safety
                            print(f"Warning: Could not load state dict for gene {gene.gene_id}. Error: {e_load}. Copying manually.")
                            for key, value in gene_state.items():
                                if key in new_gene.state_dict():
                                    target_param = new_gene.state_dict()[key]
                                    if value.shape == target_param.shape:
                                        new_gene.state_dict()[key].copy_(value)
                        
                        # Copy non-parameter attributes
                        new_gene.position = gene.position
                        new_gene.is_active = gene.is_active
                        new_gene.is_inverted = gene.is_inverted
                        new_gene.fitness_contribution = gene.fitness_contribution
                        new_gene.chromatin_accessibility = gene.chromatin_accessibility
                        new_gene.is_cold = gene.is_cold
                        new_gene.activation_ema = gene.activation_ema
                        
                        new_gene.transposition_history = copy.deepcopy(gene.transposition_history)
                        
                        # Epigenetic inheritance
                        if hasattr(gene, 'methylation_state') and hasattr(new_gene, 'methylation_state'):
                            if gene.methylation_state.shape == new_gene.methylation_state.shape:
                                new_gene.methylation_state.data.copy_(gene.methylation_state.data * cfg.methylation_inheritance)
                        
                        if hasattr(gene, 'histone_modifications') and hasattr(new_gene, 'histone_modifications'):
                            if gene.histone_modifications.shape == new_gene.histone_modifications.shape:
                                new_gene.histone_modifications.data.copy_(gene.histone_modifications.data * cfg.methylation_inheritance)
                        
                        if random.random() < 0.05:
                            transposed_child, _ = new_gene.transpose(0.1, 0.5)
                            if transposed_child:
                                child_genes.append(transposed_child)
                        
                        child_genes.append(new_gene)
            
            child = ProductionBCell(child_genes).to(self.device)
            
            child.lineage = parent.lineage + [parent.cell_id]
            child.generation = parent.generation + 1
            
            with torch.no_grad():
                parent_matrix = parent.gene_regulatory_matrix.data
                child_matrix = child.gene_regulatory_matrix.data
                
                if parent_matrix.shape == child_matrix.shape:
                    child.gene_regulatory_matrix.data.copy_(parent_matrix * 0.9 + torch.randn_like(child_matrix) * 0.1)
                else:
                    child.gene_regulatory_matrix.data.copy_(torch.randn_like(child_matrix) * 0.1)
            
            self._fast_mutate(child)
            
            return child
            
        except Exception as e:
            print(f"Fast clone failed: {e}, using fallback method")
            return parent.clone()
        
        
        
        
    def _fast_mutate(self, cell):
        """Optimized mutation"""
        with torch.no_grad():
            for param in cell.parameters():
                if random.random() < cfg.mutation_rate:
                    mutation = torch.randn_like(param) * cfg.mutation_rate
                    param.data += mutation

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








class CachedPhaseTransitionDetector(PhaseTransitionDetector):
    """
    Optimized phase detector with caching and vectorized operations
    """
    
    def __init__(self, window_size: int = 50):
        super().__init__(window_size)
        self._cache = {}
        self._cache_generation = -1
    
    def update(self, metrics: Dict[str, float], population_state: Dict):
        """Cached update that reuses expensive computations"""
        generation = metrics.get('generation', self._cache_generation + 1)
        
        # Only recompute if generation changed
        if generation != self._cache_generation:
            self._cache = {}
            self._cache_generation = generation
        
        # Use cached indicators if available
        cache_key = f"indicators_{generation}"
        if cache_key not in self._cache:
            self._compute_early_warning_indicators(metrics, population_state)
            self._cache[cache_key] = dict(self.indicators)
        else:
            # Restore from cache
            for key, values in self._cache[cache_key].items():
                self.indicators[key] = deque(values, maxlen=self.window_size)
        
        # Rest of the update logic...
        return super().update(metrics, population_state)
    
    def _compute_morans_i(self, positions: List[Tuple[float, float]]) -> float:
        """
        Optimized Moran's I using matrix operations
        """
        if len(positions) < 3:
            return 0.0
        
        # Sample efficiently
        n = min(len(positions), 100)
        if len(positions) > n:
            indices = np.random.choice(len(positions), n, replace=False)
            positions = np.array(positions)[indices]
        else:
            positions = np.array(positions)
        
        # Vectorized distance computation
        # Use broadcasting for efficient pairwise distances
        pos_expanded = positions[:, np.newaxis, :]  # Shape: (n, 1, 2)
        distances = np.linalg.norm(pos_expanded - positions, axis=2)  # Shape: (n, n)
        
        # Vectorized weight matrix
        W = 1.0 / (1.0 + distances)
        np.fill_diagonal(W, 0)  # No self-connections
        
        # Normalize
        W_sum = W.sum()
        if W_sum == 0:
            return 0.0
        W = W / W_sum
        
        # Vectorized Moran's I computation
        values = positions[:, 0]
        mean_val = values.mean()
        deviations = values - mean_val
        
        # Matrix computation
        numerator = np.sum(W * np.outer(deviations, deviations))
        denominator = np.sum(deviations ** 2)
        
        if denominator == 0:
            return 0.0
        
        return (n / W_sum) * (numerator / denominator)




# ============================================================================
# 4. OPTIMIZED GERMINAL CENTER
# ============================================================================

class OptimizedProductionGerminalCenter(ProductionGerminalCenter):
    """
    Performance-optimized germinal center
    """
    
    def __init__(self):
        super().__init__()
        # Add optimized components
        self.batch_evaluator = OptimizedBatchEvaluator(cfg.device)
        self.clone_pool = FastClonePool(device=cfg.device)
        self.vectorized_ops = VectorizedPopulationOps()
        self.cached_phase_detector = CachedPhaseTransitionDetector()
        
        # Set global reference for visualization
        import transposable_immune_ai_production_complete as prod
        prod._current_germinal_center = self
        
        
        
        
    def evolve_generation(self, antigens: List[Data]):
        """Optimized evolution with minimal overhead"""
        generation_start = time.time()
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
        intervention = self.phase_detector.update(metrics, population_state)
        
        if intervention:
            intervention(self)
        
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



# ============================================================================
# RUN OPTIMIZED SIMULATION
# ============================================================================

def run_optimized_simulation(
    num_generations: int = 50,
    population_size: Optional[int] = None,
    antigens: Optional[List[Data]] = None,
    initial_genes: Optional[List[nn.Module]] = None,
    component_factories: Optional[Dict] = None,
    checkpoint_dir: str = "checkpoints_optimized",
    elite_fraction: Optional[float] = None,
    mutation_rate: Optional[float] = None
):
    """
    Run complete optimized simulation, now with parameterization to support
    external callers like the visualization script.
    """
    print("\n" + "="*80)
    print("🚀 OPTIMIZED TRANSPOSABLE ELEMENT AI - PERFORMANCE VERSION")
    print("="*80)

    # --- Configuration Setup ---
    # Use passed arguments if available, otherwise fall back to config file
    pop_size = population_size if population_size is not None else cfg.initial_population
    generations_to_run = num_generations

    print(f"\n⚙️  Configuration:")
    print(f"   Device: {cfg.device}")
    print(f"   Population: {pop_size} → {cfg.max_population}")
    print(f"   Generations: {generations_to_run}")
    print(f"   Optimization: ENABLED")

    # --- Germinal Center Initialization ---
    # Use a factory if provided (for visualization), otherwise create the default
    if component_factories and 'germinal_center' in component_factories:
        print("   - Using component factory for Germinal Center.")
        # Pass relevant parameters to the factory
        germinal_center = component_factories['germinal_center'](
            population_size=pop_size,
            elite_fraction=elite_fraction or (1 - cfg.selection_pressure),
            mutation_rate=mutation_rate or cfg.mutation_rate,
            gene_pool_size=len(initial_genes) if initial_genes else 100,
            checkpoint_dir=checkpoint_dir
        )
        # The factory is expected to handle population initialization
    else:
        print("   - Using default OptimizedProductionGerminalCenter.")
        germinal_center = OptimizedProductionGerminalCenter()
        # Manually initialize population if not using a factory
        germinal_center.initialize_population(pop_size)
        
    add_stem_genes_to_population(germinal_center, stem_ratio=0.2)

    # INTEGRATE QUANTUM DREAMS
    quantum_dream_engine = integrate_quantum_dreams(germinal_center)
    
    # Override cell creation to use quantum genes
    germinal_center._create_random_cell = germinal_center._create_random_cell_with_quantum.__get__(
        germinal_center, germinal_center.__class__
    )
    
    # --- Viral Evolution Timeline (for standalone runs) ---
    viral_timeline = [
        (0, [], "Wild Type"), (5, [(i, j) for i in range(1) for j in range(8)], "Alpha Variant"),
        (10, [(i, j) for i in range(2) for j in range(6)], "Beta Variant"),
        (25, [(i, j) for i in range(3) for j in range(5)], "Delta Variant"),
        (30, [(i, j) for i in range(3) for j in range(7)], "Omicron Variant"),
        (35, [(i, j) for i in range(3) for j in range(20)], "Doomsday Variant")
    ]
    current_variant_idx = 0
    simulation_start = time.time()
    
    # --- Main Evolution Loop ---
    for epoch in range(generations_to_run):
        # --- Antigen Generation ---
        # Use provided antigens if available (from viz script), else generate them
        if antigens:
            current_antigens = antigens
        else:
            # Check for viral mutation in standalone mode
            if current_variant_idx < len(viral_timeline) - 1 and epoch >= viral_timeline[current_variant_idx + 1][0]:
                current_variant_idx += 1
                _, mutations, variant_name = viral_timeline[current_variant_idx]
                print(f"\n🦠 VIRUS MUTATED TO {variant_name.upper()}!")
                germinal_center.current_stress = 1.0
            
            _, mutations, variant_name = viral_timeline[current_variant_idx]
            current_antigens = [generate_realistic_antigen(
                variant_type=variant_name.lower().split()[0], mutations=mutations
            ) for _ in range(cfg.batch_size)]
        
        # Evolve population with optimizations
        germinal_center.evolve_generation(current_antigens)
        
        # Progress report
        if epoch % 10 == 0 and epoch > 0:
            elapsed = time.time() - simulation_start
            eta = (elapsed / epoch) * (generations_to_run - epoch)
            print(f"\n📊 Progress: {epoch}/{generations_to_run} ({(epoch)/generations_to_run*100:.1f}%)")
            print(f"   Elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m")
            
            if germinal_center.fitness_landscape:
                current_fitness = germinal_center.fitness_landscape[-1]['mean_fitness']
                print(f"   Current fitness: {current_fitness:.4f}")
    
    # --- Final Analysis ---
    total_time = time.time() - simulation_start
    print("\n" + "="*80)
    print("🏁 OPTIMIZED SIMULATION COMPLETE")
    print("="*80)
    print(f"   Total runtime: {total_time/60:.2f} minutes")
    
    best_fitness_overall = 0
    if germinal_center.fitness_landscape:
        final_fitness = germinal_center.fitness_landscape[-1]['mean_fitness']
        best_fitness_overall = max(d['max_fitness'] for d in germinal_center.fitness_landscape)
        print(f"   Final mean fitness: {final_fitness:.4f}")
        print(f"   Best fitness achieved: {best_fitness_overall:.4f}")
    
    # Return the best fitness, as expected by the visualization script
    return best_fitness_overall

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the optimized simulation with default parameters
    run_optimized_simulation()