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
from scripts.core.stem_gene_module import  add_stem_genes_to_population
# Import from your main file
from scripts.core.production_germinal_center import (
    ProductionGerminalCenter, 
     
)
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.core.anitgen import generate_realistic_antigen
from scripts.config import cfg

quantum_gene_module = QuantumGeneModule



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
        germinal_center = ProductionGerminalCenter()
        # Manually initialize population if not using a factory
        germinal_center._initialize_population(pop_size)
        
    add_stem_genes_to_population(germinal_center, stem_ratio=0.2)

    # INTEGRATE QUANTUM DREAMS
    quantum_dream_engine = quantum_gene_module.integrate_quantum_dreams(germinal_center)
    
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