
#symbiotic_swarm_main.py
"""
Symbiotic Swarm Main Execution Script
=====================================

This script initializes and runs the new hybrid TE-AI architecture,
demonstrating the Archipelago model, Global Workspace, and Intelligent
Gene Recombination in action.
"""

import torch
import numpy as np
from typing import List
import time
from scripts.config import cfg
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.advanced.global_workspace import GlobalWorkspace
from scripts.domains.drug_discovery.deepchem_converter import DeepChemToTEAI
from scripts.core.utils.detailed_logger import get_logger
from scripts.core.advanced.archipelago_germinal_center import ArchipelagoGerminalCenter
import random
logger = get_logger()

def run_symbiotic_swarm_test(
    dataset_name: str = cfg.dataset_name,
    generations: int = cfg.generations,
    population_per_island: int = cfg.population_per_island
):
    """
    Runs a full test of the Symbiotic Swarm architecture.
    """
    logger.info(f"\n{'='*60}")
    logger.info("INITIALIZING SYMBIOTIC SWARM TEST")
    logger.info(f"{'='*60}\n")

    # 1. Load and Prepare Data
    logger.info(f"Loading '{dataset_name}' dataset...")
    converter = DeepChemToTEAI()
    train_antigens, _, _ = converter.convert_molnet_dataset(dataset_name)
    logger.info(f"Loaded {len(train_antigens)} training samples.")

    # 2. Initialize the Core Architecture
    # The Archipelago is the main controller
    archipelago = ArchipelagoGerminalCenter(
        population_per_island=population_per_island
    )
    
    # The Global Workspace is a shared resource for all islands
    global_workspace = GlobalWorkspace()
    archipelago.global_workspace = global_workspace

    # 3. Run the Cultivation Loop
    logger.info("\n--- Starting Cultivation Loop ---")
    start_time = time.time()

    for generation in range(generations):
        # Sample a batch of antigens for this generation's challenge
        antigen_batch_domain_objects = random.sample(train_antigens, k=min(32, len(train_antigens)))
        graph_batch = [target.to_graph() for target in antigen_batch_domain_objects]

        # Evolve the entire archipelago for one generation
        archipelago.evolve_generation(graph_batch)

    end_time = time.time()
    logger.info(f"\n--- Cultivation Complete ---")
    logger.info(f"Total time for {generations} generations: {end_time - start_time:.2f} seconds.")
    
    # 4. Final Report
    print("\n" + "="*60)
    print("SYMBIOTIC SWARM FINAL REPORT")
    print("="*60)
    for i, island_gc in enumerate(archipelago.islands):
        final_fitness = island_gc.fitness_landscape[-1]['mean_fitness'] if island_gc.fitness_landscape else "N/A"
        final_diversity = island_gc.diversity_metrics[-1]['shannon_index'] if island_gc.diversity_metrics else "N/A"
        print(f"Island {i+1}:")
        print(f"  Final Population Size: {len(island_gc.population)}")
        print(f"  Final Mean Fitness: {final_fitness:.4f}")
        print(f"  Final Diversity (Shannon): {final_diversity:.4f}")

if __name__ == "__main__":
    run_symbiotic_swarm_test()
