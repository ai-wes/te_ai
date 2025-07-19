# scripts/core/advanced/symbiotic_swarm_main.py

import torch
import random
import time

# Local project imports
from scripts.core.advanced.archipelago_germinal_center import ArchipelagoGerminalCenter
from scripts.core.advanced.global_workspace import GlobalWorkspace
from scripts.domains.drug_discovery.deepchem_converter import DeepChemToTEAI
from scripts.core.utils.detailed_logger import get_logger
from scripts.config import cfg

logger = get_logger()

def run_symbiotic_swarm_test():
    """
    Main function to set up and run the Symbiotic Swarm test.
    Encapsulating this in a function ensures all objects are local and garbage
    collected after the run, preventing state leakage between runs.
    """
    logger.info(f"\n{'='*60}")
    logger.info("INITIALIZING NEW SYMBIOTIC SWARM TEST RUN")
    logger.info(f"{'='*60}\n")

    # 1. Load and Prepare Data
    logger.info(f"Loading '{cfg.dataset_name}' dataset...")
    converter = DeepChemToTEAI()
    train_antigens, _, _ = converter.convert_molnet_dataset(cfg.dataset_name)
    logger.info(f"Loaded {len(train_antigens)} training samples.")

    # 2. Initialize the Core Architecture
    # This creates NEW instances every time the function is called.
    archipelago = ArchipelagoGerminalCenter(
        population_per_island=cfg.population_per_island
    )
    
    # Reset the archipelago to ensure clean state
    archipelago.reset()
    
    global_workspace = GlobalWorkspace()
    archipelago.global_workspace = global_workspace
    # --- NEW: Establish Baselines before starting the evolution ---
    # We need a sample batch of data to run the evaluation
    logger.info("Establishing baselines...")
    baseline_antigen_objects = random.sample(train_antigens, k=min(32, len(train_antigens)))
    logger.info(f"Sampled {len(baseline_antigen_objects)} baseline antigens.")
    baseline_graph_batch = [target.to_graph() for target in baseline_antigen_objects]
    logger.info(f"Converted to {len(baseline_graph_batch)} baseline graphs.")
    archipelago.establish_baselines(baseline_graph_batch)
    
    
    # 3. Run the Cultivation Loop
    logger.info("\n--- Starting Cultivation Loop ---")
    start_time = time.time()

    try:
        for generation in range(cfg.generations):
            antigen_batch_domain_objects = random.sample(train_antigens, k=min(32, len(train_antigens)))
            graph_batch = [target.to_graph() for target in antigen_batch_domain_objects]
            archipelago.evolve_generation(graph_batch)
    finally:
        logger.info("Shutting down the Archipelago's thread pool executor...")
        archipelago.shutdown()

    end_time = time.time()
    logger.info(f"\n--- Cultivation Complete ---")
    logger.info(f"Total time for {cfg.generations} generations: {end_time - start_time:.2f} seconds.")
    
    # 4. Final Report
    print("\n" + "="*60)
    print("SYMBIOTIC SWARM FINAL REPORT")
    print("="*60)
    for i, island_gc in enumerate(archipelago.islands):
        specialization = list(cfg.island_specializations.keys())[i]
        final_fitness = island_gc.fitness_landscape[-1]['mean_fitness'] if island_gc.fitness_landscape else "N/A"
        final_diversity = island_gc.diversity_metrics[-1]['shannon_index'] if island_gc.diversity_metrics else "N/A"
        champion_fitness = island_gc.hall_of_fame['fitness']
        
        print(f"Island {i+1} ({specialization.upper()}):")
        print(f"  Final Population Size: {len(island_gc.population)}")
        print(f"  Final Mean Fitness: {final_fitness:.4f}")
        print(f"  Final Champion Fitness: {champion_fitness:.4f}")
        print(f"  Final Diversity (Shannon): {final_diversity:.4f}")




# This is the standard Python entry point.
if __name__ == "__main__":
   

    run_symbiotic_swarm_test()