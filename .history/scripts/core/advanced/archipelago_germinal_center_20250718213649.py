# scripts/core/archipelago_germinal_center.py

import torch
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.utils.detailed_logger import get_logger
from scripts.config import cfg
from scripts.core.advanced.breeder_gene import BreederGene
from scripts.core.advanced.fitness_functions import FITNESS_FUNCTIONS # <-- Import the new functions
from scripts.core.production_b_cell import ProductionBCell
import numpy as np
logger = get_logger()

class ArchipelagoGerminalCenter:
    def __init__(self, population_per_island: int = None):
        self.population_per_island = population_per_island or cfg.population_per_island
        self.island_specializations = cfg.island_specializations
        self.num_islands = len(self.island_specializations)
        
        logger.info(f"🏝️  Initializing Specialist Archipelago with {self.num_islands} islands:")
        
        self.islands = []
        self.island_fitness_fns = []
        self.island_names = []  # Track island names for better logging
        
        for specialization, device_str in self.island_specializations.items():
            if specialization not in FITNESS_FUNCTIONS:
                raise ValueError(f"Unknown specialization '{specialization}' in config. Must be one of {list(FITNESS_FUNCTIONS.keys())}")
            
            device = torch.device(device_str)
            
            # Define island emoji and description
            island_emoji = {
                'accuracy': '🧪',  # Scientist
                'speed': '⚡',     # Engineer  
                'balanced': '⚖️'   # Generalist
            }.get(specialization, '🏝️')
            
            island_desc = {
                'accuracy': 'SCIENTIST (Accuracy-focused)',
                'speed': 'ENGINEER (Speed-focused)',
                'balanced': 'GENERALIST (Balanced)'
            }.get(specialization, specialization.upper())
            
            logger.info(f"   {island_emoji} Creating Island '{island_desc}' on device {device}...")
            
            gc = ProductionGerminalCenter(
                initial_population_size=self.population_per_island,
                device=device
            )
            # Store the island type in the germinal center for logging
            gc.island_type = specialization
            gc.island_emoji = island_emoji
            gc.island_desc = island_desc
            
            self.islands.append(gc)
            self.island_fitness_fns.append(FITNESS_FUNCTIONS[specialization])
            self.island_names.append(specialization)

        self.generation = 0
        self.global_workspace = None
        self.executor = ThreadPoolExecutor(max_workers=self.num_islands)
        self.breeder = BreederGene()


    def establish_baselines(self, antigen_batch):
        """
        Evaluates an initial population against all fitness functions to establish
        a performance baseline for each specialization.
        """
        logger.info(f"\n{'='*30} 📊 ESTABLISHING PERFORMANCE BASELINES 📊 {'='*30}")
        
        # Use the first island's population as the "random" sample
        baseline_island = self.islands[0]
        baseline_population = baseline_island.population
        antigen_batch_device = [graph.to(baseline_island.device) for graph in antigen_batch]

        print("\n" + "="*60)
        print("  INITIAL POPULATION BASELINE REPORT (GENERATION 0)")
        print("="*60)
        
        for specialization, fitness_fn in FITNESS_FUNCTIONS.items():
            # Evaluate the same random population against this specific fitness function
            fitness_scores = baseline_island.batch_evaluator.evaluate_population_batch(
                baseline_population, antigen_batch_device, fitness_fn
            )
            
            if fitness_scores:
                mean_fitness = np.mean(list(fitness_scores.values()))
                max_fitness = np.max(list(fitness_scores.values()))
                print(f"  - Baseline for '{specialization.upper()}' Fitness:  Avg = {mean_fitness:.4f},  Best = {max_fitness:.4f}")
            else:
                print(f"  - Baseline for '{specialization.upper()}' Fitness:  Evaluation failed.")
        
        print("="*60 + "\n")
        logger.info("   Baselines established. Starting evolution.")



    def reset(self):
            """Resets the entire Archipelago, including all islands, to a clean state."""
            logger.info("Resetting the entire Archipelago for a new run...")
            self.generation = 0
            for island in self.islands:
                island.reset()



    def _evolve_island_worker(self, island_gc, antigen_batch, fitness_function):
        """Wrapper for the thread pool that passes the specialized fitness function."""
        island_device = island_gc.device
        antigen_batch_device = [graph.to(island_device) for graph in antigen_batch]
        return island_gc.evolve_generation(antigen_batch_device, fitness_function)

    def evolve_generation(self, antigen_batch):
        self.generation += 1
        logger.info(f"\n{'='*30} ARCHIPELAGO GENERATION {self.generation} {'='*30}")

        futures = []
        future_to_island = {}  # Map futures to island info for better logging
        
        for i, island_gc in enumerate(self.islands):
            specialization = self.island_names[i]
            fitness_fn = self.island_fitness_fns[i]
            logger.info(f"   {island_gc.island_emoji} Submitting task for '{island_gc.island_desc}' Island on {island_gc.device}...")
            future = self.executor.submit(self._evolve_island_worker, island_gc, antigen_batch, fitness_fn)
            futures.append(future)
            future_to_island[future] = (i, island_gc)

        logger.info("   All island tasks submitted. Waiting for completion...")
        
        # Process results as they complete
        for future in as_completed(futures):
            island_idx, island_gc = future_to_island[future]
            try:
                metrics = future.result()
                # Log island-specific results
                logger.info(f"\n   {island_gc.island_emoji} Island '{island_gc.island_desc}' completed:")
                logger.info(f"      • Mean Fitness: {metrics.get('mean_fitness', 0):.3f}")
                logger.info(f"      • Champion Fitness: {island_gc.hall_of_fame.get('fitness', 0):.3f}")
                
                # Log specialized metrics based on island type
                if island_gc.island_type == 'speed':
                    # For speed island, we care about inference time
                    logger.info(f"      • Focus Metric (Speed): Population optimizing for minimal inference time")
                elif island_gc.island_type == 'accuracy':
                    logger.info(f"      • Focus Metric (Accuracy): Population optimizing for prediction accuracy")
                else:  # balanced
                    logger.info(f"      • Focus Metric (Balanced): Population balancing speed and accuracy")
                    
            except Exception as e:
                import traceback
                logger.error(f"   Island '{island_gc.island_desc}' evolution task failed: {e}\n{traceback.format_exc()}")
        
        logger.info(f"\n   All islands have completed Generation {self.generation}.")

        if self.generation % cfg.breeding_summit_frequency == 0 and self.generation > 0:
            self._run_breeding_summit()
        
        if self.generation % 10 == 0 and self.generation > 0:
            self._migrate()
            
            
# In scripts/core/archipelago_germinal_center.py

    def _run_breeding_summit(self):
        """
        Identifies the champion cell from each island and breeds them to create
        "Perfect Spawn" offspring, which are then injected into other islands.
        """
        # This import is needed here to create the new cell
        from scripts.core.production_b_cell import ProductionBCell

        logger.info(f"\n{'='*25} 🏆 ARCHIPELAGO BREEDING SUMMIT at Generation {self.generation} 🏆 {'='*25}")
        
        if self.num_islands < 2:
            logger.warning("   Breeding Summit requires at least 2 islands. Skipping.")
            return

        # 1. Identify the champion of each island
        champions = []
        for i, island_gc in enumerate(self.islands):
            champion = island_gc.hall_of_fame.get("champion")
            champions.append(champion)
            
            if champion:
                fitness = island_gc.hall_of_fame.get("fitness", "N/A")
                # Assuming you add island_emoji and island_desc to your ProductionGerminalCenter
                logger.info(f"   Champion from Island {i+1}: Cell {champion.cell_id[:8]} (Fitness: {fitness:.4f}) on {island_gc.device}")
            else:
                logger.warning(f"   Island {i+1} has no champion in its Hall of Fame yet.")
                
        # 2. Breed champions in a round-robin fashion
        for i in range(self.num_islands):
            parent1 = champions[i]
            parent2 = champions[(i + 1) % self.num_islands]
            target_island_index = (i + 2) % self.num_islands
            target_island_gc = self.islands[target_island_index]
            target_device = target_island_gc.device

            if parent1 is None or parent2 is None:
                logger.warning(f"   Skipping breeding for pair ({i+1}, {(i+1)%self.num_islands+1}) due to missing champion.")
                continue

            logger.info(f"\n   Breeding Champions from Island {i+1} and Island {(i + 1) % self.num_islands + 1}...")
            
            p1_gene = next((g for g in parent1.genes if g.is_active), None)
            p2_gene = next((g for g in parent2.genes if g.is_active), None)

            if p1_gene and p2_gene:
                
                # *** THIS IS THE FIX ***
                # We no longer need temporary gene copies.
                # We pass the original genes and the target_device directly to the breeder.
                # The breeder's recombine function will handle all device placements internally.
                new_gene = self.breeder.recombine(p1_gene, p2_gene, target_device)
                
                # Create the "Perfect Spawn" cell. The new_gene is already on the correct target_device.
                perfect_spawn_cell = ProductionBCell([new_gene])
                
                # 3. Inject the Perfect Spawn into the target island
                if target_island_gc.population:
                    worst_cell = min(
                        target_island_gc.population.values(),
                        key=lambda c: c.fitness_history[-1] if c.fitness_history else 0
                    )
                    del target_island_gc.population[worst_cell.cell_id]
                    logger.info(f"   Injecting Perfect Spawn {perfect_spawn_cell.cell_id[:8]} into Island {target_island_index + 1} (on {target_device}).")

                    # Check if the worst_cell has a fitness history before logging it.
                    if worst_cell.fitness_history:
                        logger.info(f"   It replaces the worst cell {worst_cell.cell_id[:8]} (Fitness: {worst_cell.fitness_history[-1]:.4f}).")
                    else:
                        logger.info(f"   It replaces the worst cell {worst_cell.cell_id[:8]} (which had no fitness history).")
                
                target_island_gc.population[perfect_spawn_cell.cell_id] = perfect_spawn_cell
            else:
                logger.warning("   Breeding failed: one or both parents lacked an active gene.")                
                
    def _migrate(self):
        """Exchange the fittest individuals between islands, handling cross-device transfers."""
        logger.info(f"\n⛵ MIGRATION EVENT at Generation {self.generation} ⛵")
        num_migrants = 5

        migrants_per_island = []
        for i, island_gc in enumerate(self.islands):
            if not island_gc.population: continue
            sorted_cells = sorted(
                island_gc.population.values(),
                key=lambda c: c.fitness_history[-1] if c.fitness_history else 0,
                reverse=True
            )
            migrants_per_island.append(sorted_cells[:num_migrants])
            logger.info(f"   {island_gc.island_emoji} {island_gc.island_desc} (on {island_gc.device}) is sending its top {len(sorted_cells[:num_migrants])} cells.")

        if not migrants_per_island:
            logger.warning("   Migration failed: No cells to migrate.")
            return

        for i, target_island_gc in enumerate(self.islands):
            target_device = target_island_gc.device
            migrants_to_add = migrants_per_island[i - 1]
            
            sorted_cells = sorted(
                target_island_gc.population.values(),
                key=lambda c: c.fitness_history[-1] if c.fitness_history else 0
            )
            for j in range(len(migrants_to_add)):
                if j < len(sorted_cells):
                    del target_island_gc.population[sorted_cells[j].cell_id]

            for migrant_cell in migrants_to_add:
                new_migrant = target_island_gc.clone_pool.fast_clone(migrant_cell)
                new_migrant = new_migrant.to(target_device)
                target_island_gc.population[new_migrant.cell_id] = new_migrant
            
            logger.info(f"   {target_island_gc.island_emoji} {target_island_gc.island_desc} (on {target_device}) received {len(migrants_to_add)} new cells.")

    def shutdown(self):
        """Cleanly shut down the thread pool executor."""
        self.executor.shutdown()