core/archipelago_germinal_center.py
"""
Archipelago Germinal Center
===========================

Manages multiple, isolated sub-populations ("islands") on a single GPU,
fostering parallel evolution and high genetic diversity. Periodically
facilitates migration of the fittest individuals between islands.
"""

import torch
import random
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class ArchipelagoGerminalCenter:
    """
    A meta-controller for managing an archipelago of germinal centers.
    """
    def __init__(self, num_islands: int = 3, population_per_island: int = 64):
        logger.info(f"🏝️  Initializing Archipelago with {num_islands} islands, each with {population_per_island} cells.")
        self.islands = [
            ProductionGerminalCenter(initial_population_size=population_per_island)
            for _ in range(num_islands)
        ]
        self.generation = 0
        self.global_workspace = None # To be set after initialization

    def evolve_generation(self, antigen_batch):
        """Evolves all islands for one generation in parallel."""
        self.generation += 1
        logger.info(f"\n{'='*30} ARCHIPELAGO GENERATION {self.generation} {'='*30}")

        # Evolve each island (can be parallelized with CUDA streams)
        for i, island_gc in enumerate(self.islands):
            logger.info(f"\n--- Evolving Island {i+1}/{len(self.islands)} ---")
            # Pass the shared workspace to each island for the generation
            island_gc.global_workspace = self.global_workspace
            island_gc.evolve_generation(antigen_batch)

        # Periodically trigger migration
        if self.generation % 10 == 0 and self.generation > 0:
            self._migrate()

    def _migrate(self):
        """Exchange the fittest individuals between islands."""
        logger.info(f"\n⛵ MIGRATION EVENT at Generation {self.generation} ⛵")
        num_migrants = 5  # Exchange top 5 cells

        # 1. Collect the best cells (migrants) from each island
        migrants_per_island = []
        for i, island_gc in enumerate(self.islands):
            if not island_gc.population: continue
            sorted_cells = sorted(
                island_gc.population.values(),
                key=lambda c: c.fitness_history[-1] if c.fitness_history else 0,
                reverse=True
            )
            migrants_per_island.append(sorted_cells[:num_migrants])
            logger.info(f"   Island {i+1} is sending its top {len(sorted_cells[:num_migrants])} cells.")

        if not migrants_per_island:
            logger.warning("   Migration failed: No cells to migrate.")
            return

        # 2. Distribute migrants to the *next* island in the archipelago
        for i, island_gc in enumerate(self.islands):
            # Get migrants from the previous island (circularly)
            migrants_to_add = migrants_per_island[i - 1]
            
            # Remove the worst cells from the current island to make space
            sorted_cells = sorted(
                island_gc.population.values(),
                key=lambda c: c.fitness_history[-1] if c.fitness_history else 0
            )
            for j in range(len(migrants_to_add)):
                if j < len(sorted_cells):
                    del island_gc.population[sorted_cells[j].cell_id]

            # Add the new, high-fitness migrants by cloning them
            for migrant_cell in migrants_to_add:
                new_migrant = island_gc.clone_pool.fast_clone(migrant_cell)
                island_gc.population[new_migrant.cell_id] = new_migrant
            
            logger.info(f"   Island {i+1} received {len(migrants_to_add)} new cells.")
