# scripts/core/archipelago_germinal_center.py

import torch
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from .production_germinal_center import ProductionGerminalCenter
from .utils.detailed_logger import get_logger
from scripts.config import cfg

logger = get_logger()

class ArchipelagoGerminalCenter:
    """
    A multi-GPU aware meta-controller for managing an archipelago of germinal centers.
    This version uses a ThreadPoolExecutor to evolve islands in parallel.
    """
    def __init__(self, population_per_island: int = 64):
        self.island_devices = cfg.island_devices
        self.num_islands = len(self.island_devices)
        
        logger.info(f"🏝️  Initializing Archipelago with {self.num_islands} islands across devices: {self.island_devices}")
        
        self.islands = []
        for i, device_str in enumerate(self.island_devices):
            device = torch.device(device_str)
            logger.info(f"   Creating Island {i+1} on device {device}...")
            gc = ProductionGerminalCenter(
                initial_population_size=population_per_island,
                device=device
            )
            self.islands.append(gc)

        self.generation = 0
        self.global_workspace = None
        self.executor = ThreadPoolExecutor(max_workers=self.num_islands)

    def _evolve_island_worker(self, island_gc, antigen_batch):
        """A wrapper function for the thread pool to handle device placement."""
        island_device = island_gc.device
        antigen_batch_device = [graph.to(island_device) for graph in antigen_batch]
        
        if island_gc.global_workspace is not None:
            logger.info(f"   Island on {island_device} has access to the Global Workspace.")
        else:
            logger.warning(f"   Island on {island_device} DOES NOT have access to the Global Workspace.")
            
        return island_gc.evolve_generation(antigen_batch_device)

    def evolve_generation(self, antigen_batch):
        """Evolves all islands for one generation in parallel across their assigned GPUs."""
        self.generation += 1
        logger.info(f"\n{'='*30} ARCHIPELAGO GENERATION {self.generation} {'='*30}")

        futures = []
        for i, island_gc in enumerate(self.islands):
            logger.info(f"   Submitting evolution task for Island {i+1} on {self.island_devices[i]}...")
            island_gc.global_workspace = self.global_workspace
            future = self.executor.submit(self._evolve_island_worker, island_gc, antigen_batch)
            futures.append(future)

        logger.info("   All island tasks submitted. Waiting for completion...")
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                logger.error(f"An island evolution task failed: {e}\nTRACEBACK:\n{tb_str}")
        
        logger.info(f"   All islands have completed Generation {self.generation}.")

        if self.generation % 10 == 0 and self.generation > 0:
            self._migrate()

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
            logger.info(f"   Island {i+1} (on {island_gc.device}) is sending its top {len(sorted_cells[:num_migrants])} cells.")

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
            
            logger.info(f"   Island {i+1} (on {target_device}) received {len(migrants_to_add)} new cells.")

    def shutdown(self):
        """Cleanly shut down the thread pool executor."""
        self.executor.shutdown()