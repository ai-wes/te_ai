# scripts/core/archipelago_germinal_center.py

import torch
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.utils.detailed_logger import get_logger
from scripts.config import cfg

logger = get_logger()

class ArchipelagoGerminalCenter:
    """
    A multi-GPU aware meta-controller for managing an archipelago of germinal centers.
    Each island can be assigned to a specific GPU device.
    """
    def __init__(self, population_per_island: int = None):
        self.population_per_island = population_per_island or cfg.population_per_island
        self.island_devices = cfg.island_devices
        self.num_islands = len(self.island_devices)
        
        logger.info(f"🏝️  Initializing Archipelago with {self.num_islands} islands across devices: {self.island_devices}")
        
        self.islands = []
        for i, device_str in enumerate(self.island_devices):
            device = torch.device(device_str)
            logger.info(f"   Creating Island {i+1} on device {device}...")
            
            # *** CORRECTED LOGIC ***
            # Pass the device to the constructor. The GC will handle placing
            # its own cells on that device.
            gc = ProductionGerminalCenter(
                initial_population_size=population_per_island,
                device=device  # Pass the target device here
            )
            # We no longer call gc.to(device)
            self.islands.append(gc)

        self.generation = 0
        self.global_workspace = None
        self.executor = ThreadPoolExecutor(max_workers=self.num_islands)
        

    def _evolve_island_worker(self, island_gc, antigen_batch):
        """A wrapper function for the thread pool to handle device placement."""
        # Get the target device from the germinal center object itself
        island_device = island_gc.device
        
        # Move the input data to the island's specific GPU
        antigen_batch_device = [graph.to(island_device) for graph in antigen_batch]
        
        # Run the evolution
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
                logger.error(f"An island evolution task failed: {e}", exc_info=True)
        
        logger.info(f"   All islands have completed Generation {self.generation}.")

        if self.generation % cfg.migration_frequency == 0 and self.generation > 0:
            self._migrate()

    def _migrate(self):
        """Exchange the fittest individuals between islands, handling cross-device transfers."""
        logger.info(f"\n⛵ MIGRATION EVENT at Generation {self.generation} ⛵")
        num_migrants = cfg.num_migrants

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
                # Clone the cell (it will be created on the source device)
                new_migrant = target_island_gc.clone_pool.fast_clone(migrant_cell)
                
                # *** CRITICAL STEP: Move the cloned cell to the target island's device ***
                new_migrant = new_migrant.to(target_device)
                
                target_island_gc.population[new_migrant.cell_id] = new_migrant
            
            logger.info(f"   Island {i+1} (on {target_device}) received {len(migrants_to_add)} new cells.")
            
            
            
            
    def shutdown(self):
        """Cleanly shut down the thread pool executor."""
        self.executor.shutdown()