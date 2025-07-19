# scripts/core/archipelago_germinal_center.py

import torch
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.utils.detailed_logger import get_logger
from scripts.config import cfg
from scripts.core.advanced.breeder_gene import BreederGene


logger = get_logger()

class ArchipelagoGerminalCenter:
    """
    A multi-GPU aware meta-controller for managing an archipelago of germinal centers.
    This version uses a ThreadPoolExecutor to evolve islands in parallel.
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
            
            # *** CORRECTED CONSTRUCTOR CALL ***
            # Pass the specific population size for this island.
            gc = ProductionGerminalCenter(
                initial_population_size=self.population_per_island,
                device=device
            )
            self.islands.append(gc)

        self.generation = 0
        self.global_workspace = None
        self.executor = ThreadPoolExecutor(max_workers=self.num_islands)
        self.breeder = BreederGene()
        
        
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

        # --- MODIFIED: Use the new config variable ---
        # Run the Breeding Summit at the frequency defined in the config file.
        if self.generation % cfg.breeding_summit_frequency == 0 and self.generation > 0:
            self._run_breeding_summit()
        
        # Run migration every 10 generations (as before)
        if self.generation % cfg.migration_frequency == 0 and self.generation > 0:
            self._migrate()
            
            
    def _run_breeding_summit(self):
        """
        Identifies the champion cell from each island and breeds them to create
        "Perfect Spawn" offspring, which are then injected into other islands.
        """
        logger.info(f"\n{'='*25} 🏆 ARCHIPELAGO BREEDING SUMMIT at Generation {self.generation} 🏆 {'='*25}")
        
        if self.num_islands < 2:
            logger.warning("   Breeding Summit requires at least 2 islands. Skipping.")
            return

        # 1. Identify the champion of each island
        champions = []
        for i, island_gc in enumerate(self.islands):
            # *** MODIFIED: Get champion from the Hall of Fame ***
            champion = island_gc.hall_of_fame.get("champion")
            champions.append(champion)
            
            if champion:
                fitness = island_gc.hall_of_fame.get("fitness", "N/A")
                logger.info(f"   Island {i+1} Champion (from Hall of Fame): Cell {champion.cell_id[:8]} (Fitness: {fitness:.4f}) on {island_gc.device}")
            else:
                logger.warning(f"   Island {i+1} has no champion in its Hall of Fame yet.")
                
                
        # 2. Breed champions in a round-robin fashion
        for i in range(self.num_islands):
            # Parent 1 is the champion of the current island
            parent1 = champions[i]
            # Parent 2 is the champion of the *next* island (circularly)
            parent2 = champions[(i + 1) % self.num_islands]
            # The target island is the one *after* the next one (circularly)
            target_island_index = (i + 2) % self.num_islands
            target_island_gc = self.islands[target_island_index]
            target_device = target_island_gc.device

            if parent1 is None or parent2 is None:
                logger.warning(f"   Skipping breeding for pair ({i+1}, {(i+1)%self.num_islands+1}) due to missing champion.")
                continue

            logger.info(f"\n   Breeding Champions from Island {i+1} and Island {(i + 1) % self.num_islands + 1}...")
            
            # The breeder creates a new gene from the best parts of the parents' genes
            p1_gene = next((g for g in parent1.genes if g.is_active), None)
            p2_gene = next((g for g in parent2.genes if g.is_active), None)

            if p1_gene and p2_gene:
                # Ensure parents are on the same device for breeding analysis if needed
                # For our simple breeder, this isn't strictly necessary, but good practice
                p1_gene_temp = p1_gene.to(target_device)
                p2_gene_temp = p2_gene.to(target_device)
                
                new_gene = self.breeder.recombine(p1_gene_temp, p2_gene_temp)
                
                # Create the "Perfect Spawn" cell on the target device
                perfect_spawn_cell = ProductionBCell([new_gene]).to(target_device)
                
                # 3. Inject the Perfect Spawn into the target island
                # Replace the worst-performing cell on that island
                if target_island_gc.population:
                    worst_cell = min(
                        target_island_gc.population.values(),
                        key=lambda c: c.fitness_history[-1] if c.fitness_history else 0
                    )
                    del target_island_gc.population[worst_cell.cell_id]
                    logger.info(f"   Injecting Perfect Spawn {perfect_spawn_cell.cell_id[:8]} into Island {target_island_index + 1} (on {target_device}).")
                    logger.info(f"   It replaces the worst cell {worst_cell.cell_id[:8]} (Fitness: {worst_cell.fitness_history[-1]:.4f}).")
                
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