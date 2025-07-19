# scripts/core/advanced/run_synthesis_experiment.py
import torch
import random
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.deepchem_to_teai import DeepChemToTEAI
from scripts.core.production_germinal_center import ProductionGerminalCenter
import networkx as nx
from scripts.core.causal_tapestry import CausalTapestry
from scripts.core.production_b_cell import ProductionBCell # Needed to reconstruct cells
from scripts.core.advanced.archipelago_germinal_center import ArchipelagoGerminalCenter
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

def find_champion_from_tapestry(graph: nx.DiGraph, island_name: str) -> str:
    """
    Queries a Causal Tapestry graph to find the ID of the cell that was
    crowned champion of a specific island with the highest fitness.
    """
    champion_candidates = []
    for node, data in graph.nodes(data=True):
        # Find all nodes that are cells and belong to the target island
        if data.get('type') == 'cell' and data.get('island') == island_name:
            # Check if this cell was ever crowned a champion. We can do this by
            # seeing if it has an incoming edge from a "NEW_CHAMPION" event.
            for predecessor in graph.predecessors(node):
                if graph.nodes[predecessor].get('event_type') == 'NEW_CHAMPION':
                    champion_candidates.append((node, data.get('fitness', -1.0)))
                    break # Move to the next cell node

    if not champion_candidates:
        logger.warning(f"No champions found for island '{island_name}' in the tapestry.")
        return None

    # Find the champion with the highest fitness score among all candidates
    best_champion_id, best_fitness = max(champion_candidates, key=lambda item: item[1])
    logger.info(f"Found champion for '{island_name}': Cell {best_champion_id[:8]} with fitness {best_fitness:.4f}")
    
    return best_champion_id

def reconstruct_cell_from_tapestry(graph: nx.DiGraph, cell_id: str, device) -> ProductionBCell:
    """
    Reconstructs a ProductionBCell object from the genetic information
    stored in the Causal Tapestry.
    """
    from scripts.core.ode import ContinuousDepthGeneModule # Import locally
    
    cell_data = graph.nodes[cell_id]
    gene_ids = cell_data.get('genes', [])
    
    reconstructed_genes = []
    for gene_id in gene_ids:
        gene_data = graph.nodes[gene_id]
        # NOTE: This is a simplified reconstruction. A real implementation would
        # need to store and reload the actual model weights (state_dict) as node attributes.
        # For now, we re-create the gene with its basic type and variant ID.
        gene = ContinuousDepthGeneModule(
            gene_type=gene_data.get('gene_type'),
            variant_id=gene_data.get('variant_id')
        )
        reconstructed_genes.append(gene)
        
    # The ProductionBCell constructor is now device-aware
    reconstructed_cell = ProductionBCell(reconstructed_genes, device=device)
    # We can assign the original ID for tracking
    reconstructed_cell.cell_id = cell_id 
    
    return reconstructed_cell

def run_super_champion_synthesis():
    """
    The "Grand Synthesis" pipeline. Loads multiple experimental histories,
    extracts the best specialists, and seeds a new "Super Champion Island"
    to evolve an integrated solution.
    """
    logger.info(f"\n{'='*60}")
    logger.info("🧬 STARTING GRAND SYNTHESIS EXPERIMENT 🧬")
    logger.info(f"{'='*60}\n")

    # --- Stage 1: Load and Merge Tapestries ---
    logger.info("--- Stage 1: Loading and Merging Evolutionary Histories ---")
    GRAND_TAPESTRY = CausalTapestry()
    
    # NOTE: For this to work, you must have previously run experiments and saved these files.
    # We will use placeholder names.
    tapestry_files = [
        "tapestry_efficacy_run.graphml",
        "tapestry_safety_run.graphml",
        "tapestry_adme_run.graphml"
    ]
    
    for f in tapestry_files:
        try:
            GRAND_TAPESTRY.merge_tapestry(f)
        except FileNotFoundError:
            logger.error(f"Could not find tapestry file: {f}. Skipping.")
            # In a real run, you might want to exit here if files are missing.
    
    logger.info(f"Grand Tapestry now contains {GRAND_TAPESTRY.graph.number_of_nodes()} nodes and {GRAND_TAPESTRY.graph.number_of_edges()} edges.")

    # --- Stage 2: Identify the All-Star Champions ---
    logger.info("\n--- Stage 2: Identifying All-Star Champions from History ---")
    
    # These names MUST match the island names used in the original experiments
    efficacy_champion_id = find_champion_from_tapestry(GRAND_TAPESTRY.graph, "SCIENTIST (Accuracy-focused)")
    safety_champion_id = find_champion_from_tapestry(GRAND_TAPESTRY.graph, "TOXICOLOGIST (Safety-focused)")
    adme_champion_id = find_champion_from_tapestry(GRAND_TAPESTRY.graph, "PHARMACIST (ADME-focused)")
    
    all_star_ids = [id for id in [efficacy_champion_id, safety_champion_id, adme_champion_id] if id is not None]
    
    if not all_star_ids:
        logger.critical("Could not find any champions in the loaded tapestries. Aborting synthesis.")
        return

    # --- Stage 3: Create and Seed the Super Champion Island ---
    logger.info("\n--- Stage 3: Seeding the Super Champion Island ---")
    
    # Define the device for our new island
    super_island_device = torch.device("cuda:0")
    
    # Reconstruct the champion cells from the graph data
    super_champion_population = {
        cell_id: reconstruct_cell_from_tapestry(GRAND_TAPESTRY.graph, cell_id, device=super_island_device)
        for cell_id in all_star_ids
    }
    
    # Create a new, single Germinal Center for our super island
    super_island = ProductionGerminalCenter(
        initial_population_size=len(super_champion_population),
        device=super_island_device
    )
    
    # Replace its random initial population with our hand-picked champions
    super_island.population = super_champion_population
    logger.info(f"Super Champion Island seeded with {len(super_island.population)} elite individuals.")

    # --- Stage 4: Evolve the Super Champions ---
    logger.info("\n--- Stage 4: Running Final Refinement Evolution ---")
    
    # Here you would define a new, final fitness function that combines all objectives
    # For now, we'll just reuse the 'balanced' one as a placeholder.
    from scripts.core.advanced.fitness_functions import calculate_balanced_fitness
    final_fitness_function = calculate_balanced_fitness
    
    # Load the training data
    converter = DeepChemToTEAI()
    train_antigens, _, _ = converter.convert_molnet_dataset(cfg.dataset_name)
    
    # Run for a few generations to allow the elite genes to combine and refine
    num_refinement_generations = 10
    for gen in range(num_refinement_generations):
        antigen_batch_domain_objects = random.sample(train_antigens, k=min(32, len(train_antigens)))
        graph_batch = [target.to_graph() for target in antigen_batch_domain_objects]
        antigen_batch_device = [g.to(super_island_device) for g in graph_batch]
        
        super_island.evolve_generation(antigen_batch_device, final_fitness_function)

    # --- Stage 5: Final Result ---
    final_champion = super_island.hall_of_fame.get("champion")
    final_fitness = super_island.hall_of_fame.get("fitness")
    
    print("\n" + "="*60)
    print("🏆 GRAND SYNTHESIS COMPLETE 🏆")
    print("="*60)
    if final_champion:
        print(f"The ultimate 'Super Spawn' is Cell {final_champion.cell_id[:8]} with a final fitness of {final_fitness:.4f}")
    else:
        print("The super champion island did not produce a final champion.")

if __name__ == "__main__":
    run_super_champion_synthesis()