# main_drug_discovery.py

import os
import json
import datetime
from scripts.domains.drug_discovery.tcga_adapter import TCGADataConverter
from scripts.domains.drug_discovery.drug_discovery_germinal_center import DrugDiscoveryGerminalCenter
from scripts.domains.drug_discovery.drug_target_evaluator import DrugTargetEvaluator
import random
from scripts.config import cfg

def main():
    print("Initializing TCGA Data Converter...")
    # 1. Initialize the Domain Adapter
    # Convert Windows path to WSL path
    tcga_data_dir = "C:\\Users\\wes\\Desktop\\te_ai\\maia_tcga_pancan"
    # Check if directory exists
    if not os.path.exists(tcga_data_dir):
        print(f"ERROR: TCGA data directory not found at {tcga_data_dir}")
        print("Please ensure the TCGA data is available at the specified path.")
        return
    tcga_converter = TCGADataConverter(tcga_data_dir=tcga_data_dir)

    print("Loading TCGA samples (max 500)...")
    # 2. Load and Process Domain-Specific Data
    samples = tcga_converter.load_tcga_samples(max_samples=500)
    n_samples = len(samples['sample_ids']) if isinstance(samples, dict) and 'sample_ids' in samples else len(samples)
    print(f"Loaded {n_samples} samples.")

    print("Identifying top 20 drug targets from TCGA samples...")
    target_ids = tcga_converter.identify_drug_targets_from_tcga(samples, top_k=20)
    print(f"Identified {len(target_ids)} target IDs.")

    print("Converting TCGA targets to antigen objects...")
    drug_targets = tcga_converter.convert_tcga_to_antigens(target_ids, samples)
    print(f"Converted {len(drug_targets)} drug targets to antigens.")

    # 3. Initialize the Specialized Evolutionary Framework
    print("Initializing DrugDiscoveryGerminalCenter...")
    germinal_center = DrugDiscoveryGerminalCenter()

    # For tracking evolution history
    evolution_history = []

    # 4. Run the Evolution Loop
    print("Starting evolution loop for 200 generations...")
    for generation in range(200):
        print(f"Generation {generation+1}/200")
        # Sample a batch of drug targets to challenge the population
        target_batch_domain_objects = random.sample(drug_targets, k=min(cfg.batch_size, len(drug_targets)))
        print(f"  Sampled {len(target_batch_domain_objects)} targets for this generation.")

        # Use the adapter's .to_graph() method to convert them into the universal format
        graph_batch = [target.to_graph() for target in target_batch_domain_objects]
        print(f"  Converted batch to {len(graph_batch)} graph objects.")

        # The germinal center evolves the population using the universal graph format
        # Its overridden _evaluate_population_parallel method will use the correct fitness logic.
        stats = germinal_center.evolve_generation(graph_batch)
        print(f"  Evolution step complete for generation {generation+1}.")

        # Optionally, collect stats for report
        if stats is not None:
            stats['generation'] = generation + 1
            evolution_history.append(stats)

    print("Evolution complete.")

    # 5. Final Analysis
    print("Evaluating final population and generating report...")
    evaluator = DrugTargetEvaluator()
    final_population = getattr(germinal_center, "population", None)
    evaluation_results = None
    if final_population is not None:
        evaluation_results = evaluator.evaluate_population(final_population, drug_targets)
    else:
        print("Warning: No final population found for evaluation.")

    # Prepare results for saving
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "drug_discovery_results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"drug_discovery_run_{now}.json")

    results = {
        "run_timestamp": now,
        "n_samples": n_samples,
        "n_targets": len(target_ids),
        "generations": 200,
        "batch_size": cfg.batch_size,
        "evolution_history": evolution_history,
        "final_evaluation": evaluation_results,
        "target_ids": target_ids,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print comprehensive report
    print("\n" + "="*80)
    print("DRUG DISCOVERY RUN REPORT")
    print("="*80)
    print(f"Run timestamp: {now}")
    print(f"Samples loaded: {n_samples}")
    print(f"Drug targets identified: {len(target_ids)}")
    print(f"Generations: 200")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Results file: {results_path}")

    if evolution_history:
        print("\nEvolution Progress (last 5 generations):")
        for stat in evolution_history[-5:]:
            print(f"  Generation {stat.get('generation', '?')}: {stat}")

    if evaluation_results:
        print("\nFinal Population Evaluation (top 5):")
        if isinstance(evaluation_results, list):
            for i, res in enumerate(evaluation_results[:5]):
                print(f"  {i+1}. {res}")
        elif isinstance(evaluation_results, dict):
            for k, v in list(evaluation_results.items())[:5]:
                print(f"  {k}: {v}")
        else:
            print(f"  {evaluation_results}")
    else:
        print("\nNo evaluation results available.")

    print("="*80)
    print("End of report.")

if __name__ == "__main__":
    main()