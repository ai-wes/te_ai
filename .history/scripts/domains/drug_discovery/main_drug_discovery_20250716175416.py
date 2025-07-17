# main_drug_discovery.py

from scripts.domains.drug_discovery.tcga_adapter import TCGADataConverter
from scripts.domains.drug_discovery.drug_discovery_germinal_center import DrugDiscoveryGerminalCenter
import random
from scripts.config import cfg

def main():
    # 1. Initialize the Domain Adapter
    tcga_converter = TCGADataConverter(tcga_data_dir=r"D:\masked-moics-maia\data\maia_tcga_pancan")

    # 2. Load and Process Domain-Specific Data
    # This is a one-time setup cost
    samples = tcga_converter.load_tcga_samples(max_samples=500)
    target_ids = tcga_converter.identify_drug_targets_from_tcga(samples, top_k=20)
    drug_targets = tcga_converter.convert_tcga_to_antigens(target_ids, samples)

    # 3. Initialize the Specialized Evolutionary Framework
    germinal_center = DrugDiscoveryGerminalCenter()

    # 4. Run the Evolution Loop
    for generation in range(200):
        # Sample a batch of drug targets to challenge the population
        target_batch_domain_objects = random.sample(drug_targets, k=min(cfg.batch_size, len(drug_targets)))

        # Use the adapter's .to_graph() method to convert them into the universal format
        graph_batch = [target.to_graph() for target in target_batch_domain_objects]

        # The germinal center evolves the population using the universal graph format
        # Its overridden _evaluate_population_parallel method will use the correct fitness logic.
        germinal_center.evolve_generation(graph_batch)

    # 5. Final Analysis
    # Use your DrugTargetEvaluator or other analysis scripts on the final population.
    print("Evolution complete.")

if __name__ == "__main__":
    main()