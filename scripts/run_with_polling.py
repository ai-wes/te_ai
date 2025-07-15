"""
Run TE-AI training with polling-based visualization
"""

import sys
import time
import torch
import os
from torch_geometric.data import Data
from polling_bridge import InstrumentedGeneModule, InstrumentedBCell, PollingGerminalCenter, polling_bridge
from fast_optimized_te_ai import run_optimized_simulation
from config import CFG


def create_synthetic_dataset(num_samples=1000, input_dim=128, num_classes=10):
    """Create a simple synthetic dataset for testing"""
    dataset = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(num_samples):
        x = torch.randn(20, input_dim, device=device)
        edge_index = []
        for j in range(20):
            num_neighbors = torch.randint(2, 5, (1,)).item()
            neighbors = torch.randperm(20)[:num_neighbors]
            for n in neighbors:
                if n != j:
                    edge_index.append([j, n.item()])
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        y = torch.randint(0, num_classes, (1,), device=device)
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    
    return dataset


def run_with_polling():
    """Run training with polling-based visualization"""
    
    # Create polling mode flag to prevent architecture modifier from overwriting
    with open('.polling_mode', 'w') as f:
        f.write('1')
    
    # Load configuration from the central file
    cfg = CFG()
    
    # Use config values
    num_generations = cfg.num_generations
    population_size = cfg.initial_population
    num_antigens = cfg.num_environments
    checkpoint_dir = os.path.join(cfg.save_dir, "polling_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "="*60)
    print("🧬 TE-AI WITH POLLING VISUALIZATION 🧬")
    print("="*60)
    print(f"Loaded configuration: Generations={num_generations}, Population={population_size}")
    
    print("\n📊 Polling state will be written to te_ai_state.json")
    print("🌐 Open neural-clockwork-live_1.html in your browser")
    print("\n⏳ Starting in 3 seconds...")
    time.sleep(3)
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    dataset = create_synthetic_dataset(num_samples=1000, input_dim=cfg.feature_dim)
    
    # Initialize antigens
    antigens = [dataset[torch.randint(0, len(dataset), (1,)).item()] for _ in range(num_antigens)]
    
    # Create initial gene pool with instrumented genes from polling_bridge
    print("\n🧬 Initializing gene pool...")
    initial_genes = []
    gene_types = ['V', 'D', 'J', 'Q']
    for i in range(30):
        gene_type = gene_types[i % len(gene_types)]
        gene = InstrumentedGeneModule(
            gene_type=gene_type,
            variant_id=i
        )
        initial_genes.append(gene)

    # Define factories for the polling components
    component_factories = {
        'germinal_center': lambda **kwargs: PollingGerminalCenter(),
        'b_cell': lambda genes: InstrumentedBCell(genes)
    }

    # Report initial state
    polling_bridge.update_metrics(
        generation=0,
        population=population_size,
        fitness=0.0,
        diversity=0.0,
        stress=0.0,
        phase='initializing'
    )
    
    polling_bridge.add_event('initialization_complete', {
        'population_size': population_size,
        'gene_pool_size': len(initial_genes),
        'num_antigens': len(antigens)
    })
    
    # Run the optimized simulation with polling components
    print("\n🚀 Starting evolution with polling...")
    print("="*60)
    
    best_fitness_overall = run_optimized_simulation(
        num_generations=num_generations,
        population_size=population_size,
        antigens=antigens,
        initial_genes=initial_genes,
        component_factories=component_factories,
        checkpoint_dir=checkpoint_dir,
        elite_fraction=1.0 - cfg.selection_pressure,
        mutation_rate=cfg.mutation_rate,
    )
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"🏆 Best fitness achieved: {best_fitness_overall:.4f}")
    print("="*60)
    
    # Keep polling active
    print("\n📊 Polling state still being written to te_ai_state.json...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        # Clean up polling mode flag
        if os.path.exists('.polling_mode'):
            os.remove('.polling_mode')


if __name__ == "__main__":
    run_with_polling()