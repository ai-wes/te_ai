"""
Run TE-AI training with live 3D visualization, configured by config.py
"""

import sys
import time
import torch
import os
from torch_geometric.data import Data
from scripts.core.instrumented_components import (
    InstrumentedGeneModule, InstrumentedBCell, VisualizableGerminalCenter, VisualizationBridge
)
from scripts.run_optimized_simulation import run_optimized_simulation
# Import visualization bridge from main code
# Import the central configuration
from scripts.config import cfg


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


def run_with_visualization():
    """Run training with live visualization, using parameters from config.py"""
    
    # Load configuration from the central file
    
    # Use config values
    num_generations = cfg.num_generations
    population_size = cfg.initial_population
    num_antigens = cfg.num_environments  # Using num_environments as the source for antigen count
    checkpoint_dir = os.path.join(cfg.save_dir, "viz_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "="*60)
    print("🧬 TE-AI WITH LIVE VISUALIZATION (CONFIG-DRIVEN) 🧬")
    print("="*60)
    print(f"Loaded configuration: Generations={num_generations}, Population={population_size}")
    
    print("\n🎨 Visualization server running on ws://localhost:8765")
    print("🌐 Open neural-clockwork-live.html in your browser")
    print("\n⏳ Waiting 3 seconds for browser connection...")
    time.sleep(3)
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    dataset = create_synthetic_dataset(num_samples=1000, input_dim=cfg.feature_dim)
    
    # Initialize antigens
    antigens = [dataset[torch.randint(0, len(dataset), (1,)).item()] for _ in range(num_antigens)]
    
    # Create initial gene pool with instrumented genes
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

    # Define factories for the instrumented components
    component_factories = {
        'germinal_center': lambda **kwargs: VisualizableGerminalCenter(viz_bridge=viz_bridge, **kwargs),
        'b_cell': lambda genes: InstrumentedBCell(genes, viz_bridge=viz_bridge)
    }

    # Report initial state
    viz_bridge.emit_event('initialization_complete', {
        'population_size': population_size,
        'gene_pool_size': len(initial_genes),
        'num_antigens': len(antigens)
    })
    
    # Run the optimized simulation with our visualization components
    print("\n🚀 Starting evolution with visualization...")
    print("="*60)
    
    best_fitness_overall = run_optimized_simulation(
        num_generations=num_generations,
        population_size=population_size,
        antigens=antigens,
        initial_genes=initial_genes,
        component_factories=component_factories,
        checkpoint_dir=checkpoint_dir,
        # Pass other relevant parameters from the config
        elite_fraction=1.0 - cfg.selection_pressure,
        mutation_rate=cfg.mutation_rate,
    )
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"🏆 Best fitness achieved: {best_fitness_overall:.4f}")
    print("="*60)
    
    # Keep visualization running
    print("\n🎨 Visualization server still running...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == "__main__":
    # No more command-line arguments; just run the function.
    # All configuration is now loaded from config.py
    run_with_visualization()