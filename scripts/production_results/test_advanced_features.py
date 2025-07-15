"""
Test Advanced Features
=====================
Demonstration of all enhanced TE-AI capabilities including:
- GPU acceleration
- Continuous depth
- Epigenetic memory
- Dream consolidation
- Causal reasoning
- Phase transitions
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transposable_immune_ai_v2 import (
    ConfigV2, GPUAcceleratedGerminalCenter, 
    generate_antigen_graph, visualize_continuous_depth_evolution,
    plot_enhanced_metrics
)
from advanced_transposon_modules import (
    integrate_advanced_modules, DreamConsolidator,
    CausalTranspositionEngine, PhaseTransitionDetector
)
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def demonstrate_advanced_features():
    """Run comprehensive test of all advanced features"""
    
    print("\n" + "="*70)
    print("🧬 TRANSPOSABLE ELEMENT AI - ADVANCED FEATURES DEMONSTRATION")
    print("="*70 + "\n")
    
    # Enhanced configuration
    config = ConfigV2()
    config.epochs = 50  # Shorter for demo
    config.save_dir = "advanced_demo_results"
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Initialize population with advanced modules
    print("📦 Initializing enhanced population...")
    center = GPUAcceleratedGerminalCenter()
    center = integrate_advanced_modules(center, config)
    
    # Test 1: Dream Consolidation
    print("\n" + "-"*50)
    print("TEST 1: Dream-Based Consolidation")
    print("-"*50)
    
    # Record some experiences first
    for _ in range(20):
        antigen = generate_antigen_graph()
        center.evolve([antigen])
        
        # Record experiences for dreaming
        for cell in list(center.population.values())[:10]:
            if hasattr(cell, 'genes') and cell.genes:
                gene_state = cell.genes[0].output_proj.weight.data.mean(dim=0)
                fitness = cell.fitness_history[-1] if cell.fitness_history else 0.5
                center.dream_consolidator.record_experience(
                    gene_state, fitness, center.current_stress
                )
    
    # Run dream phase
    center.dream_consolidator.dream_phase(center.population, num_cycles=3)
    
    # Test 2: Causal Reasoning
    print("\n" + "-"*50)
    print("TEST 2: Causal Transposition Reasoning")
    print("-"*50)
    
    # Simulate some transpositions and record causal effects
    test_cell = list(center.population.values())[0]
    test_gene = test_cell.genes[0]
    
    # Record causal relationships
    for i in range(5):
        fitness_before = np.random.random()
        action = np.random.choice(['jump', 'duplicate', 'invert'])
        fitness_after = fitness_before + np.random.uniform(-0.2, 0.3)
        
        center.causal_engine.observe_transposition(
            test_gene.gene_id, action, fitness_before, fitness_after,
            {'stress': center.current_stress, 'generation': center.generation}
        )
    
    # Generate counterfactuals
    counterfactuals = center.causal_engine.generate_counterfactual(
        test_gene.gene_id,
        {'stress': 0.7, 'generation': 10}
    )
    
    print("🔮 Counterfactual predictions:")
    for cf in counterfactuals:
        print(f"  - {cf['action']}: effect={cf['predicted_effect']:.3f}, "
              f"confidence={cf['confidence']:.2f}")
    
    # Test 3: Phase Transition Detection
    print("\n" + "-"*50)
    print("TEST 3: Phase Transition Detection")
    print("-"*50)
    
    # Simulate approaching phase transition
    for i in range(30):
        # Artificially create critical slowing
        fitness = 0.5 + 0.01 * i + np.random.normal(0, 0.01 * (1 + i/30))
        stress = 0.1 + 0.02 * i
        
        center.phase_detector.update_metrics({
            'mean_fitness': fitness,
            'stress_level': stress,
            'population_size': 100 + np.random.randint(-5, 5)
        })
        
        # Check for warnings
        alerts = center.phase_detector.alert_if_critical({
            'mean_fitness': fitness,
            'stress_level': stress
        })
        
        if alerts:
            for alert in alerts:
                print(alert)
    
    # Test 4: Self-Modifying Architecture
    print("\n" + "-"*50)
    print("TEST 4: Self-Modifying Neural Architecture")
    print("-"*50)
    
    test_cell = list(center.population.values())[0]
    if hasattr(test_cell, 'self_modifier'):
        # Simulate performance feedback
        perf_gradient = torch.randn(config.hidden_dim)
        current_state = torch.randn(config.hidden_dim)
        
        # Decide and apply modifications
        for i in range(5):
            mod_type = test_cell.self_modifier.decide_modification(
                perf_gradient, current_state
            )
            print(f"  Modification {i+1}: {mod_type}")
            
            if mod_type != 'none':
                test_cell.self_modifier.apply_modification(
                    mod_type, test_cell.genes[0]
                )
    
    # Test 5: Advanced Gene Regulation
    print("\n" + "-"*50)
    print("TEST 5: Gene Regulatory Networks")
    print("-"*50)
    
    if hasattr(test_cell, 'gene_regulatory_network'):
        # Get gene embeddings
        gene_embeddings = []
        for gene in test_cell.genes[:5]:
            if gene.is_active:
                embedding = torch.randn(config.hidden_dim)
                gene_embeddings.append(embedding)
        
        if gene_embeddings:
            # Compute regulatory matrix
            reg_matrix = test_cell.gene_regulatory_network.compute_regulatory_matrix(
                gene_embeddings
            )
            
            print(f"📊 Regulatory matrix shape: {reg_matrix.shape}")
            print(f"   Promoter relationships: {(reg_matrix > 0).sum().item()}")
            print(f"   Repressor relationships: {(reg_matrix < 0).sum().item()}")
            
            # Test oscillation detection
            history = [torch.randn(5, config.hidden_dim) for _ in range(15)]
            has_oscillation = test_cell.gene_regulatory_network.detect_oscillations(
                history
            )
            print(f"   Oscillatory behavior detected: {has_oscillation}")
    
    # Test 6: Full Evolution with All Features
    print("\n" + "-"*50)
    print("TEST 6: Full Evolution with All Advanced Features")
    print("-"*50)
    
    # Run complete evolution
    for epoch in range(10):
        # Generate antigens
        antigens = [generate_antigen_graph() for _ in range(config.batch_size)]
        
        # Evolve with all features active
        center.evolve(antigens)
        
        # Periodic dream consolidation
        if epoch % 5 == 0 and epoch > 0:
            center.dream_consolidator.dream_phase(center.population, num_cycles=2)
        
        # Check phase transitions
        if center.fitness_landscape:
            latest = center.fitness_landscape[-1]
            alerts = center.phase_detector.alert_if_critical({
                'mean_fitness': latest['mean_fitness'],
                'stress_level': latest['stress_level'],
                'unique_genes': latest['unique_genes']
            })
    
    # Final visualization
    print("\n📊 Generating final visualizations...")
    
    # Gene topology
    visualize_continuous_depth_evolution(
        center, f"{config.save_dir}/final_gene_topology.png"
    )
    
    # Metrics
    plot_enhanced_metrics(center, f"{config.save_dir}/evolution_metrics.png")
    
    # Phase transition analysis
    if center.phase_detector.metric_history:
        plt.figure(figsize=(12, 8))
        
        # Plot autocorrelation trends
        for metric in ['mean_fitness', 'stress_level']:
            autocorr_key = f"{metric}_autocorr"
            if autocorr_key in center.phase_detector.metric_history:
                autocorr = center.phase_detector.metric_history[autocorr_key]
                plt.plot(autocorr, label=f'{metric} autocorrelation')
        
        plt.xlabel('Time')
        plt.ylabel('Autocorrelation')
        plt.title('Critical Slowing Down Indicators')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{config.save_dir}/phase_transitions.png", dpi=150)
        plt.close()
    
    print(f"\n✅ All tests completed! Results saved to {config.save_dir}/")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    if center.fitness_landscape:
        final_stats = center.fitness_landscape[-1]
        print(f"Final mean fitness: {final_stats['mean_fitness']:.4f}")
        print(f"Final max fitness: {final_stats['max_fitness']:.4f}")
        print(f"Unique genes: {final_stats['unique_genes']}")
        print(f"Mean gene depth: {final_stats['mean_depth']:.3f}")
        print(f"Simpson diversity: {final_stats['simpson_index']:.3f}")
    
    print(f"\nTotal transposition events: {len(center.transposition_events)}")
    print(f"Dream consolidation cycles: {len(center.dream_consolidator.memory_buffer)}")
    print(f"Causal interventions recorded: {len(center.causal_engine.intervention_history)}")
    print(f"Phase transition alerts: {len(center.phase_detector.transition_alerts)}")
    
    return center

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Run demonstration
    center = demonstrate_advanced_features()
    
    print("\n🎉 Advanced features demonstration complete!")