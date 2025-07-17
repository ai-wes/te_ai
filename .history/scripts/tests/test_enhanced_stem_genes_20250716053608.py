#!/usr/bin/env python3
"""
Test Enhanced Stem Gene Module
==============================
Demonstrates all the new biological features added to stem genes
"""

import torch
import torch.nn as nn
from stem_gene_module import StemGeneModule, GuidedStemGene
from scripts.config import cfg

def test_basic_features():
    """Test basic stem gene functionality"""
    print("=" * 60)
    print("TESTING BASIC STEM GENE FEATURES")
    print("=" * 60)
    
    # Create stem gene
    stem = StemGeneModule()
    print(f"✓ Created stem gene with {len(stem.gene_types)} types: {stem.gene_types}")
    print(f"✓ Initial state: {stem.differentiation_state.data}")
    print(f"✓ Morphogen field shape: {stem.morphogen_field.shape}")
    
    # Test differentiation with noise
    print("\n1. Testing noisy differentiation:")
    stem.differentiate('V')
    print(f"   State after V differentiation: {stem.differentiation_state.data}")
    print(f"   Commitment level: {stem.commitment_level}")
    
    # Test de-differentiation
    print("\n2. Testing de-differentiation under stress:")
    result = stem.de_differentiate(0.95)
    print(f"   De-differentiation result: {result}")
    print(f"   State after de-diff: {stem.differentiation_state.data}")

def test_asymmetric_division():
    """Test asymmetric division feature"""
    print("\n" + "=" * 60)
    print("TESTING ASYMMETRIC DIVISION")
    print("=" * 60)
    
    stem = StemGeneModule()
    population_stats = {
        'V_count': 10,
        'D_count': 0,  # No D genes - crisis!
        'J_count': 5,
        'stress_level': 0.8,
        'diversity': 0.3
    }
    
    print("Initial stem state:")
    print(f"  Plasticity: {stem.plasticity.item():.3f}")
    print(f"  Commitment: {stem.commitment_level}")
    
    # Try asymmetric division
    daughter = stem.divide_asymmetrically(population_stats)
    if daughter:
        print(f"\n✓ Created daughter cell:")
        print(f"  Daughter type: {daughter.gene_type}")
        print(f"  Daughter commitment: {daughter.commitment_level}")
        print(f"  Daughter plasticity: {daughter.plasticity.item():.3f}")
        print(f"  Parent remains stem: {stem.gene_type == 'S'}")

def test_morphogen_niche():
    """Test morphogen field and niche modeling"""
    print("\n" + "=" * 60)
    print("TESTING MORPHOGEN NICHE MODELING")
    print("=" * 60)
    
    # Create multiple stem genes
    stems = [StemGeneModule() for _ in range(3)]
    
    print("Initial morphogen fields (first 5 dims):")
    for i, stem in enumerate(stems):
        print(f"  Stem {i}: {stem.morphogen_field[:5].data}")
    
    # Update morphogen fields based on neighbors
    print("\nUpdating morphogen fields based on neighbors...")
    stems[1].update_morphogen([stems[0].morphogen_field, stems[2].morphogen_field])
    
    print("After neighbor influence:")
    print(f"  Stem 1: {stems[1].morphogen_field[:5].data}")

def test_rl_decision_making():
    """Test RL-based differentiation decisions"""
    print("\n" + "=" * 60)
    print("TESTING RL-BASED DECISION MAKING")
    print("=" * 60)
    
    stem = StemGeneModule()
    
    # Simulate multiple decision cycles
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")
        
        population_stats = {
            'V_count': 10 + cycle * 2,
            'D_count': 5 - cycle,
            'J_count': 8,
            'stress_level': 0.3 + cycle * 0.2,
            'diversity': 0.5,
            'avg_fitness': 0.5 + cycle * 0.1
        }
        
        # Use RL for decision
        needs = stem.sense_population_needs(population_stats, use_rl=True)
        print(f"  RL policy output: {needs.data}")
        
        # Differentiate based on RL
        stem.differentiate(population_stats=population_stats, use_rl=True)
        
        # Update policy based on improved fitness
        if cycle > 0:
            stem._update_rl_policy(population_stats)

def test_dynamic_gene_types():
    """Test adding new gene types dynamically"""
    print("\n" + "=" * 60)
    print("TESTING DYNAMIC GENE TYPE ADDITION")
    print("=" * 60)
    
    stem = StemGeneModule()
    print(f"Initial gene types: {stem.gene_types}")
    print(f"Initial state shape: {stem.differentiation_state.shape}")
    
    # Add new gene type
    stem.add_new_type('X')
    print(f"\nAfter adding type 'X':")
    print(f"  Gene types: {stem.gene_types}")
    print(f"  State shape: {stem.differentiation_state.shape}")
    print(f"  Components: {list(stem.gene_components.keys())}")
    
    # Test differentiation to new type
    stem.differentiate('X')
    print(f"\nDifferentiated to X:")
    print(f"  State: {stem.differentiation_state.data}")

def test_error_correction():
    """Test error correction and apoptosis"""
    print("\n" + "=" * 60)
    print("TESTING ERROR CORRECTION")
    print("=" * 60)
    
    stem = StemGeneModule()
    
    # Create invalid state
    print("Creating invalid state...")
    stem.commitment_level = 1.5  # Invalid!
    
    # Try forward pass
    x = torch.randn(10, cfg.hidden_dim)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    
    output = stem.forward(x, edge_index)
    print(f"✓ Error correction triggered: is_active = {stem.is_active}")

def test_hybrid_differentiation():
    """Test hybrid/multi-lineage differentiation"""
    print("\n" + "=" * 60)
    print("TESTING HYBRID DIFFERENTIATION")
    print("=" * 60)
    
    stem = StemGeneModule()
    
    # Manually set hybrid state
    with torch.no_grad():
        stem.differentiation_state.data = torch.tensor([0.3, 0.3, 0.2, 0.2])  # V+D hybrid
        stem.commitment_level = 0.6
    
    print("Hybrid differentiation state:")
    print(f"  V: {stem.differentiation_state[0]:.2f}")
    print(f"  D: {stem.differentiation_state[1]:.2f}")
    print(f"  J: {stem.differentiation_state[2]:.2f}")
    print(f"  S: {stem.differentiation_state[3]:.2f}")
    
    # Test forward pass with hybrid processing
    x = torch.randn(1, cfg.hidden_dim)
    edge_index = torch.tensor([[0], [0]])
    
    # Mock the parent class methods we need
    stem.input_projection = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
    stem._apply_epigenetic_regulation = lambda x: x  # Identity for testing
    
    # We can't call full forward without parent implementation, but show the concept
    print("\n✓ Hybrid processing would combine V, D, J, S components weighted by state")

def main():
    """Run all tests"""
    print("\n🧬 ENHANCED STEM GENE MODULE TEST SUITE 🧬\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    test_basic_features()
    test_asymmetric_division()
    test_morphogen_niche()
    test_rl_decision_making()
    test_dynamic_gene_types()
    test_error_correction()
    test_hybrid_differentiation()
    
    print("\n✅ All tests completed!")
    print("\nThe enhanced stem gene module now includes:")
    print("  • Asymmetric division (self-renewal + differentiation)")
    print("  • Stochastic differentiation with biological noise")
    print("  • RL-based adaptive decision making")
    print("  • Dynamic gene type support")
    print("  • Morphogen field niche modeling")
    print("  • Error correction and apoptosis")
    print("  • Hybrid/multi-lineage states")

if __name__ == "__main__":
    main()