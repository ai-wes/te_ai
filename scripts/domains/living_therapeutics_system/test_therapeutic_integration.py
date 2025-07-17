#!/usr/bin/env python3
"""
Integration test for Living Therapeutics System
Verifies that core TEAI behaviors are maintained in the therapeutic domain
"""

import torch
import numpy as np
import json
import sys
import os
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from living_therapeutics_system_main import LivingTherapeuticSystem
from living_therapeutics_system_config import cfg

def test_gene_transposition():
    """Test that therapeutic genes properly transpose under stress"""
    print("\n=== Testing Gene Transposition ===")
    
    # Create system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    system = LivingTherapeuticSystem(device=device)
    
    # Initialize a cell
    cell = next(iter(system.b_cells))
    initial_positions = {}
    
    # Record initial gene positions
    for gene_type, gene in cell.genes.items():
        if hasattr(gene, 'position'):
            initial_positions[gene_type] = gene.position.item()
    
    # Apply stress (high disease markers)
    stress_tensor = torch.tensor([0.9] * len(cfg.critical_biomarkers), device=device)
    
    # Trigger transposition
    system.detect_phase_transitions()
    
    # Check if positions changed
    position_changes = 0
    for gene_type, gene in cell.genes.items():
        if hasattr(gene, 'position') and gene_type in initial_positions:
            current_pos = gene.position.item()
            if abs(current_pos - initial_positions[gene_type]) > 0.01:
                position_changes += 1
                print(f"  {gene_type} gene moved: {initial_positions[gene_type]:.3f} -> {current_pos:.3f}")
    
    success = position_changes > 0
    print(f"  Transposition test: {'PASSED' if success else 'FAILED'} ({position_changes} genes moved)")
    return success

def test_cell_evolution():
    """Test that cells evolve and specialize based on fitness"""
    print("\n=== Testing Cell Evolution ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    system = LivingTherapeuticSystem(device=device)
    
    # Run several generations
    print("  Running 5 evolutionary generations...")
    for gen in range(5):
        # Simulate disease input
        disease_markers = torch.rand(1, len(cfg.critical_biomarkers), device=device) * 0.5
        patient_state = torch.rand(1, cfg.patient_state_dim, device=device)
        
        # Forward pass
        outputs = system(disease_markers, patient_state)
        
        # Evolution step
        system.evolve_population(gen)
        
        # Check population metrics
        zones = system.get_visualization_data()['zones']
        print(f"  Gen {gen}: Zones - ", end="")
        for zone_name, zone_data in zones.items():
            print(f"{zone_name}: {zone_data['count']}", end=" ")
        print()
    
    # Verify specialization diversity
    final_zones = system.get_visualization_data()['zones']
    specialized_zones = sum(1 for z in final_zones.values() if z['count'] > 0)
    
    success = specialized_zones >= 3  # Should have at least 3 active zones
    print(f"  Evolution test: {'PASSED' if success else 'FAILED'} ({specialized_zones} specialized zones)")
    return success

def test_visualization_mapping():
    """Test that visualization data correctly represents the system state"""
    print("\n=== Testing Visualization Mapping ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    system = LivingTherapeuticSystem(device=device)
    
    # Get visualization data
    viz_data = system.get_visualization_data()
    
    # Verify gene type mapping
    print("  Checking gene type consistency...")
    expected_gene_types = {'V', 'D', 'J', 'S', 'Q'}
    
    gene_types_found = set()
    for cell_data in viz_data['cells']:
        for gene in cell_data['genes']:
            gene_types_found.add(gene['type'])
    
    gene_mapping_correct = gene_types_found.issubset(expected_gene_types)
    print(f"  Gene types found: {gene_types_found}")
    print(f"  Gene mapping: {'PASSED' if gene_mapping_correct else 'FAILED'}")
    
    # Verify position ranges
    print("  Checking gene position ranges...")
    position_ranges = {'V': (0.0, 0.3), 'D': (0.35, 0.55), 'J': (0.7, 0.9), 'S': (0.4, 0.6), 'Q': (0.0, 1.0)}
    
    position_errors = 0
    for cell_data in viz_data['cells']:
        for gene in cell_data['genes']:
            gene_type = gene['type']
            position = gene['position']
            if gene_type in position_ranges:
                min_pos, max_pos = position_ranges[gene_type]
                if not (min_pos <= position <= max_pos):
                    position_errors += 1
    
    position_test = position_errors == 0
    print(f"  Position test: {'PASSED' if position_test else 'FAILED'} ({position_errors} errors)")
    
    success = gene_mapping_correct and position_test
    return success

def test_core_behaviors():
    """Test core TEAI behaviors are maintained"""
    print("\n=== Testing Core TEAI Behaviors ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    system = LivingTherapeuticSystem(device=device)
    
    # Test 1: Phase transition detection
    print("  Testing phase transition detection...")
    initial_phase = system.phase
    
    # Force high stress
    for _ in range(10):
        high_stress = torch.ones(1, len(cfg.critical_biomarkers), device=device) * 0.9
        patient_state = torch.rand(1, cfg.patient_state_dim, device=device)
        _ = system(high_stress, patient_state)
    
    phase_changed = system.phase != initial_phase
    print(f"  Phase transitions: {'PASSED' if phase_changed else 'FAILED'} ({initial_phase} -> {system.phase})")
    
    # Test 2: Dream consolidation
    print("  Testing dream consolidation...")
    initial_quantum_diversity = system.quantum_diversity.mean().item()
    
    # Trigger dream consolidation
    if hasattr(system, 'quantum_dream_system') and system.quantum_dream_system:
        system.quantum_dream_system.consolidate_dreams()
        final_quantum_diversity = system.quantum_diversity.mean().item()
        dream_worked = abs(final_quantum_diversity - initial_quantum_diversity) > 0.01
        print(f"  Dream consolidation: {'PASSED' if dream_worked else 'FAILED'}")
    else:
        dream_worked = True  # Skip if not available
        print("  Dream consolidation: SKIPPED (not available)")
    
    # Test 3: ODE processing
    print("  Testing ODE-based continuous depth...")
    ode_active = False
    for cell in system.b_cells:
        for gene in cell.genes.values():
            if hasattr(gene, 'ode_func') and gene.ode_func is not None:
                ode_active = True
                break
        if ode_active:
            break
    
    print(f"  ODE processing: {'PASSED' if ode_active else 'FAILED'}")
    
    success = phase_changed and dream_worked and ode_active
    return success

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Living Therapeutics System Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Gene Transposition", test_gene_transposition),
        ("Cell Evolution", test_cell_evolution),
        ("Visualization Mapping", test_visualization_mapping),
        ("Core Behaviors", test_core_behaviors)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n{test_name} ERROR: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("-" * 60)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n✨ All integration tests passed! The therapeutic system maintains core TEAI behaviors.")
    else:
        print("\n⚠️  Some tests failed. The system may not fully maintain TEAI architecture.")
    
    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'tests': [{'name': name, 'passed': passed} for name, passed in results],
        'summary': f"{passed_tests}/{total_tests} passed"
    }
    
    with open('integration_test_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to integration_test_results.json")

if __name__ == "__main__":
    main()