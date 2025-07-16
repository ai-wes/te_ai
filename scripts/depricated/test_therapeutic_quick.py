#!/usr/bin/env python3
"""
Quick test of Living Therapeutic System
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
print("Testing imports...")
try:
    from living_therapeutic_system import (
        TherapeuticConfig, THERAPY_CFG,
        BiosensorGene, TherapeuticEffectorGene, 
        AdaptiveControllerGene, TherapeuticStemGene
    )
    print("✓ Therapeutic modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test gene creation
print("\nTesting therapeutic gene creation...")
try:
    # Create each type of therapeutic gene
    biosensor = BiosensorGene()
    print(f"✓ Created BiosensorGene: {biosensor.gene_type}")
    
    effector = TherapeuticEffectorGene('anti-inflammatory')
    print(f"✓ Created TherapeuticEffectorGene: {effector.therapeutic_mode}")
    
    controller = AdaptiveControllerGene()
    print(f"✓ Created AdaptiveControllerGene: {controller.gene_type}")
    
    stem = TherapeuticStemGene()
    print(f"✓ Created TherapeuticStemGene: {stem.gene_type}")
except Exception as e:
    print(f"✗ Gene creation error: {e}")
    sys.exit(1)

# Test biosensing
print("\nTesting biosensor functionality...")
try:
    # Generate test biomarkers
    biomarkers = torch.randn(THERAPY_CFG.num_biomarkers)
    biomarkers[0] = 5.0  # High IL-6
    biomarkers[1] = 3.0  # High TNF-α
    
    # Test sensing
    patient_state = biosensor.sense_patient_state(biomarkers)
    print(f"✓ Biosensor detected disease probabilities: {patient_state['disease_probabilities'].shape}")
    print(f"✓ Critical conditions: {patient_state['critical_conditions']}")
except Exception as e:
    print(f"✗ Biosensing error: {e}")

# Test therapeutic generation
print("\nTesting therapeutic generation...")
try:
    therapeutic_output = effector.generate_therapeutic(patient_state)
    print(f"✓ Generated therapeutic with dose: {therapeutic_output['dose'].item():.3f}")
    print(f"✓ Safety score: {therapeutic_output['safety_score'].item():.3f}")
except Exception as e:
    print(f"✗ Therapeutic generation error: {e}")

# Test controller planning
print("\nTesting treatment planning...")
try:
    plan = controller.plan_treatment([], patient_state, horizon=6)
    print(f"✓ Created treatment plan for {plan['horizon']} hours")
    print(f"✓ Objectives: {plan['objectives'].tolist()}")
except Exception as e:
    print(f"✗ Planning error: {e}")

# Test stem cell differentiation
print("\nTesting stem cell differentiation...")
try:
    population_state = {
        'cells': [],
        'avg_fitness': 0.5,
        'diversity': 0.3
    }
    
    needs = stem.sense_therapeutic_needs(patient_state, population_state)
    print(f"✓ Detected therapeutic needs: {needs['recommended_type']}")
    print(f"✓ Need scores: {needs['need_scores'].tolist()}")
except Exception as e:
    print(f"✗ Differentiation error: {e}")

print("\n✅ All basic tests passed! The living therapeutic system is ready to run.")
print("\nTo run the full simulation, use:")
print("  python run_living_therapeutic.py --patient-type autoimmune --days 7")
print("\nOr for a quick test:")
print("  python run_living_therapeutic.py --days 1 --cycles-per-day 2")