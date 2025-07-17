#!/usr/bin/env python3
"""
Debug script for Living Therapeutic System
"""

import torch
import sys
import traceback

# Test imports
print("Testing therapeutic system components...")

try:
    from living_therapeutic_system import (
        TherapeuticConfig, THERAPY_CFG,
        BiosensorGene, TherapeuticEffectorGene,
        TherapeuticStemGene
    )
    from config import cfg
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test each therapeutic mode
print("\nTesting therapeutic effector modes...")

for mode in THERAPY_CFG.therapeutic_modes:
    print(f"\nTesting {mode}...")
    try:
        # Create effector
        effector = TherapeuticEffectorGene(mode)
        
        # Create test patient state
        biomarkers = torch.randn(THERAPY_CFG.num_biomarkers)
        encoded_state = torch.randn(cfg.hidden_dim)
        
        patient_state = {
            'biomarkers': biomarkers,
            'encoded_state': encoded_state,
            'disease_probabilities': torch.randn(10),
            'disease_severity': 0.7,
            'critical_conditions': {},
            'inflammatory_score': 0.5,
            'metabolic_score': 0.5
        }
        
        # Test therapeutic generation
        result = effector.generate_therapeutic(patient_state)
        
        print(f"  ✓ {mode}: therapeutic shape = {result['therapeutic'].shape}")
        print(f"    dose = {result['dose'].item():.3f}")
        print(f"    safety = {result['safety_score'].item():.3f}")
        
    except Exception as e:
        print(f"  ✗ {mode} failed: {str(e)}")
        traceback.print_exc()

print("\n✅ Debug complete")