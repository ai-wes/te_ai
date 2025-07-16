#!/usr/bin/env python3
"""Minimal test for therapeutic system"""

import torch
import sys

print("Testing therapeutic system initialization...")

# Test imports
try:
    from living_therapeutic_system import TherapeuticConfig, THERAPY_CFG
    print("✓ Config imported")
except Exception as e:
    print(f"✗ Config import failed: {e}")
    sys.exit(1)

try:
    from living_therapeutic_system import BiosensorGene, TherapeuticEffectorGene
    print("✓ Gene classes imported")
except Exception as e:
    print(f"✗ Gene import failed: {e}")
    sys.exit(1)

try:
    from living_therapeutic_system import LivingTherapeuticSystem
    print("✓ System class imported")
except Exception as e:
    print(f"✗ System import failed: {e}")
    sys.exit(1)

# Test basic initialization
try:
    patient_profile = {
        'id': 'TEST001',
        'disease': 'autoimmune_inflammatory',
        'severity': 0.8,
        'age': 45,
        'comorbidities': []
    }
    
    print("\nInitializing therapeutic system...")
    system = LivingTherapeuticSystem(patient_profile)
    print("✓ System initialized successfully")
    print(f"  Population size: {len(system.population)}")
    
    # Test biomarker generation
    print("\nTesting biomarker generation...")
    biomarkers = torch.randn(THERAPY_CFG.num_biomarkers)
    if torch.cuda.is_available():
        biomarkers = biomarkers.cuda()
    print(f"✓ Generated biomarkers: shape={biomarkers.shape}, device={biomarkers.device}")
    
    # Test patient assessment
    print("\nTesting patient assessment...")
    patient_state = system._comprehensive_patient_assessment(biomarkers)
    print(f"✓ Patient state keys: {list(patient_state.keys())}")
    
    # Test therapeutic response
    print("\nTesting therapeutic response...")
    response = system._generate_population_response(patient_state)
    print(f"✓ Response keys: {list(response.keys())}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)