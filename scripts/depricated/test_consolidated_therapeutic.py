#!/usr/bin/env python3
"""
Test Consolidated Therapeutic System
===================================
Quick test to verify the consolidated domain structure works correctly
"""

import sys
import os

# Add the domains directory to path
domains_path = os.path.join(os.path.dirname(__file__), 'domains')
sys.path.insert(0, domains_path)

try:
    # Test imports
    print("🧪 Testing Consolidated Therapeutic System")
    print("=" * 50)
    
    print("1. Testing configuration import...")
    from living_therapeutics_system import THERAPY_CFG, get_device
    print(f"   ✅ Config loaded: {THERAPY_CFG.num_biomarkers} biomarkers")
    
    print("2. Testing gene imports...")
    from living_therapeutics_system import (
        BiosensorGene, TherapeuticEffectorGene, 
        AdaptiveControllerGene, TherapeuticStemGene
    )
    print("   ✅ All gene classes imported")
    
    print("3. Testing cell management...")
    from living_therapeutics_system import create_living_therapeutic_population
    print("   ✅ Cell management functions imported")
    
    print("4. Testing main system...")
    from living_therapeutics_system import LivingTherapeuticSystem
    print("   ✅ Main system class imported")
    
    print("5. Testing runner functions...")
    from living_therapeutics_system import quick_start
    print("   ✅ Runner functions imported")
    
    print("\n6. Testing system creation...")
    from living_therapeutics_system import create_patient_profile
    
    # Create test patient
    patient = create_patient_profile('autoimmune', severity=0.6)
    print(f"   ✅ Patient profile created: {patient['id']}")
    
    # Create therapeutic system
    system = LivingTherapeuticSystem(patient)
    print(f"   ✅ Therapeutic system created with {len(system.population)} cells")
    
    print("\n7. Testing single treatment cycle...")
    result = system.run_treatment_cycle()
    print(f"   ✅ Treatment cycle completed")
    print(f"   📊 Disease severity: {result['patient_state'].get('disease_severity', 0):.3f}")
    print(f"   📊 Treatment efficacy: {result['response']['efficacy_score']:.3f}")
    print(f"   📊 Safety score: {result['response']['safety_score']:.3f}")
    
    print("\n8. Testing package info...")
    from living_therapeutics_system import get_domain_info
    info = get_domain_info()
    print(f"   ✅ Domain: {info['name']} v{info['version']}")
    print(f"   📋 Components: {len(info['components'])}")
    print(f"   🎯 Capabilities: {len(info['capabilities'])}")
    
    print(f"\n{'='*50}")
    print("🎉 ALL TESTS PASSED!")
    print("🎯 Consolidated domain structure is working correctly")
    print(f"{'='*50}")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)