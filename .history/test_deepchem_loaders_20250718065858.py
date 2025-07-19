#!/usr/bin/env python3

import deepchem as dc

# List all available loader functions in dc.molnet
print("Available DeepChem MoleculeNet loader functions:")
print("=" * 50)

# Get all attributes of dc.molnet that start with 'load_'
loader_functions = [attr for attr in dir(dc.molnet) if attr.startswith('load_')]

for loader in sorted(loader_functions):
    print(f"  {loader}")

print(f"\nTotal loader functions found: {len(loader_functions)}")

# Test a few specific ones
test_loaders = ['load_bbbp', 'load_tox21', 'load_sider', 'load_hiv', 'load_bace_classification']

print("\nTesting specific loaders:")
print("=" * 30)

for loader_name in test_loaders:
    if hasattr(dc.molnet, loader_name):
        print(f"✅ {loader_name} - AVAILABLE")
    else:
        print(f"❌ {loader_name} - NOT FOUND")

print("\nDone!") 