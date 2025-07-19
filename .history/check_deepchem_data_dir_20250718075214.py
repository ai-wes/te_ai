#!/usr/bin/env python3

import deepchem as dc
import os

print("DeepChem Data Directory Configuration")
print("=" * 40)

# Check current data directory
current_dir = dc.utils.get_data_dir()
print(f"Current DeepChem data directory: {current_dir}")

# Check if the directory exists and is writable
if os.path.exists(current_dir):
    print(f"✅ Directory exists: {current_dir}")
    if os.access(current_dir, os.W_OK):
        print(f"✅ Directory is writable")
    else:
        print(f"❌ Directory is NOT writable")
else:
    print(f"❌ Directory does not exist: {current_dir}")

# Show how to set a custom data directory
print("\nTo set a custom data directory, you can:")
print("1. Set the DEEPCHEM_DATA_DIR environment variable:")
print("   export DEEPCHEM_DATA_DIR=/path/to/your/data/directory")
print("2. Or set it programmatically:")
print("   dc.utils.set_data_dir('/path/to/your/data/directory')")

# Suggest a better location
suggested_dir = os.path.expanduser("~/deepchem_data")
print(f"\nSuggested data directory: {suggested_dir}")

# Check if we can create the suggested directory
try:
    os.makedirs(suggested_dir, exist_ok=True)
    print(f"✅ Successfully created/verified suggested directory: {suggested_dir}")
    
    # Test if we can write to it
    test_file = os.path.join(suggested_dir, "test_write.txt")
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print(f"✅ Directory is writable")
    
except Exception as e:
    print(f"❌ Cannot create/use suggested directory: {e}")

print("\nDone!") 