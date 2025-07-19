#!/usr/bin/env python3

import deepchem as dc
import os

print("🔍 DeepChem Data Storage Location")
print("=" * 40)

# Show current data directory
current_dir = dc.utils.get_data_dir()
print(f"📍 Current DeepChem data directory: {current_dir}")

# Check if it exists and is writable
if os.path.exists(current_dir):
    print(f"✅ Directory exists")
    if os.access(current_dir, os.W_OK):
        print(f"✅ Directory is writable")
    else:
        print(f"❌ Directory is NOT writable")
else:
    print(f"❌ Directory does not exist")

# Show how to change it
print("\n🛠️  How to change the data directory:")
print("1. Environment variable:")
print("   export DEEPCHEM_DATA_DIR=/your/custom/path")
print("2. Programmatically:")
print("   dc.utils.set_data_dir('/your/custom/path')")

# Suggest better locations
print("\n💡 Suggested better locations:")
home_dir = os.path.expanduser("~")
suggestions = [
    os.path.join(home_dir, "deepchem_data"),
    os.path.join(home_dir, "data", "deepchem"),
    "/opt/deepchem_data",
    "./deepchem_data"
]

for i, suggestion in enumerate(suggestions, 1):
    print(f"   {i}. {suggestion}")

print("\n📝 To use a custom directory, modify the script call:")
print("   download_and_cache_deepchem_datasets(data_dir='/your/custom/path')")

print("\nDone! 🎉") 