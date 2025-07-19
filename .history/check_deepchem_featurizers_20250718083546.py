#!/usr/bin/env python3

import deepchem as dc

print("DeepChem Featurizers Available")
print("=" * 40)

# Check what's available in dc.feat
featurizers = [attr for attr in dir(dc.feat) if not attr.startswith('_')]

print("Available featurizers in dc.feat:")
for featurizer in sorted(featurizers):
    print(f"  {featurizer}")

# Check specifically for Morgan-related featurizers
print("\nMorgan-related featurizers:")
morgan_featurizers = [f for f in featurizers if 'morgan' in f.lower() or 'morgan' in f.lower()]
for featurizer in morgan_featurizers:
    print(f"  {featurizer}")

# Check for CircularFingerprint
print("\nCircularFingerprint info:")
if hasattr(dc.feat, 'CircularFingerprint'):
    print("  ✅ CircularFingerprint is available")
    try:
        cf = dc.feat.CircularFingerprint(size=2048, radius=2)
        print("  ✅ CircularFingerprint can be instantiated")
    except Exception as e:
        print(f"  ❌ CircularFingerprint instantiation failed: {e}")
else:
    print("  ❌ CircularFingerprint is NOT available")

# Check for MorganGenerator
print("\nMorganGenerator info:")
if hasattr(dc.feat, 'MorganGenerator'):
    print("  ✅ MorganGenerator is available")
    try:
        mg = dc.feat.MorganGenerator(radius=2, fpSize=2048)
        print("  ✅ MorganGenerator can be instantiated")
    except Exception as e:
        print(f"  ❌ MorganGenerator instantiation failed: {e}")
else:
    print("  ❌ MorganGenerator is NOT available")

# Check for MorganFingerprint
print("\nMorganFingerprint info:")
if hasattr(dc.feat, 'MorganFingerprint'):
    print("  ✅ MorganFingerprint is available")
    try:
        mf = dc.feat.MorganFingerprint(radius=2, fpSize=2048)
        print("  ✅ MorganFingerprint can be instantiated")
    except Exception as e:
        print(f"  ❌ MorganFingerprint instantiation failed: {e}")
else:
    print("  ❌ MorganFingerprint is NOT available")

print("\nDone!") 