"""Fix for DeepChem import error with torchao"""

import sys

# Monkey patch to fix the torchao error
class FakeTag:
    needs_fixed_stride_order = None

# Inject the fix before importing
if 'torch' in sys.modules:
    import torch
    if not hasattr(torch._C.Tag, 'needs_fixed_stride_order'):
        torch._C.Tag.needs_fixed_stride_order = None

# Now try importing deepchem
try:
    import deepchem as dc
    print("DeepChem imported successfully!")
except Exception as e:
    print(f"Still failed: {e}")