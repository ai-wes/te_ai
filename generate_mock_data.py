import numpy as np
import os

# Create the directory if it doesn't exist
output_dir = '/mnt/c/Users/wes/desktop/te_ai/maia_tcga_pancan'
os.makedirs(output_dir, exist_ok=True)

# Generate 10 mock sample files
for i in range(10):
    sample_id = f'sample_{i}'
    cancer_type = np.random.choice(['BRCA', 'LUSC', 'OV'])
    transcriptomics = np.random.rand(20530)
    genomics_mutations = np.random.randint(0, 2, size=40543)
    
    np.savez_compressed(
        os.path.join(output_dir, f'{sample_id}.npz'),
        sample_id=sample_id,
        cancer_type=cancer_type,
        transcriptomics=transcriptomics,
        genomics_mutations=genomics_mutations
    )

print('Mock data generation complete.')