# scripts/prepare_datasets.py

import os
import deepchem as dc
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

def download_and_cache_deepchem_datasets(datasets_to_prepare: list = None):
    """
    Explicitly downloads and caches specified DeepChem datasets to prevent
    download errors during training runs.
    """
    if datasets_to_prepare is None:
        # Use the correct DeepChem MoleculeNet loader function names
        datasets_to_prepare = ['bbbp', 'tox21', 'sider', 'hiv', 'bace_classification']

    logger.info("="*60)
    logger.info("DEEPCHEM DATASET PREPARATION SCRIPT")
    logger.info("="*60)
    logger.info(f"Attempting to download and cache the following datasets: {datasets_to_prepare}")
    logger.info(f"DeepChem data will be stored in: {dc.utils.get_data_dir()}")

    print("="*60)
    print("DEEPCHEM DATASET PREPARATION SCRIPT")
    print("="*60)
    print(f"Attempting to download and cache the following datasets: {datasets_to_prepare}")
    print(f"DeepChem data will be stored in: {dc.utils.get_data_dir()}")

    # Map dataset names to their correct loader function names
    dataset_mapping = {
        'bbbp': 'load_bbbp',
        'tox21': 'load_tox21', 
        'sider': 'load_sider',
        'hiv': 'load_hiv',
        'bace_classification': 'load_bace_classification'
    }

    for dataset_name in datasets_to_prepare:
        logger.info(f"\n--- Preparing '{dataset_name}' ---")
        print(f"\n--- Preparing '{dataset_name}' ---")
        
        # Get the correct loader function name
        loader_name = dataset_mapping.get(dataset_name, f'load_{dataset_name.lower()}')
        
        try:
            # Check if the loader function exists
            if not hasattr(dc.molnet, loader_name):
                logger.error(f"❌ Failed: No loader function '{loader_name}' found in dc.molnet")
                print(f"❌ Failed: No loader function '{loader_name}' found in dc.molnet")
                continue
                
            loader_fn = getattr(dc.molnet, loader_name)
            
            # Use the modern MorganGenerator to avoid deprecation warnings
            featurizer = dc.feat.MorganGenerator(radius=2, fpSize=2048)
            splitter = 'scaffold'
            
            try:
                logger.info(f"   Using MorganGenerator featurizer")
                print(f"   Using MorganGenerator featurizer")
                
                # Load the dataset with modern featurizer
                result = loader_fn(featurizer=featurizer, splitter=splitter)
                
                # Handle different return formats
                if isinstance(result, tuple):
                    if len(result) == 2:
                        datasets, transformers = result
                        logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' (2-tuple return).")
                        print(f"✅ Successfully downloaded and cached '{dataset_name}' (2-tuple return).")
                    elif len(result) == 3:
                        datasets, transformers, tasks = result
                        logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' (3-tuple return).")
                        print(f"✅ Successfully downloaded and cached '{dataset_name}' (3-tuple return).")
                    else:
                        logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' (tuple return).")
                        print(f"✅ Successfully downloaded and cached '{dataset_name}' (tuple return).")
                else:
                    logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' (single return).")
                    print(f"✅ Successfully downloaded and cached '{dataset_name}' (single return).")
                    
            except Exception as modern_error:
                logger.info(f"   Modern featurizer failed, trying legacy approach...")
                print(f"   Modern featurizer failed, trying legacy approach...")
                
                # Fallback to legacy approach without specifying featurizer
                try:
                    result = loader_fn(splitter=splitter)
                    
                    # Handle different return formats
                    if isinstance(result, tuple):
                        if len(result) == 2:
                            datasets, transformers = result
                            logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' with default featurizer (2-tuple).")
                            print(f"✅ Successfully downloaded and cached '{dataset_name}' with default featurizer (2-tuple).")
                        elif len(result) == 3:
                            datasets, transformers, tasks = result
                            logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' with default featurizer (3-tuple).")
                            print(f"✅ Successfully downloaded and cached '{dataset_name}' with default featurizer (3-tuple).")
                        else:
                            logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' with default featurizer (tuple).")
                            print(f"✅ Successfully downloaded and cached '{dataset_name}' with default featurizer (tuple).")
                    else:
                        logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' with default featurizer (single).")
                        print(f"✅ Successfully downloaded and cached '{dataset_name}' with default featurizer (single).")
                        
                except Exception as legacy_error:
                    raise Exception(f"Both modern and legacy approaches failed. Modern error: {modern_error}. Legacy error: {legacy_error}")
                    
        except Exception as e:
            logger.error(f"❌ An error occurred while preparing '{dataset_name}': {e}")
            logger.error("   Please check your internet connection and permissions for the data directory.")
            print(f"❌ An error occurred while preparing '{dataset_name}': {e}")
            print("   Please check your internet connection and permissions for the data directory.")
            
    logger.info("\nDataset preparation complete.")
    logger.info("You can now run your training scripts.")
    print("\nDataset preparation complete.")
    print("You can now run your training scripts.")

if __name__ == "__main__":
    # You can run this script directly to pre-download all necessary data
    download_and_cache_deepchem_datasets()
    print("Done")