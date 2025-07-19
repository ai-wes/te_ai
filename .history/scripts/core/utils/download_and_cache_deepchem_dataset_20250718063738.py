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

    for dataset_name in datasets_to_prepare:
        logger.info(f"\n--- Preparing '{dataset_name}' ---")
        print(f"\n--- Preparing '{dataset_name}' ---")
        try:
            # The act of calling the loader function with default parameters
            # is enough to trigger the download and caching process.
            loader_fn = getattr(dc.molnet, f'load_{dataset_name.lower()}')
            
            # We don't need the data itself, just the action of loading it.
            # We pass a dummy featurizer to make it fast.
            loader_fn(featurizer='ECFP', splitter='scaffold')
            
            logger.info(f"✅ Successfully downloaded and cached '{dataset_name}'.")
            print(f"✅ Successfully downloaded and cached '{dataset_name}'.")
        except AttributeError:
            logger.error(f"❌ Failed: No loader function found for '{dataset_name}'. Is the name correct?")
            print(f"❌ Failed: No loader function found for '{dataset_name}'. Is the name correct?")
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