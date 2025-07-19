# scripts/prepare_datasets.py

import os
import deepchem as dc
import logging

logger = logging.getLogger(__name__)

def download_and_cache_deepchem_datasets(datasets_to_prepare: list = None, data_dir: str = None):
    """
    Explicitly downloads and caches specified DeepChem datasets to prevent
    download errors during training runs.
    
    Args:
        datasets_to_prepare: List of dataset names to download
        data_dir: Custom directory to store DeepChem data (optional)
    """
    if datasets_to_prepare is None:
        # Use the correct DeepChem MoleculeNet loader function names
        datasets_to_prepare = ['bbbp', 'tox21', 'sider', 'hiv', 'bace_classification']

    # Set custom data directory if provided
    if data_dir:
        dc.utils.set_data_dir(data_dir)
        logger.info(f"Set DeepChem data directory to: {data_dir}")
        print(f"Set DeepChem data directory to: {data_dir}")
    else:
        # Use a better default location than /tmp
        default_dir = os.path.expanduser("~/deepchem_data")
        try:
            os.makedirs(default_dir, exist_ok=True)
            dc.utils.set_data_dir(default_dir)
            logger.info(f"Set DeepChem data directory to: {default_dir}")
            print(f"Set DeepChem data directory to: {default_dir}")
        except Exception as e:
            logger.warning(f"Could not set custom data directory: {e}. Using default.")
            print(f"Could not set custom data directory: {e}. Using default.")

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
            
            # Try different featurizer approaches
            featurizer_attempts = []
            
            # Try MorganGenerator first (as suggested by deprecation warnings)
            if hasattr(dc.feat, 'MorganGenerator'):
                try:
                    featurizer_attempts.append(('MorganGenerator', dc.feat.MorganGenerator(radius=2, fpSize=2048)))
                except Exception as e:
                    logger.debug(f"MorganGenerator failed to instantiate: {e}")
            
            # Try MorganFingerprint as fallback
            if hasattr(dc.feat, 'MorganFingerprint'):
                try:
                    featurizer_attempts.append(('MorganFingerprint', dc.feat.MorganFingerprint(radius=2, fpSize=2048)))
                except Exception as e:
                    logger.debug(f"MorganFingerprint failed to instantiate: {e}")
            
            # Try CircularFingerprint as another option
            if hasattr(dc.feat, 'CircularFingerprint'):
                try:
                    featurizer_attempts.append(('CircularFingerprint', dc.feat.CircularFingerprint(size=2048, radius=2)))
                except Exception as e:
                    logger.debug(f"CircularFingerprint failed to instantiate: {e}")
            
            splitter = 'scaffold'
            success = False
            
            # Try each featurizer
            for featurizer_name, featurizer in featurizer_attempts:
                try:
                    logger.info(f"   Trying {featurizer_name} featurizer")
                    print(f"   Trying {featurizer_name} featurizer")
                    
                    # Load the dataset with current featurizer
                    result = loader_fn(featurizer=featurizer, splitter=splitter)
                    
                    # Handle different return formats
                    if isinstance(result, tuple):
                        if len(result) == 2:
                            datasets, transformers = result
                            logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' with {featurizer_name} (2-tuple return).")
                            print(f"✅ Successfully downloaded and cached '{dataset_name}' with {featurizer_name} (2-tuple return).")
                        elif len(result) == 3:
                            datasets, transformers, tasks = result
                            logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' with {featurizer_name} (3-tuple return).")
                            print(f"✅ Successfully downloaded and cached '{dataset_name}' with {featurizer_name} (3-tuple return).")
                        else:
                            logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' with {featurizer_name} (tuple return).")
                            print(f"✅ Successfully downloaded and cached '{dataset_name}' with {featurizer_name} (tuple return).")
                    else:
                        logger.info(f"✅ Successfully downloaded and cached '{dataset_name}' with {featurizer_name} (single return).")
                        print(f"✅ Successfully downloaded and cached '{dataset_name}' with {featurizer_name} (single return).")
                    
                    success = True
                    break
                    
                except Exception as featurizer_error:
                    logger.debug(f"   {featurizer_name} failed: {featurizer_error}")
                    continue
            
            # If all featurizers failed, try without specifying featurizer
            if not success:
                logger.info(f"   All featurizers failed, trying without specifying featurizer...")
                print(f"   All featurizers failed, trying without specifying featurizer...")
                
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
                    raise Exception(f"All approaches failed. Last error: {legacy_error}")
                    
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
    # Optionally specify a custom data directory:
    # download_and_cache_deepchem_datasets(data_dir="/path/to/your/data/directory")
    download_and_cache_deepchem_datasets()
    print("Done")