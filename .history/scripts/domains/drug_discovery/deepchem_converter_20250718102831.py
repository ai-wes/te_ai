"""
DeepChem to TE-AI Data Converter
================================

Converts DeepChem molecular datasets into TE-AI compatible DrugTargetAntigen objects.
Supports multiple featurization schemes and preserves data splits.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union, Any
import deepchem as dc
from deepchem.feat import Featurizer
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from scripts.domains.drug_discovery.drug_target_evaluator import DrugTargetAntigen
from scripts.core.utils.detailed_logger import get_logger
from scripts.config import cfg
from rdkit import RDLogger

# Suppress RDKit verbose error messages for a cleaner log
RDLogger.DisableLog('rdApp.*')

logger = get_logger()


class DeepChemToTEAI:
    """Converts DeepChem datasets to TE-AI format"""
    
    def __init__(self, featurization_mode: str = 'hybrid'):
        """
        Args:
            featurization_mode: 'simple', 'hybrid', or specific featurizer name
        """
        self.featurization_mode = featurization_mode
        self.device = torch.device(cfg.device)
        
    def convert_dataset(
        self, 
        deepchem_dataset: dc.data.Dataset,
        task_name: str = None
    ) -> List[DrugTargetAntigen]:
        """
        Convert a DeepChem dataset to TE-AI antigens
        
        Args:
            deepchem_dataset: DeepChem dataset object
            task_name: Specific task to use (for multitask datasets)
            
        Returns:
            List of DrugTargetAntigen objects
        """
        antigens = []
        
        # Get task index
        tasks = deepchem_dataset.tasks
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()  # Convert numpy array to list
        
        if task_name and task_name in tasks:
            task_idx = tasks.index(task_name)
        else:
            task_idx = 0  # Use first task by default
            
        logger.info(f"Converting {len(deepchem_dataset)} molecules from DeepChem")
        
        for i in range(len(deepchem_dataset)):
            try:
                # Get molecular data
                mol_features = deepchem_dataset.X[i]
                label = float(deepchem_dataset.y[i, task_idx])
                mol_id = deepchem_dataset.ids[i]
                
                # Create antigen based on featurization mode
                if self.featurization_mode == 'hybrid':
                    antigen = self._create_hybrid_antigen(
                        mol_features, label, mol_id, deepchem_dataset
                    )
                else:
                    antigen = self._create_simple_antigen(
                        mol_features, label, mol_id
                    )
                    
                antigens.append(antigen)
                
            except Exception as e:
                logger.warning(f"Failed to convert molecule {i}: {e}")
                continue
                
        logger.info(f"Successfully converted {len(antigens)} molecules")
        return antigens
    
    def _create_simple_antigen(
        self,
        features: np.ndarray,
        label: float,
        mol_id: str
    ) -> DrugTargetAntigen:
        """Create a simple antigen from features"""
        
        # Convert features to molecular properties
        if isinstance(features, np.ndarray):
            # For fingerprint features, create pseudo-properties
            mol_weight = np.sum(features[:100]) * 10 + 200  # Pseudo MW
            logp = np.mean(features[100:200]) * 5  # Pseudo LogP
            num_h_donors = int(np.sum(features[200:210]))
            num_h_acceptors = int(np.sum(features[210:220]))
        else:
            # For graph features, extract from the graph
            mol_weight = 300.0  # Default
            logp = 2.5
            num_h_donors = 2
            num_h_acceptors = 3
            
        # Create a minimal protein structure for the molecule
        from scripts.domains.drug_discovery.drug_target_antigen import ProteinStructure, BindingPocket
        
        protein_structure = ProteinStructure(
            sequence="MOLECULE",  # Placeholder for small molecule
            coordinates=np.zeros((10, 3)),  # Minimal coordinates
            secondary_structure="C" * 8  # Coil structure
        )
        
        # Create a synthetic binding pocket with molecular properties
        binding_pocket = BindingPocket(
            pocket_id=mol_id,
            residue_indices=list(range(8)),
            volume=mol_weight * 1.5,  # Approximate volume from MW
            hydrophobicity=logp / 5.0,  # Normalize LogP
            electrostatic_potential=0.0,
            druggability_score=label,  # Use label as druggability
            known_ligands=[]
        )
        
        antigen = DrugTargetAntigen(
            protein_structure=protein_structure,
            binding_pockets=[binding_pocket],
            disease_association="BBBP"  # Blood-brain barrier penetration
        )
        
        # Store original features for the model
        antigen._deepchem_features = features
        
        return antigen
    
    def _create_hybrid_antigen(
        self,
        features: np.ndarray,
        label: float,
        mol_id: str,
        dataset: dc.data.Dataset
    ) -> DrugTargetAntigen:
        """Create a hybrid antigen with multiple representations"""
        
        # Try to get SMILES from mol_id
        smiles = mol_id if Chem.MolFromSmiles(mol_id) else None
        
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            # Calculate real molecular properties
            mol_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            num_h_donors = Lipinski.NumHDonors(mol)
            num_h_acceptors = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            num_rotatable = Descriptors.NumRotatableBonds(mol)
        else:
            # Fallback to pseudo-properties
            mol_weight = 300.0
            logp = 2.5
            num_h_donors = 2
            num_h_acceptors = 3
            tpsa = 60.0
            num_rotatable = 4
            
        antigen = DrugTargetAntigen(
            target_id=mol_id,
            sequence="HYBRID",
            binding_sites=[],
            known_drugs=[],
            disease_association="Benchmark",
            druggability_score=label,
            molecular_weight=mol_weight,
            logp=logp,
            num_h_donors=num_h_donors,
            num_h_acceptors=num_h_acceptors,
            tpsa=tpsa,
            num_rotatable_bonds=num_rotatable
        )
        
        # Store multiple representations
        antigen._representations = {
            'deepchem_features': features,
            'smiles': smiles,
            'label': label
        }
        
        return antigen

    def convert_molnet_dataset(
        self,
        dataset_name: str,
        featurizer: str = 'ECFP',
        splitter: str = 'scaffold'
    ) -> Tuple[List[DrugTargetAntigen], List[DrugTargetAntigen], List[DrugTargetAntigen]]:
        """
        Load and convert a MoleculeNet dataset with robust featurization and pre-filtering.
        """
        logger.info(f"Loading {dataset_name} from MoleculeNet...")

        # --- NEW: Robust Featurizer Selection Logic ---
        base_featurizer = None
        featurizer_name = "Unknown"
        expected_size = 1024  # Default for ECFP/Morgan

        if featurizer == 'ECFP':
            try:
                # Use DeepChem's CircularFingerprint (which is Morgan fingerprints)
                from deepchem.feat import CircularFingerprint
                base_featurizer = CircularFingerprint(size=expected_size, radius=2)
                featurizer_name = "CircularFingerprint (Morgan)"
                logger.info("Successfully initialized CircularFingerprint (Morgan fingerprints).")
            except (ImportError, Exception) as e:
                logger.error(f"FATAL: Could not import CircularFingerprint from DeepChem: {e}")
                raise RuntimeError(f"Could not load {dataset_name} because essential featurizers are missing.")
        else:
            # For other featurizer types, this would need to be expanded
            raise NotImplementedError(f"Featurizer type '{featurizer}' is not yet supported in this robust pipeline.")

        # Wrap the chosen base featurizer with our sanitizing logic
        sanitizing_wrapper = RobustFeaturizer(
            base_featurizer=base_featurizer,
            expected_size=expected_size
        )
        logger.info(f"Using RobustFeaturizer wrapping {featurizer_name}.")
        
        # --- The rest of the pipeline remains the same robust version ---
        data_dir = dc.utils.get_data_dir()
        logger.info(f"Using DeepChem data directory: {data_dir}")
        
        # 1. Load raw data with a DummyFeaturizer to allow for cleaning
        loader_fn = getattr(dc.molnet, f'load_{dataset_name.lower()}')
        tasks, (raw_dataset,), transformers = loader_fn(
            featurizer=dc.feat.DummyFeaturizer(),
            splitter=None,
            data_dir=data_dir
        )

        # 2. Filter out invalid molecules that RDKit cannot parse
        valid_indices = [i for i, sm in enumerate(raw_dataset.ids) if Chem.MolFromSmiles(sm) is not None]
        invalid_count = len(raw_dataset) - len(valid_indices)
        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} invalid SMILES strings from '{dataset_name}'.")
        
        cleaned_dataset = raw_dataset.select(valid_indices)

        # --- PROCEED WITH CLEANED DATA AND ROBUST FEATURIZER ---
        logger.info("Step 3: Featurizing the cleaned dataset...")
        # We must featurize *before* splitting for the scaffold splitter to work correctly.
        cleaned_dataset.reshard(shard_size=8192) # Resharding can help with memory
        featurized_dataset = cleaned_dataset.featurize(sanitizing_wrapper, log_every_n=1000)

        logger.info("Step 4: Splitting the featurized dataset...")
        splitter_fn = dc.splits.ScaffoldSplitter()
        train_data, valid_data, test_data = splitter_fn.train_valid_test_split(featurized_dataset)
        
        logger.info("Step 5: Applying data transformations...")
        for transformer in transformers:
            train_data = transformer.transform(train_data)
            valid_data = transformer.transform(valid_data)
            test_data = transformer.transform(test_data)
        
        # Convert each split to TE-AI Antigens
        train_antigens = self.convert_dataset(train_data, tasks[0])
        valid_antigens = self.convert_dataset(valid_data, tasks[0])
        test_antigens = self.convert_dataset(test_data, tasks[0])
        
        logger.info(f"Successfully converted {dataset_name}:")
        logger.info(f"  Train: {len(train_antigens)} antigens")
        logger.info(f"  Valid: {len(valid_antigens)} antigens")
        logger.info(f"  Test: {len(test_antigens)} antigens")
        
        return train_antigens, valid_antigens, test_antigens





class HybridDrugAntigen(DrugTargetAntigen):
    """
    Enhanced antigen that can use multiple molecular representations
    """
    
    def __init__(self, base_antigen: DrugTargetAntigen, featurizers: List):
        """
        Create a hybrid antigen with multiple featurizations
        
        Args:
            base_antigen: Base DrugTargetAntigen
            featurizers: List of DeepChem featurizers
        """
        # Copy base properties
        super().__init__(
            target_id=base_antigen.target_id,
            sequence=base_antigen.sequence,
            binding_sites=base_antigen.binding_sites,
            known_drugs=base_antigen.known_drugs,
            disease_association=base_antigen.disease_association,
            druggability_score=base_antigen.druggability_score
        )
        
        # Store multiple representations
        self.representations = {}
        if hasattr(base_antigen, '_representations'):
            smiles = base_antigen._representations.get('smiles')
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                for featurizer in featurizers:
                    try:
                        features = featurizer.featurize([mol])[0]
                        self.representations[featurizer.__class__.__name__] = features
                    except Exception as e:
                        logger.warning(f"Failed to featurize with {featurizer}: {e}")
                        
                        
                        
                        
                        
                        
    def to_graph(self) -> Data:
        """Convert to graph with hybrid features"""
        # Get base graph
        graph = super().to_graph()
        
        # Add all representations as additional features
        if self.representations:
            # Concatenate all features
            all_features = []
            for name, features in self.representations.items():
                if isinstance(features, np.ndarray):
                    all_features.append(torch.tensor(features, dtype=torch.float32))
                    
            if all_features:
                graph.hybrid_features = torch.cat(all_features, dim=-1)
                
        return graph


def prepare_deepchem_benchmarks(datasets: List[str] = None) -> Dict[str, Tuple]:
    """
    Prepare multiple DeepChem datasets for benchmarking
    
    Args:
        datasets: List of dataset names (default: ['bbbp', 'tox21', 'hiv'])
        
    Returns:
        Dictionary mapping dataset names to (train, valid, test) antigen tuples
    """
    if datasets is None:
        datasets = ['bbbp', 'tox21', 'hiv', 'bace_classification']
        
    converter = DeepChemToTEAI(featurization_mode='hybrid')
    benchmark_data = {}
    
    for dataset_name in datasets:
        try:
            train, valid, test = converter.convert_molnet_dataset(dataset_name)
            benchmark_data[dataset_name] = (train, valid, test)
            logger.info(f"Prepared {dataset_name} for benchmarking")
        except Exception as e:
            logger.error(f"Failed to prepare {dataset_name}: {e}")
            
    return benchmark_data







class RobustFeaturizer(Featurizer):
    """
    A wrapper around a DeepChem featurizer that handles featurization failures
    by returning a zero vector of the expected size. This prevents errors
    with inhomogeneous array shapes during dataset loading.
    """
    def __init__(self, base_featurizer: Featurizer, expected_size: int):
        """
        Args:
            base_featurizer: The DeepChem featurizer to wrap (e.g., MorganGenerator).
            expected_size: The expected length of the feature vector for error cases.
        """
        self.base_featurizer = base_featurizer
        self.expected_size = expected_size
        super().__init__()

    def _featurize(self, datapoint: Any, **kwargs) -> np.ndarray:
        """
        Featurizes a single datapoint by first converting it to a Mol object.
        """
        # 1. Sanitize the input: Convert SMILES string to RDKit Mol object
        mol = None
        if isinstance(datapoint, str):
            mol = Chem.MolFromSmiles(datapoint)
        elif isinstance(datapoint, Chem.Mol):
            mol = datapoint
        
        # 2. Handle invalid molecules
        if mol is None:
            logger.warning(f"Could not create a valid molecule from datapoint: '{datapoint}'. Returning zero vector.")
            return np.zeros(self.expected_size, dtype=float)

        # 3. Featurize the now-guaranteed-valid Mol object
        try:
            # The base featurizer now receives the correct input type
            features = self.base_featurizer._featurize(mol, **kwargs)
            
            # Ensure the output is a flat numpy array of the correct size
            if isinstance(features, np.ndarray):
                features = features.flatten()
                if features.shape[0] == self.expected_size:
                    return features
                else:
                    # Pad or truncate if necessary (should be rare with fingerprints)
                    padded = np.zeros(self.expected_size, dtype=float)
                    size = min(len(features), self.expected_size)
                    padded[:size] = features[:size]
                    return padded
            else:
                # Fallback if featurizer returns something unexpected
                logger.warning(f"Featurizer returned non-array type for {datapoint}. Returning zero vector.")
                return np.zeros(self.expected_size, dtype=float)

        except Exception as e:
            logger.error(f"Featurization failed for a valid molecule. Error: {e}. Returning zero vector.")
            return np.zeros(self.expected_size, dtype=float)







if __name__ == "__main__":
    # Test the converter
    converter = DeepChemToTEAI()
    
    # Load BBBP dataset
    train, valid, test = converter.convert_molnet_dataset('bbbp')
    
    print(f"BBBP Dataset converted:")
    print(f"  Train: {len(train)} antigens")
    print(f"  Valid: {len(valid)} antigens")
    print(f"  Test: {len(test)} antigens")
    
    # Check an antigen
    if train:
        antigen = train[0]
        print(f"\nFirst antigen:")
        print(f"  ID: {antigen.target_id}")
        print(f"  Druggability: {antigen.druggability_score}")
        print(f"  MW: {antigen.molecular_weight}")
        print(f"  LogP: {antigen.logp}")