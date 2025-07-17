"""
TCGA Data Converter for Drug Discovery
======================================

Converts TCGA multi-omics data from NPZ format into DrugTargetAntigen
objects for TE-AI processing. Handles transcriptomics, proteomics,
mutations, and clinical data.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple
import os
import json
from pathlib import Path
import pickle
from collections import defaultdict

from .drug_target_antigen import DrugTargetAntigen, BindingPocket, ProteinStructure, AntigenEpitope
from .omics_to_antigen_converter import OmicsData, OmicsToAntigenConverter
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()


class TCGADataConverter:
    """
    Converts TCGA multi-omics data to drug target antigens.
    Leverages the comprehensive TCGA dataset for drug discovery.
    """
    
    def __init__(
        self,
        tcga_data_dir: str,
        gene_mapping_file: Optional[str] = None,
        cancer_focus: Optional[str] = None,
        min_samples_per_gene: int = 100
    ):
        """
        Initialize TCGA converter.
        
        Args:
            tcga_data_dir: Path to TCGA NPZ files
            gene_mapping_file: Optional gene symbol mapping
            cancer_focus: Focus on specific cancer type
            min_samples_per_gene: Minimum samples required per gene
        """
        self.tcga_data_dir = Path(tcga_data_dir)
        self.cancer_focus = cancer_focus
        self.min_samples_per_gene = min_samples_per_gene
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load gene mapping if available
        self.gene_mapping = self._load_gene_mapping(gene_mapping_file)
        
        # Cache for loaded data
        self._sample_cache = {}
        self._expression_matrix = None
        self._mutation_data = None
        
    def _load_metadata(self) -> Dict:
        """Load dataset metadata"""
        metadata_path = self.tcga_data_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning("No metadata file found, using defaults")
            return {
                "modalities": ["transcriptomics", "genomics_mutations"],
                "modality_shapes": {"transcriptomics": 20530, "genomics_mutations": 40543}
            }
            
    def _load_gene_mapping(self, mapping_file: Optional[str]) -> Optional[Dict]:
        """Load gene ID to symbol mapping"""
        if mapping_file and os.path.exists(mapping_file):
            try:
                return pd.read_csv(mapping_file, index_col=0).to_dict()
            except Exception as e:
                logger.warning(f"Failed to load gene mapping: {e}")
        return None
        
    def load_tcga_samples(
        self,
        max_samples: Optional[int] = None,
        cancer_types: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load TCGA samples from NPZ files.
        
        Args:
            max_samples: Maximum number of samples to load
            cancer_types: Specific cancer types to include
            
        Returns:
            Dictionary with sample data
        """
        logger.info(f"Loading TCGA samples from {self.tcga_data_dir}")
        
        sample_files = list(self.tcga_data_dir.glob("sample_*.npz"))
        if max_samples:
            sample_files = sample_files[:max_samples]
            
        samples_data = {
            'sample_ids': [],
            'cancer_types': [],
            'transcriptomics': [],
            'mutations': [],
            'clinical': []
        }
        
        for i, sample_file in enumerate(sample_files):
            if i % 100 == 0:
                logger.info(f"Loading sample {i+1}/{len(sample_files)}")
                
            try:
                data = np.load(sample_file)
                
                # Extract basic info
                sample_id = str(data['sample_id'])
                cancer_type = str(data['cancer_type'])
                
                # Filter by cancer type if specified
                if cancer_types and cancer_type not in cancer_types:
                    continue
                    
                if self.cancer_focus and cancer_type != self.cancer_focus:
                    continue
                    
                # Store data
                samples_data['sample_ids'].append(sample_id)
                samples_data['cancer_types'].append(cancer_type)
                
                # Transcriptomics (gene expression)
                if 'transcriptomics' in data:
                    samples_data['transcriptomics'].append(data['transcriptomics'])
                else:
                    samples_data['transcriptomics'].append(np.zeros(self.metadata['modality_shapes']['transcriptomics']))
                    
                # Mutations
                if 'genomics_mutations' in data:
                    samples_data['mutations'].append(data['genomics_mutations'])
                else:
                    samples_data['mutations'].append(np.zeros(self.metadata['modality_shapes']['genomics_mutations']))
                    
                # Clinical data
                if 'clinical' in data:
                    samples_data['clinical'].append(data['clinical'])
                else:
                    samples_data['clinical'].append(np.zeros(self.metadata['modality_shapes'].get('clinical', 14)))
                    
            except Exception as e:
                logger.warning(f"Failed to load {sample_file}: {e}")
                continue
                
        # Convert to arrays
        for key in ['transcriptomics', 'mutations', 'clinical']:
            if samples_data[key]:
                samples_data[key] = np.array(samples_data[key])
                
        logger.info(f"Loaded {len(samples_data['sample_ids'])} samples")
        return samples_data
        
    def identify_drug_targets_from_tcga(
        self,
        samples_data: Dict[str, np.ndarray],
        top_k: int = 50,
        differential_threshold: float = 2.0,
        mutation_frequency_threshold: float = 0.1
    ) -> List[str]:
        """
        Identify potential drug targets from TCGA data.
        
        Args:
            samples_data: Loaded TCGA samples
            top_k: Number of top targets to return
            differential_threshold: Fold-change threshold for expression
            mutation_frequency_threshold: Mutation frequency threshold
            
        Returns:
            List of target gene identifiers
        """
        logger.info("Identifying drug targets from TCGA data")
        
        target_scores = defaultdict(float)
        
        # Analyze differential expression
        if len(samples_data['transcriptomics']) > 0:
            expr_matrix = samples_data['transcriptomics']
            cancer_types = samples_data['cancer_types']
            
            # Find cancer vs normal patterns
            cancer_samples = [i for i, ct in enumerate(cancer_types) if ct != 'normal']
            normal_samples = [i for i, ct in enumerate(cancer_types) if ct == 'normal']
            
            if len(normal_samples) == 0:
                # If no normal samples, use lowest quartile as baseline
                baseline_expr = np.percentile(expr_matrix, 25, axis=0)
                cancer_expr = np.mean(expr_matrix[cancer_samples], axis=0)
            else:
                baseline_expr = np.mean(expr_matrix[normal_samples], axis=0)
                cancer_expr = np.mean(expr_matrix[cancer_samples], axis=0)
                
            # Calculate fold changes
            fold_changes = cancer_expr / (baseline_expr + 1e-6)
            
            # Score highly upregulated genes
            upregulated_genes = np.where(fold_changes > differential_threshold)[0]
            for gene_idx in upregulated_genes:
                fc = fold_changes[gene_idx]
                target_scores[f"GENE_{gene_idx}"] += min(10.0, np.log2(fc))
                
        # Analyze mutation frequency
        if len(samples_data['mutations']) > 0:
            mut_matrix = samples_data['mutations']
            mut_frequencies = np.mean(mut_matrix > 0, axis=0)
            
            # Score frequently mutated genes
            frequent_mutations = np.where(mut_frequencies > mutation_frequency_threshold)[0]
            for mut_idx in frequent_mutations:
                freq = mut_frequencies[mut_idx]
                target_scores[f"MUT_{mut_idx}"] += freq * 5.0
                
        # Add cancer type specific scoring
        cancer_type_counts = defaultdict(int)
        for ct in samples_data['cancer_types']:
            cancer_type_counts[ct] += 1
            
        # Bonus for genes expressed across multiple cancer types
        if len(cancer_type_counts) > 1:
            logger.info(f"Found {len(cancer_type_counts)} cancer types: {list(cancer_type_counts.keys())}")
            
        # Sort and return top targets
        sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
        top_targets = [target_id for target_id, score in sorted_targets[:top_k]]
        
        logger.info(f"Identified {len(top_targets)} drug targets")
        logger.info(f"Top 10 targets: {top_targets[:10]}")
        
        return top_targets
        
    def convert_tcga_to_antigens(
        self,
        target_gene_ids: List[str],
        samples_data: Dict[str, np.ndarray],
        include_mutations: bool = True
    ) -> List[DrugTargetAntigen]:
        """
        Convert TCGA target genes to DrugTargetAntigen objects.
        
        Args:
            target_gene_ids: List of target gene identifiers
            samples_data: Loaded TCGA samples  
            include_mutations: Include mutation data in antigens
            
        Returns:
            List of DrugTargetAntigen objects
        """
        logger.info(f"Converting {len(target_gene_ids)} targets to antigens")
        
        antigens = []
        converter = OmicsToAntigenConverter()
        
        for target_id in target_gene_ids:
            try:
                # Extract gene-specific data
                if target_id.startswith("GENE_"):
                    gene_idx = int(target_id.split("_")[1])
                    antigen = self._create_antigen_from_expression(
                        gene_idx, target_id, samples_data, converter
                    )
                elif target_id.startswith("MUT_"):
                    mut_idx = int(target_id.split("_")[1])
                    antigen = self._create_antigen_from_mutation(
                        mut_idx, target_id, samples_data, converter
                    )
                else:
                    # Direct gene name
                    antigen = self._create_antigen_from_gene_name(
                        target_id, samples_data, converter
                    )
                    
                if antigen:
                    antigens.append(antigen)
                    
            except Exception as e:
                logger.warning(f"Failed to create antigen for {target_id}: {e}")
                continue
                
        logger.info(f"Successfully created {len(antigens)} antigens")
        return antigens
        
    def _create_antigen_from_expression(
        self,
        gene_idx: int,
        target_id: str,
        samples_data: Dict[str, np.ndarray],
        converter: OmicsToAntigenConverter
    ) -> Optional[DrugTargetAntigen]:
        """Create antigen from expression data"""
        
        # Extract expression profile
        expr_data = samples_data['transcriptomics'][:, gene_idx]
        cancer_types = samples_data['cancer_types']
        
        # Calculate expression by cancer type
        expression_by_type = {}
        for ct in set(cancer_types):
            ct_samples = [i for i, c in enumerate(cancer_types) if c == ct]
            if ct_samples:
                expression_by_type[ct] = float(np.mean(expr_data[ct_samples]))
                
        # Generate protein sequence (mock for now)
        protein_id = f"TCGA_TARGET_{target_id}"
        sequence = converter._get_protein_sequence(protein_id, None)
        
        if not sequence:
            return None
            
        # Create structure
        structure = ProteinStructure(
            sequence=sequence,
            coordinates=converter._generate_mock_coordinates(sequence),
            secondary_structure='C' * len(sequence),
            pdb_id=protein_id
        )
        
        # Predict pockets
        pockets = converter._predict_binding_pockets(structure)
        
        # Get mutation data if available
        mutation_data = []
        if len(samples_data['mutations']) > 0 and gene_idx < samples_data['mutations'].shape[1]:
            # Find samples with mutations in this gene region
            gene_mut_mask = samples_data['mutations'][:, gene_idx] > 0
            if np.any(gene_mut_mask):
                # Add some mock mutations
                mutation_data = [
                    (np.random.randint(50, len(sequence)-50), 'A', 'V'),
                    (np.random.randint(50, len(sequence)-50), 'R', 'K')
                ]
                
        # Determine disease association
        cancer_type_list = list(set(cancer_types))
        disease_association = cancer_type_list[0] if cancer_type_list else "cancer"
        
        return DrugTargetAntigen(
            protein_structure=structure,
            binding_pockets=pockets,
            disease_association=disease_association,
            expression_data=expression_by_type,
            mutation_data=mutation_data
        )
        
    def _create_antigen_from_mutation(
        self,
        mut_idx: int,
        target_id: str,
        samples_data: Dict[str, np.ndarray],
        converter: OmicsToAntigenConverter
    ) -> Optional[DrugTargetAntigen]:
        """Create antigen from mutation data"""
        
        # Extract mutation profile
        mut_data = samples_data['mutations'][:, mut_idx]
        mutation_frequency = np.mean(mut_data > 0)
        
        # Generate protein
        protein_id = f"TCGA_MUT_{target_id}"
        sequence = converter._get_protein_sequence(protein_id, None)
        
        if not sequence:
            return None
            
        structure = ProteinStructure(
            sequence=sequence,
            coordinates=converter._generate_mock_coordinates(sequence),
            secondary_structure='C' * len(sequence),
            pdb_id=protein_id
        )
        
        pockets = converter._predict_binding_pockets(structure)
        
        # Add mutation information
        mutation_position = int(len(sequence) * mutation_frequency)  # Position based on frequency
        mutation_data = [(mutation_position, 'W', 'L')]  # Wild-type to mutant
        
        return DrugTargetAntigen(
            protein_structure=structure,
            binding_pockets=pockets,
            disease_association="cancer_mutation",
            mutation_data=mutation_data,
            epitopes=[AntigenEpitope(sequence=structure.sequence, structure_coords=structure.coordinates, hydrophobicity=0, charge=0)]
        )
        
    def _create_antigen_from_gene_name(
        self,
        gene_name: str,
        samples_data: Dict[str, np.ndarray],
        converter: OmicsToAntigenConverter
    ) -> Optional[DrugTargetAntigen]:
        """Create antigen from gene name"""
        
        # Use standard converter approach
        sequence = converter._get_protein_sequence(gene_name, None)
        if not sequence:
            return None
            
        structure = ProteinStructure(
            sequence=sequence,
            coordinates=converter._generate_mock_coordinates(sequence),
            secondary_structure='C' * len(sequence),
            pdb_id=gene_name
        )
        
        pockets = converter._predict_binding_pockets(structure)
        
        return DrugTargetAntigen(
            protein_structure=structure,
            binding_pockets=pockets,
            disease_association="tcga_derived"
        )
        
    def create_comprehensive_drug_target_report(
        self,
        samples_data: Dict[str, np.ndarray],
        output_file: str = "tcga_drug_targets_report.json"
    ) -> Dict:
        """Create comprehensive report of drug targets from TCGA data"""
        
        report = {
            "dataset_summary": {
                "total_samples": len(samples_data['sample_ids']),
                "cancer_types": list(set(samples_data['cancer_types'])),
                "transcriptomics_genes": samples_data['transcriptomics'].shape[1] if len(samples_data['transcriptomics']) > 0 else 0,
                "mutation_features": samples_data['mutations'].shape[1] if len(samples_data['mutations']) > 0 else 0
            },
            "target_identification": {},
            "expression_analysis": {},
            "mutation_analysis": {}
        }
        
        # Cancer type distribution
        cancer_counts = defaultdict(int)
        for ct in samples_data['cancer_types']:
            cancer_counts[ct] += 1
        report["dataset_summary"]["cancer_type_distribution"] = dict(cancer_counts)
        
        # Expression analysis
        if len(samples_data['transcriptomics']) > 0:
            expr_matrix = samples_data['transcriptomics']
            
            # High variance genes (potential targets)
            gene_variances = np.var(expr_matrix, axis=0)
            high_var_genes = np.argsort(gene_variances)[-100:]  # Top 100
            
            report["expression_analysis"] = {
                "high_variance_genes": high_var_genes.tolist(),
                "mean_expression_per_gene": np.mean(expr_matrix, axis=0).tolist()[:100],  # First 100 for brevity
                "expression_ranges": {
                    "min": float(np.min(expr_matrix)),
                    "max": float(np.max(expr_matrix)),
                    "mean": float(np.mean(expr_matrix))
                }
            }
            
        # Mutation analysis
        if len(samples_data['mutations']) > 0:
            mut_matrix = samples_data['mutations']
            mut_frequencies = np.mean(mut_matrix > 0, axis=0)
            
            # Most frequently mutated
            freq_mut_indices = np.argsort(mut_frequencies)[-50:]  # Top 50
            
            report["mutation_analysis"] = {
                "frequent_mutations": freq_mut_indices.tolist(),
                "mutation_frequencies": mut_frequencies[freq_mut_indices].tolist(),
                "overall_mutation_rate": float(np.mean(mut_frequencies))
            }
            
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Comprehensive report saved to {output_file}")
        return report


def run_tcga_drug_discovery_pipeline(
    tcga_data_dir: str,
    max_samples: int = 1000,
    top_targets: int = 20,
    cancer_focus: Optional[str] = None
):
    """
    Complete pipeline for TCGA drug discovery.
    
    Args:
        tcga_data_dir: Path to TCGA NPZ files
        max_samples: Maximum samples to load
        top_targets: Number of targets to evaluate
        cancer_focus: Focus on specific cancer type
    """
    logger.info("="*80)
    logger.info("TCGA DRUG DISCOVERY PIPELINE")
    logger.info("="*80)
    
    # Initialize converter
    converter = TCGADataConverter(
        tcga_data_dir=tcga_data_dir,
        cancer_focus=cancer_focus
    )
    
    # Load TCGA data
    logger.info("Step 1: Loading TCGA samples...")
    samples_data = converter.load_tcga_samples(max_samples=max_samples)
    
    # Generate comprehensive report
    logger.info("Step 2: Analyzing TCGA data...")
    report = converter.create_comprehensive_drug_target_report(samples_data)
    
    # Identify drug targets
    logger.info("Step 3: Identifying drug targets...")
    target_ids = converter.identify_drug_targets_from_tcga(
        samples_data,
        top_k=top_targets
    )
    
    # Convert to antigens
    logger.info("Step 4: Converting to antigens...")
    antigens = converter.convert_tcga_to_antigens(target_ids, samples_data)
    
    logger.info(f"\nPipeline completed successfully!")
    logger.info(f"Samples loaded: {len(samples_data['sample_ids'])}")
    logger.info(f"Targets identified: {len(target_ids)}")
    logger.info(f"Antigens created: {len(antigens)}")
    
    return {
        'samples_data': samples_data,
        'target_ids': target_ids,
        'antigens': antigens,
        'report': report
    }