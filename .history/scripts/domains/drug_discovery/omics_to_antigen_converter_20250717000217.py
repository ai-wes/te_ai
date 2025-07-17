"""
Omics to Antigen Converter
==========================

Converts various omics data formats (gene expression, proteomics, 
structural data) into DrugTargetAntigen objects that can be processed
by the TE-AI system.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import json
from dataclasses import dataclass
import hashlib

from scripts.core.antigen import AntigenEpitope
from .drug_target_antigen import DrugTargetAntigen, BindingPocket, ProteinStructure


@dataclass
class OmicsData:
    """Container for various omics data types"""
    gene_expression: Optional[pd.DataFrame] = None  # Samples x Genes
    protein_abundance: Optional[pd.DataFrame] = None  # Samples x Proteins
    mutations: Optional[pd.DataFrame] = None  # Gene, Position, WT, Mutant, Frequency
    structural_data: Optional[Dict[str, Dict]] = None  # Protein -> Structure info
    pathway_data: Optional[Dict[str, List[str]]] = None  # Pathway -> Genes
    disease_associations: Optional[Dict[str, float]] = None  # Disease -> Score


class OmicsToAntigenConverter:
    """
    Converts omics data into DrugTargetAntigen objects for TE-AI processing.
    """
    
    def __init__(
        self,
        structure_predictor: Optional[str] = None,  # 'alphafold', 'rosetta', etc.
        pocket_predictor: str = 'cavityplus',  # Tool for pocket prediction
        epitope_length: int = 15,
        min_pocket_volume: float = 100.0  # Cubic angstroms
    ):
        self.structure_predictor = structure_predictor
        self.pocket_predictor = pocket_predictor
        self.epitope_length = epitope_length
        self.min_pocket_volume = min_pocket_volume
        
        # Amino acid properties for feature calculation
        self.aa_properties = self._load_aa_properties()
        
    def _load_aa_properties(self) -> Dict[str, Dict[str, float]]:
        """Load amino acid physicochemical properties"""
        return {
            'A': {'hydrophobicity': 1.8, 'charge': 0, 'size': 88.6},
            'R': {'hydrophobicity': -4.5, 'charge': 1, 'size': 173.4},
            'N': {'hydrophobicity': -3.5, 'charge': 0, 'size': 114.1},
            'D': {'hydrophobicity': -3.5, 'charge': -1, 'size': 111.1},
            'C': {'hydrophobicity': 2.5, 'charge': 0, 'size': 108.5},
            'Q': {'hydrophobicity': -3.5, 'charge': 0, 'size': 143.8},
            'E': {'hydrophobicity': -3.5, 'charge': -1, 'size': 138.4},
            'G': {'hydrophobicity': -0.4, 'charge': 0, 'size': 60.1},
            'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'size': 153.2},
            'I': {'hydrophobicity': 4.5, 'charge': 0, 'size': 166.7},
            'L': {'hydrophobicity': 3.8, 'charge': 0, 'size': 166.7},
            'K': {'hydrophobicity': -3.9, 'charge': 1, 'size': 168.6},
            'M': {'hydrophobicity': 1.9, 'charge': 0, 'size': 162.9},
            'F': {'hydrophobicity': 2.8, 'charge': 0, 'size': 189.9},
            'P': {'hydrophobicity': -1.6, 'charge': 0, 'size': 112.7},
            'S': {'hydrophobicity': -0.8, 'charge': 0, 'size': 89.0},
            'T': {'hydrophobicity': -0.7, 'charge': 0, 'size': 116.1},
            'W': {'hydrophobicity': -0.9, 'charge': 0, 'size': 227.8},
            'Y': {'hydrophobicity': -1.3, 'charge': 0, 'size': 193.6},
            'V': {'hydrophobicity': 4.2, 'charge': 0, 'size': 140.0}
        }
    
    def convert_omics_to_antigens(
        self,
        omics_data: OmicsData,
        target_proteins: Optional[List[str]] = None,
        disease_focus: Optional[str] = None
    ) -> List[DrugTargetAntigen]:
        """
        Convert omics data to drug target antigens.
        
        Args:
            omics_data: Multi-omics data container
            target_proteins: Specific proteins to focus on (None = all)
            disease_focus: Disease context for prioritization
            
        Returns:
            List of DrugTargetAntigen objects
        """
        antigens = []
        
        # Identify target proteins
        if target_proteins is None:
            target_proteins = self._identify_target_proteins(omics_data, disease_focus)
            
        for protein_id in target_proteins:
            try:
                antigen = self._create_antigen_from_protein(
                    protein_id,
                    omics_data,
                    disease_focus
                )
                if antigen:
                    antigens.append(antigen)
            except Exception as e:
                print(f"Failed to create antigen for {protein_id}: {e}")
                continue
                
        return antigens
    
    def _identify_target_proteins(
        self,
        omics_data: OmicsData,
        disease_focus: Optional[str] = None
    ) -> List[str]:
        """Identify potential target proteins from omics data"""
        candidates = set()
        
        # From gene expression - highly variable or disease-associated genes
        if omics_data.gene_expression is not None:
            expr_df = omics_data.gene_expression
            
            # Calculate variance across samples
            variances = expr_df.var()
            top_variable = variances.nlargest(100).index.tolist()
            candidates.update(top_variable)
            
            # Disease-specific upregulation
            if disease_focus and 'condition' in expr_df.columns:
                disease_samples = expr_df[expr_df['condition'] == disease_focus]
                control_samples = expr_df[expr_df['condition'] == 'control']
                
                if len(disease_samples) > 0 and len(control_samples) > 0:
                    fold_changes = disease_samples.mean() / (control_samples.mean() + 1e-6)
                    upregulated = fold_changes[fold_changes > 2.0].index.tolist()
                    candidates.update(upregulated[:50])
        
        # From protein abundance
        if omics_data.protein_abundance is not None:
            # Similar analysis for proteins
            high_abundance = omics_data.protein_abundance.mean().nlargest(50).index.tolist()
            candidates.update(high_abundance)
            
        # From mutation data - frequently mutated genes
        if omics_data.mutations is not None:
            mutation_counts = omics_data.mutations['Gene'].value_counts()
            frequently_mutated = mutation_counts.head(30).index.tolist()
            candidates.update(frequently_mutated)
            
        # From disease associations
        if omics_data.disease_associations and disease_focus:
            disease_genes = [
                gene for gene, score in omics_data.disease_associations.items()
                if score > 0.5
            ]
            candidates.update(disease_genes[:40])
            
        return list(candidates)[:100]  # Limit to top 100 candidates
    
    def _create_antigen_from_protein(
        self,
        protein_id: str,
        omics_data: OmicsData,
        disease_focus: Optional[str] = None
    ) -> Optional[DrugTargetAntigen]:
        """Create a DrugTargetAntigen from protein data"""
        
        # Get or predict structure
        structure = self._get_protein_structure(protein_id, omics_data)
        if not structure:
            return None
            
        # Predict binding pockets
        pockets = self._predict_binding_pockets(structure)
        if not pockets:
            # Create a default pocket if none found
            pockets = [self._create_default_pocket(structure)]
            
        # Extract expression data
        expression_data = self._extract_expression_data(protein_id, omics_data)
        
        # Extract mutation data
        mutation_data = self._extract_mutation_data(protein_id, omics_data)
        
        # Create antigen
        antigen = DrugTargetAntigen(
            protein_structure=structure,
            binding_pockets=pockets,
            disease_association=disease_focus,
            expression_data=expression_data,
            mutation_data=mutation_data
        )
        
        return antigen
    
    def _get_protein_structure(
        self,
        protein_id: str,
        omics_data: OmicsData
    ) -> Optional[ProteinStructure]:
        """Get or predict protein structure"""
        
        # Check if structure data is provided
        if omics_data.structural_data and protein_id in omics_data.structural_data:
            struct_info = omics_data.structural_data[protein_id]
            return ProteinStructure(
                sequence=struct_info.get('sequence', 'UNKNOWN'),
                coordinates=np.array(struct_info.get('coordinates', [])),
                secondary_structure=struct_info.get('secondary_structure', ''),
                pdb_id=struct_info.get('pdb_id'),
                resolution=struct_info.get('resolution')
            )
            
        # Otherwise, create a mock structure
        # In a real implementation, this would call AlphaFold or similar
        sequence = self._get_protein_sequence(protein_id, omics_data)
        if not sequence:
            return None
            
        # Generate mock coordinates (in real implementation, use structure prediction)
        coords = self._generate_mock_coordinates(sequence)
        
        return ProteinStructure(
            sequence=sequence,
            coordinates=coords,
            secondary_structure='C' * len(sequence),  # Mock: all coil
            pdb_id=None,
            resolution=None
        )
    
    def _get_protein_sequence(
        self,
        protein_id: str,
        omics_data: Optional[OmicsData] = None
    ) -> Optional[str]:
        """Get protein sequence from omics data or database"""
        # In real implementation, query UniProt or similar
        # For now, generate a mock sequence
        
        # Use protein_id as seed for reproducibility
        np.random.seed(int(hashlib.md5(protein_id.encode()).hexdigest()[:8], 16))
        
        # Generate random sequence
        amino_acids = list(self.aa_properties.keys())
        length = np.random.randint(200, 600)
        sequence = ''.join(np.random.choice(amino_acids, length))
        
        return sequence
    
    def _generate_mock_coordinates(self, sequence: str) -> np.ndarray:
        """Generate mock 3D coordinates for sequence"""
        n_atoms = len(sequence) * 5  # Assume 5 atoms per residue on average
        
        # Generate a simple helix-like structure
        coords = np.zeros((n_atoms, 3))
        atoms_per_residue = 5
        
        for i, aa in enumerate(sequence):
            # Place residue atoms in a helical pattern
            angle = i * 0.1
            radius = 10.0
            
            for j in range(atoms_per_residue):
                atom_idx = i * atoms_per_residue + j
                coords[atom_idx, 0] = radius * np.cos(angle + j * 0.2)
                coords[atom_idx, 1] = radius * np.sin(angle + j * 0.2)
                coords[atom_idx, 2] = i * 3.4  # Helical rise
                
        return coords
    
    def _predict_binding_pockets(
        self,
        structure: ProteinStructure
    ) -> List[BindingPocket]:
        """Predict binding pockets from structure"""
        pockets = []
        
        # In real implementation, use CAVITYplus, P2Rank, or similar
        # For now, create mock pockets based on sequence properties
        
        sequence = structure.sequence
        pocket_regions = self._find_hydrophobic_clusters(sequence)
        
        for i, (start, end) in enumerate(pocket_regions[:5]):  # Max 5 pockets
            # Calculate pocket properties
            pocket_seq = sequence[start:end]
            hydrophobicity = np.mean([
                self.aa_properties.get(aa, {}).get('hydrophobicity', 0)
                for aa in pocket_seq
            ])
            
            charge = sum([
                self.aa_properties.get(aa, {}).get('charge', 0)
                for aa in pocket_seq
            ])
            
            # Estimate volume based on residue count
            volume = len(pocket_seq) * 150.0  # ~150 cubic angstroms per residue
            
            # Simple druggability scoring
            druggability = self._calculate_druggability_score(
                volume, hydrophobicity, charge
            )
            
            pocket = BindingPocket(
                pocket_id=f"pocket_{i}",
                residue_indices=list(range(start, end)),
                volume=volume,
                hydrophobicity=hydrophobicity,
                electrostatic_potential=charge,
                druggability_score=druggability
            )
            
            pockets.append(pocket)
            
        return pockets
    
    def _find_hydrophobic_clusters(self, sequence: str) -> List[Tuple[int, int]]:
        """Find hydrophobic clusters that might form pockets"""
        hydrophobic_aa = set(['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'])
        clusters = []
        
        in_cluster = False
        start = 0
        
        for i, aa in enumerate(sequence):
            if aa in hydrophobic_aa:
                if not in_cluster:
                    start = i
                    in_cluster = True
            else:
                if in_cluster and i - start >= 5:  # Min 5 residues
                    clusters.append((start, i))
                in_cluster = False
                
        # Handle last cluster
        if in_cluster and len(sequence) - start >= 5:
            clusters.append((start, len(sequence)))
            
        return clusters
    
    def _calculate_druggability_score(
        self,
        volume: float,
        hydrophobicity: float,
        charge: float
    ) -> float:
        """Calculate druggability score based on pocket properties"""
        # Ideal pocket: 300-500 cubic angstroms, moderate hydrophobicity
        volume_score = 1.0 - abs(volume - 400) / 400
        volume_score = max(0, min(1, volume_score))
        
        # Moderate hydrophobicity is best
        hydro_score = 1.0 - abs(hydrophobicity - 1.0) / 3.0
        hydro_score = max(0, min(1, hydro_score))
        
        # Neutral charge is preferred
        charge_score = 1.0 - abs(charge) / 5.0
        charge_score = max(0, min(1, charge_score))
        
        # Weighted combination
        druggability = (
            volume_score * 0.4 +
            hydro_score * 0.4 +
            charge_score * 0.2
        )
        
        return druggability
    
    def _create_default_pocket(self, structure: ProteinStructure) -> BindingPocket:
        """Create a default pocket when none are found"""
        # Use center region of protein
        seq_len = len(structure.sequence)
        start = seq_len // 3
        end = 2 * seq_len // 3
        
        return BindingPocket(
            pocket_id="default_pocket",
            residue_indices=list(range(start, end)),
            volume=300.0,
            hydrophobicity=0.0,
            electrostatic_potential=0.0,
            druggability_score=0.5
        )
    
    def _extract_expression_data(
        self,
        protein_id: str,
        omics_data: OmicsData
    ) -> Dict[str, float]:
        """Extract expression data for protein across tissues/conditions"""
        expression = {}
        
        if omics_data.gene_expression is not None and protein_id in omics_data.gene_expression.columns:
            # Get expression by sample group
            expr_series = omics_data.gene_expression[protein_id]
            
            # Group by tissue/condition if metadata available
            if 'tissue' in omics_data.gene_expression.index:
                for tissue in omics_data.gene_expression.index.unique():
                    tissue_expr = expr_series[omics_data.gene_expression.index == tissue]
                    expression[tissue] = tissue_expr.mean()
            else:
                expression['mean'] = expr_series.mean()
                expression['std'] = expr_series.std()
                
        return expression
    
    def _extract_mutation_data(
        self,
        protein_id: str,
        omics_data: OmicsData
    ) -> List[Tuple[int, str, str]]:
        """Extract mutation data for protein"""
        mutations = []
        
        if omics_data.mutations is not None:
            protein_mutations = omics_data.mutations[
                omics_data.mutations['Gene'] == protein_id
            ]
            
            for _, mut in protein_mutations.iterrows():
                mutations.append((
                    int(mut.get('Position', 0)),
                    mut.get('WT', 'X'),
                    mut.get('Mutant', 'X')
                ))
                
        return mutations
    
    def create_antigen_from_csv(
        self,
        csv_path: str,
        protein_column: str = 'protein_id',
        expression_columns: Optional[List[str]] = None,
        mutation_info: Optional[Dict] = None
    ) -> List[DrugTargetAntigen]:
        """
        Convenience method to create antigens from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            protein_column: Column containing protein IDs
            expression_columns: Columns with expression data
            mutation_info: Additional mutation information
            
        Returns:
            List of DrugTargetAntigen objects
        """
        df = pd.read_csv(csv_path)
        
        # Create OmicsData object
        omics_data = OmicsData()
        
        # Extract gene expression if columns specified
        if expression_columns:
            omics_data.gene_expression = df[expression_columns]
            
        # Extract protein list
        if protein_column in df.columns:
            target_proteins = df[protein_column].unique().tolist()
        else:
            target_proteins = None
            
        # Add mutation info if provided
        if mutation_info:
            omics_data.mutations = pd.DataFrame(mutation_info)
            
        return self.convert_omics_to_antigens(omics_data, target_proteins)