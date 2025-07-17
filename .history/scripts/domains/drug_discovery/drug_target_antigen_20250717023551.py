"""
Drug Target Antigen Implementation
==================================

Extends the BiologicalAntigen class to represent drug targets with
protein-specific features like binding pockets, druggability scores,
and structural information.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
from torch_geometric.data import Data

from scripts.core.antigen import BiologicalAntigen, AntigenEpitope


@dataclass
class BindingPocket:
    """Represents a potential drug binding pocket"""
    pocket_id: str
    residue_indices: List[int]
    volume: float  # Angstrom^3
    hydrophobicity: float
    electrostatic_potential: float
    druggability_score: float
    known_ligands: List[str] = field(default_factory=list)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert pocket features to tensor"""
        return torch.tensor([
            self.volume / 1000.0,  # Normalize to reasonable range
            self.hydrophobicity,
            self.electrostatic_potential,
            self.druggability_score
        ])


@dataclass
class ProteinStructure:
    """Protein structural information"""
    sequence: str
    coordinates: np.ndarray  # (N_atoms, 3)
    secondary_structure: str  # DSSP notation
    b_factors: Optional[np.ndarray] = None
    pdb_id: Optional[str] = None
    resolution: Optional[float] = None


class DrugTargetAntigen(BiologicalAntigen):
    """
    Specialized antigen for drug target discovery.
    Adds protein-specific features and drug discovery metrics.
    """
    
    def __init__(
        self,
                target_id: str,  # <-- Add this line

        protein_structure: ProteinStructure,
        binding_pockets: List[BindingPocket],
        epitopes: Optional[List[AntigenEpitope]] = None,
        known_drugs: Optional[List[Dict]] = None,
        disease_association: Optional[str] = None,
        expression_data: Optional[Dict[str, float]] = None,
        mutation_data: Optional[List[Tuple[int, str, str]]] = None
    ):
        """
        Initialize drug target antigen.
        
        Args:
            protein_structure: Protein structural data
            binding_pockets: Identified binding pockets
            epitopes: Antigenic epitopes (auto-generated if None)
            known_drugs: List of known drugs targeting this protein
            disease_association: Associated disease
            expression_data: Gene expression levels by tissue
            mutation_data: Known disease-associated mutations
        """
        # Generate epitopes from binding pockets if not provided
        if epitopes is None:
            epitopes = self._generate_epitopes_from_pockets(
                protein_structure, binding_pockets
            )
        
        # Initialize base antigen
        antigen_type = f"drug_target_{disease_association or 'unknown'}"
        super().__init__(antigen_type=antigen_type)
        self.epitopes = epitopes
        
        self.protein_structure = protein_structure
        self.binding_pockets = binding_pockets
        self.known_drugs = known_drugs or []
        self.disease_association = disease_association
        self.expression_data = expression_data or {}
        self.mutation_data = mutation_data or []
        
        # Compute druggability features
        self.global_druggability = self._compute_global_druggability()
        self.selectivity_score = self._compute_selectivity_potential()
        
    def _generate_epitopes_from_pockets(
        self,
        structure: ProteinStructure,
        pockets: List[BindingPocket]
    ) -> List[AntigenEpitope]:
        """Generate epitopes from binding pockets"""
        epitopes = []
        
        for i, pocket in enumerate(pockets):
            # Extract pocket residues
            pocket_coords = structure.coordinates[pocket.residue_indices]
            center = pocket_coords.mean(axis=0)
            
            # Create epitope from pocket
            epitope = AntigenEpitope(
                epitope_id=f"pocket_{pocket.pocket_id}",
                position=i * 0.2,  # Distribute across gene positions
                amino_acid_sequence=self._extract_pocket_sequence(
                    structure.sequence, pocket.residue_indices
                ),
                structure_coords=pocket_coords,
                binding_affinity=pocket.druggability_score,
                hydrophobicity=pocket.hydrophobicity,
                charge_distribution=np.array([pocket.electrostatic_potential]),
                glycosylation_sites=[],
                disulfide_bonds=[]
            )
            epitopes.append(epitope)
            
        return epitopes
    
    def _extract_pocket_sequence(
        self,
        sequence: str,
        residue_indices: List[int]
    ) -> str:
        """Extract sequence for pocket residues"""
        # Assuming residue_indices are 0-based
        pocket_seq = ''.join([
            sequence[i] if i < len(sequence) else 'X'
            for i in sorted(residue_indices)[:20]  # Limit to 20 residues
        ])
        return pocket_seq
    
    def _generate_conformational_states(
        self,
        structure: ProteinStructure
    ) -> List[Dict]:
        """Generate conformational states from structure"""
        states = []
        
        # Active state (original structure)
        states.append({
            'state_id': 'active',
            'probability': 0.7,
            'coordinate_transform': np.eye(3),
            'energy': 0.0
        })
        
        # Inactive state (slightly perturbed)
        rotation = self._random_rotation_matrix(angle=0.1)
        states.append({
            'state_id': 'inactive',
            'probability': 0.3,
            'coordinate_transform': rotation,
            'energy': 5.0  # kcal/mol
        })
        
        return states
    
    def _random_rotation_matrix(self, angle: float) -> np.ndarray:
        """Generate small random rotation matrix"""
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        return R
    
    def _compute_global_druggability(self) -> float:
        """Compute overall druggability score"""
        if not self.binding_pockets:
            return 0.0
            
        # Weighted average of pocket druggability scores
        scores = [p.druggability_score for p in self.binding_pockets]
        volumes = [p.volume for p in self.binding_pockets]
        
        weighted_score = sum(s * v for s, v in zip(scores, volumes)) / sum(volumes)
        
        # Bonus for having known drugs
        if self.known_drugs:
            weighted_score *= 1.2
            
        return min(1.0, weighted_score)
    
    def _compute_selectivity_potential(self) -> float:
        """Estimate potential for selective targeting"""
        if not self.expression_data:
            return 0.5  # Default medium selectivity
            
        # Calculate tissue specificity
        expression_values = list(self.expression_data.values())
        if not expression_values:
            return 0.5
            
        mean_expr = np.mean(expression_values)
        std_expr = np.std(expression_values)
        
        # High std relative to mean indicates tissue-specific expression
        if mean_expr > 0:
            specificity = std_expr / mean_expr
            return min(1.0, specificity)
        return 0.5
    
    def to_graph(self) -> Data:
        """Convert to graph representation for GNN processing"""
        # Get base graph
        graph_data = super().to_graph()
        
        # Add drug discovery specific features
        pocket_features = torch.stack([
            pocket.to_tensor() for pocket in self.binding_pockets
        ]) if self.binding_pockets else torch.zeros(1, 4)
        
        # Add to graph data
        graph_data.pocket_features = pocket_features
        graph_data.druggability = torch.tensor([self.global_druggability])
        graph_data.selectivity = torch.tensor([self.selectivity_score])
        
        graph_data.y = torch.tensor([self.global_druggability], dtype=torch.float)
        
        # Add expression data as node features if available
        if self.expression_data:
            expr_vector = torch.tensor(list(self.expression_data.values()))
            graph_data.expression = expr_vector
            
        return graph_data
    
    def apply_disease_mutations(self) -> 'DrugTargetAntigen':
        """Apply known disease-associated mutations"""
        mutated = super().copy()
        
        for position, wild_type, mutant in self.mutation_data:
            if position < len(self.protein_structure.sequence):
                # Apply mutation to relevant epitopes
                for epitope in mutated.epitopes:
                    if position in epitope.structure_coords:
                        # Simplified mutation effect
                        epitope.binding_affinity *= 0.8
                        epitope.hydrophobicity += np.random.randn() * 0.1
                        
        return mutated
    
    def get_druggability_report(self) -> Dict:
        """Generate comprehensive druggability report"""
        return {
            'protein_id': self.protein_structure.pdb_id or 'unknown',
            'sequence_length': len(self.protein_structure.sequence),
            'num_pockets': len(self.binding_pockets),
            'best_pocket': max(self.binding_pockets, key=lambda p: p.druggability_score).pocket_id
            if self.binding_pockets else None,
            'global_druggability': self.global_druggability,
            'selectivity_score': self.selectivity_score,
            'has_known_drugs': len(self.known_drugs) > 0,
            'disease_mutations': len(self.mutation_data),
            'tissue_expression': self.expression_data,
            'pocket_details': [
                {
                    'id': p.pocket_id,
                    'volume': p.volume,
                    'druggability': p.druggability_score,
                    'known_ligands': p.known_ligands
                }
                for p in self.binding_pockets
            ]
        }