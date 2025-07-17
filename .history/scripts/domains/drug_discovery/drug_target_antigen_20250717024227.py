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
        target_id: str,
        sequence: str,
        binding_sites: Optional[List[BindingPocket]] = None,
        known_drugs: Optional[List[Dict]] = None,
        disease_association: Optional[str] = None,
        druggability_score: Optional[float] = None,
        molecular_weight: Optional[float] = None,
        logp: Optional[float] = None,
        num_h_donors: Optional[int] = None,
        num_h_acceptors: Optional[int] = None,
        tpsa: Optional[float] = None,
        num_rotatable_bonds: Optional[int] = None,
        epitopes: Optional[List[AntigenEpitope]] = None,
        expression_data: Optional[Dict[str, float]] = None,
        mutation_data: Optional[List[Tuple[int, str, str]]] = None,
    ):
        # Store all attributes
        self.target_id = target_id
        self.sequence = sequence
        self.binding_sites = binding_sites or []
        self.known_drugs = known_drugs or []
        self.disease_association = disease_association
        self.druggability_score = druggability_score
        self.molecular_weight = molecular_weight
        self.logp = logp
        self.num_h_donors = num_h_donors
        self.num_h_acceptors = num_h_acceptors
        self.tpsa = tpsa
        self.num_rotatable_bonds = num_rotatable_bonds
        self.epitopes = epitopes or []
        self.expression_data = expression_data or {}
        self.mutation_data = mutation_data or []

        # Call super with appropriate antigen_type
        antigen_type = f"drug_target_{disease_association or 'unknown'}"
        super().__init__(antigen_type=antigen_type)
        
        # Restore selectivity_score computation
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
        if not self.binding_sites:
            return 0.0
            
        # Weighted average of pocket druggability scores
        scores = [p.druggability_score for p in self.binding_sites]
        volumes = [p.volume for p in self.binding_sites]
        
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
            pocket.to_tensor() for pocket in self.binding_sites
        ]) if self.binding_sites else torch.zeros(1, 4)
        
        # Add to graph data
        graph_data.pocket_features = pocket_features
        graph_data.druggability = torch.tensor([self.druggability_score])
        graph_data.selectivity = torch.tensor([self.selectivity_score])
        
        graph_data.y = torch.tensor([self.druggability_score], dtype=torch.float)
        
        # Add expression data as node features if available
        if self.expression_data:
            expr_vector = torch.tensor(list(self.expression_data.values()))
            graph_data.expression = expr_vector
            
        return graph_data
    
    def apply_disease_mutations(self) -> 'DrugTargetAntigen':
        """Apply known disease-associated mutations"""
        mutated = super().copy()
        
        for position, wild_type, mutant in self.mutation_data:
            if position < len(self.sequence):
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
            'protein_id': self.target_id,
            'sequence_length': len(self.sequence),
            'num_pockets': len(self.binding_sites),
            'best_pocket': max(self.binding_sites, key=lambda p: p.druggability_score).pocket_id
            if self.binding_sites else None,
            'global_druggability': self.druggability_score,
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
                for p in self.binding_sites
            ]
        }