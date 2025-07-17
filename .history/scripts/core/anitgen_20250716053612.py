from typing import List, Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
import random
from scripts.config import cfg




# ============================================================================
# Biologically Accurate Antigen Modeling
# ============================================================================

class AntigenEpitope:
    """Biologically accurate epitope representation"""
    def __init__(self, sequence: str, structure_coords: np.ndarray, 
                 hydrophobicity: float, charge: float):
        self.sequence = sequence
        self.structure_coords = structure_coords
        self.hydrophobicity = hydrophobicity
        self.charge = charge
        self.mutations = []
        
    def mutate(self, position: int, new_residue: str):
        """Apply point mutation"""
        old_residue = self.sequence[position]
        self.sequence = self.sequence[:position] + new_residue + self.sequence[position+1:]
        self.mutations.append((position, old_residue, new_residue))
        
        # Update biophysical properties
        self._update_properties()
    
    def _update_properties(self):
        """Recalculate properties after mutation"""
        # Hydrophobicity scale (Kyte-Doolittle)
        hydro_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        self.hydrophobicity = np.mean([
            hydro_scale.get(aa, 0.0) for aa in self.sequence
        ])

class BiologicalAntigen:
    """Complete antigen with multiple epitopes and realistic properties"""
    
    def __init__(self, antigen_type: str = "viral_spike"):
        self.antigen_type = antigen_type
        self.epitopes = self._generate_epitopes()
        self.glycosylation_sites = self._identify_glycosylation()
        self.conformational_states = self._generate_conformations()
        self.current_conformation = 0
        
    def _generate_epitopes(self) -> List[AntigenEpitope]:
        """Generate biologically realistic epitopes"""
        epitopes = []
        
        if self.antigen_type == "viral_spike":
            # RBD epitopes (based on SARS-CoV-2 spike)
            rbd_sequences = [
                "RVQPTESIVRFPNITNLCPF",  # RBD core
                "GVYYHKNNKSWMESEFRVY",   # RBD tip
                "CVADYSVLYNSASFSTFKCY"   # RBD base
            ]
            
            for i, seq in enumerate(rbd_sequences):
                # Generate 3D coordinates (simplified protein structure)
                coords = self._generate_structure_coords(len(seq), region=i)
                hydro = np.random.uniform(-2, 2)
                charge = np.random.uniform(-5, 5)
                
                epitope = AntigenEpitope(seq, coords, hydro, charge)
                epitopes.append(epitope)
        
        return epitopes
    
    def _generate_structure_coords(self, length: int, region: int) -> np.ndarray:
        """Generate realistic 3D protein structure coordinates"""
        # Simplified alpha helix/beta sheet generation
        coords = np.zeros((length, 3))
        
        if region % 2 == 0:  # Alpha helix
            for i in range(length):
                angle = i * 100 * np.pi / 180  # 100 degrees per residue
                coords[i] = [
                    2.3 * np.cos(angle),
                    2.3 * np.sin(angle),
                    1.5 * i  # 1.5 Å rise per residue
                ]
        else:  # Beta sheet
            for i in range(length):
                coords[i] = [
                    3.3 * i,  # 3.3 Å between residues
                    2 * (i % 2),  # Alternating positions
                    0
                ]
        
        return coords
    
    def _identify_glycosylation(self) -> List[int]:
        """Identify N-glycosylation sites (N-X-S/T motif)"""
        sites = []
        for i, epitope in enumerate(self.epitopes):
            seq = epitope.sequence
            for j in range(len(seq) - 2):
                if seq[j] == 'N' and seq[j+2] in ['S', 'T'] and seq[j+1] != 'P':
                    sites.append((i, j))
        return sites
    
    def _generate_conformations(self) -> List[Dict]:
        """Generate different conformational states"""
        conformations = []
        
        # Closed conformation
        conformations.append({
            'name': 'closed',
            'accessibility': 0.3,
            'stability': 0.9,
            'epitope_exposure': [0.2, 0.3, 0.1]
        })
        
        # Open conformation
        conformations.append({
            'name': 'open',
            'accessibility': 0.9,
            'stability': 0.6,
            'epitope_exposure': [0.9, 0.8, 0.7]
        })
        
        # Intermediate
        conformations.append({
            'name': 'intermediate',
            'accessibility': 0.6,
            'stability': 0.7,
            'epitope_exposure': [0.5, 0.6, 0.4]
        })
        
        return conformations
    
    def to_graph(self) -> Data:
        """Convert antigen to graph representation for GNN processing"""
        all_coords = []
        all_features = []
        
        for i, epitope in enumerate(self.epitopes):
            # Add epitope coordinates
            all_coords.append(epitope.structure_coords)
            
            # Create feature vectors for each residue
            conf = self.conformational_states[self.current_conformation]
            exposure = conf['epitope_exposure'][i]
            
            for j, aa in enumerate(epitope.sequence):
                features = [
                    epitope.hydrophobicity,
                    epitope.charge,
                    exposure,
                    float(aa in 'KR'),  # Positive charge
                    float(aa in 'DE'),  # Negative charge
                    float(aa in 'AILMFWYV'),  # Hydrophobic
                    float((i, j) in self.glycosylation_sites)  # Glycosylated
                ]
                all_features.append(features)
        
        # Combine all coordinates
        coords = np.vstack(all_coords)
        features = np.array(all_features)
        
        # Build graph based on spatial proximity
        distances = np.linalg.norm(
            coords[:, np.newaxis] - coords[np.newaxis, :], 
            axis=2
        )
        
        # Connect residues within 8 Angstroms
        edge_index = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                if distances[i, j] < 8.0:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Pad features to match expected dimension
        if features.shape[1] < cfg.feature_dim:
            padding = np.random.normal(0, 0.1, (features.shape[0], 
                                                cfg.feature_dim - features.shape[1]))
            features = np.hstack([features, padding])
        
        # Calculate realistic binding affinity
        affinity = self._calculate_binding_affinity()
        
        return Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            affinity=affinity,
            num_nodes=len(coords),
            pos=torch.tensor(coords, dtype=torch.float32)
        )
    
    def _calculate_binding_affinity(self) -> float:
        """Calculate realistic antibody binding affinity"""
        conf = self.conformational_states[self.current_conformation]
        
        # Base affinity from epitope properties
        base_affinity = 0.5
        
        # Modify based on accessibility and hydrophobicity
        for i, epitope in enumerate(self.epitopes):
            exposure = conf['epitope_exposure'][i]
            
            # Hydrophobic residues buried = lower affinity
            hydro_penalty = max(0, epitope.hydrophobicity * (1 - exposure))
            
            # Charged residues exposed = higher affinity
            charge_bonus = abs(epitope.charge) * exposure * 0.1
            
            base_affinity += charge_bonus - hydro_penalty * 0.05
        
        # Add noise and clamp
        affinity = base_affinity + np.random.normal(0, 0.05)
        return float(np.clip(affinity, 0.1, 0.95))
    
    def apply_mutations(self, mutation_sites: List[Tuple[int, int]]):
        """Apply mutations at specified epitope positions"""
        amino_acids = 'ARNDCEQGHILKMFPSTWYV'
        
        for epitope_idx, position in mutation_sites:
            if epitope_idx < len(self.epitopes):
                epitope = self.epitopes[epitope_idx]
                if position < len(epitope.sequence):
                    # Choose mutation based on chemical similarity
                    old_aa = epitope.sequence[position]
                    new_aa = self._choose_similar_amino_acid(old_aa)
                    epitope.mutate(position, new_aa)
    
    def _choose_similar_amino_acid(self, aa: str) -> str:
        """
        Choose chemically similar amino acid for realistic mutations
        MODIFIED: Choose a completely random amino acid for stronger mutations.
        """
        # Original code for realistic mutations is commented out.
        # similar_groups = [
        #     'AILMV',  # Aliphatic
        #     'FWY',    # Aromatic
        #     'ST',     # Hydroxyl
        #     'DE',     # Acidic
        #     'KRH',    # Basic
        #     'NQ',     # Amide
        #     'GP',     # Special
        #     'C'       # Cysteine
        # ]
        #
        # for group in similar_groups:
        #     if aa in group:
        #         # Higher chance of mutating within group
        #         if random.random() < 0.7:
        #             return random.choice(group.replace(aa, ''))
        #]
        # HACK APPLIED: Always choose a completely random amino acid.
        # This will cause more drastic changes to the antigen's properties.
        return random.choice('ARNDCEQGHILKMFPSTWYV'.replace(aa, ''))
    
    
def generate_realistic_antigen(variant_type: str = "wild_type", 
                             mutations: List[Tuple[int, int]] = None) -> Data:
    """Generate biologically accurate antigen"""
    antigen = BiologicalAntigen(antigen_type="viral_spike")
    
    # Apply variant-specific mutations
    if variant_type == "alpha":
        antigen.apply_mutations([(0, 5), (1, 12)])  # N501Y-like
    elif variant_type == "delta":
        antigen.apply_mutations([(0, 5), (1, 12), (2, 18)])  # L452R-like
    elif variant_type == "omicron":
        # Many mutations
        for i in range(3):
            for j in [3, 7, 12, 15, 18]:
                if j < len(antigen.epitopes[i].sequence):
                    antigen.apply_mutations([(i, j)])
    
    # Apply additional custom mutations
    if mutations:
        antigen.apply_mutations(mutations)
    
    # Randomly select conformation
    antigen.current_conformation = random.randint(0, 
                                                 len(antigen.conformational_states) - 1)
    
    return antigen.to_graph()
