"""
Drug Discovery Specialized Gene Modules
======================================

Extends the existing TE-AI gene architecture with drug-specific
recognition capabilities while maintaining the transposition and
evolution mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.core.stem_gene_module import StemGeneModule
from scripts.config import cfg


class BindingPocketGene(ContinuousDepthGeneModule):
    """
    Specialized gene for recognizing drug binding pockets.
    Extends ContinuousDepthGeneModule with pocket-specific features.
    """
    
    def __init__(
        self,
        variant_id: int,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        position: float = 0.15,  # V-region position for recognition
        depth_min: float = 0.5,
        depth_max: float = 2.0
    ):
        super().__init__(
            gene_type='V',  # Variable region for recognition
            variant_id=variant_id
        )
        
        # Store the additional parameters as instance variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth_min = depth_min
        self.depth_max = depth_max
        
        # Add pocket-specific layers
        self.pocket_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 4 pocket features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Pocket attention mechanism
        self.pocket_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Volume-aware processing
        self.volume_processor = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, 
                global_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Process antigen with focus on binding pocket features.
        """
        # Get base processing from parent
        base_output = super().forward(x, edge_index, batch, global_features=global_features)
        base_metadata = {}
        
        # Extract pocket features if available
        if hasattr(batch, 'pocket_features') and batch.pocket_features is not None:
            # Encode pocket features
            pocket_enc = self.pocket_encoder(batch.pocket_features)
            
            # Apply attention between base output and pocket features
            attended, attention_weights = self.pocket_attention(
                base_output.unsqueeze(0),
                pocket_enc.unsqueeze(0),
                pocket_enc.unsqueeze(0)
            )
            attended = attended.squeeze(0)
            
            # Volume-based modulation
            if batch.pocket_features.shape[1] > 0:
                volumes = batch.pocket_features[:, 0:1]  # First feature is volume
                volume_mod = self.volume_processor(volumes)
                output = base_output + attended + volume_mod
            else:
                output = base_output + attended
                
            # Update metadata
            base_metadata['pocket_attention'] = attention_weights
            base_metadata['pocket_focus'] = True
        else:
            output = base_output
            base_metadata['pocket_focus'] = False
            
        return output, base_metadata
    
    def transpose(self, target_position: Optional[float] = None) -> 'BindingPocketGene':
        """Maintain pocket recognition during transposition"""
        new_gene = super().transpose(target_position)
        # Preserve pocket-specific components
        new_gene.pocket_encoder = self.pocket_encoder
        new_gene.pocket_attention = self.pocket_attention
        new_gene.volume_processor = self.volume_processor
        return new_gene


class PharmacophoreGene(QuantumGeneModule):
    """
    Quantum gene specialized for pharmacophore recognition.
    Uses superposition states to recognize multiple binding modes.
    """
    
    def __init__(
        self,
        variant_id: int,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        position: float = 0.45,  # D-region for diversity
        n_qubits: int = 8
    ):
        super().__init__(
            gene_type='D',  # Diversity region for pharmacophore variation
            variant_id=variant_id
        )
        
        # Store the additional parameters as instance variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        
        # Pharmacophore feature extractors
        self.hydrophobic_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_qubits)
        )
        
        self.hbond_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_qubits)
        )
        
        self.aromatic_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_qubits)
        )
        
        # Quantum circuit for pharmacophore matching
        self.pharmacophore_circuit = self._build_pharmacophore_circuit()
        
    def _build_pharmacophore_circuit(self) -> nn.Module:
        """Build quantum circuit for pharmacophore recognition"""
        return nn.Sequential(
            nn.Linear(self.n_qubits * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_qubits),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                global_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Quantum processing with pharmacophore recognition.
        """
        # Base quantum processing
        # QuantumGeneModule doesn't accept global_features
        quantum_output = super().forward(x, edge_index, batch)
        metadata = {}
        
        # Extract pharmacophore features
        hydrophobic = self.hydrophobic_detector(x)
        hbond = self.hbond_detector(x)
        aromatic = self.aromatic_detector(x)
        
        # Combine in quantum superposition
        pharmacophore_state = torch.cat([hydrophobic, hbond, aromatic], dim=-1)
        pharmacophore_encoding = self.pharmacophore_circuit(pharmacophore_state)
        
        # Interfere with base quantum state
        if hasattr(self, 'quantum_state') and self.quantum_state is not None:
            interference = self.compute_interference(
                self.quantum_state,
                pharmacophore_encoding
            )
            output = quantum_output + 0.5 * interference
        else:
            output = quantum_output
            
        metadata['pharmacophore_features'] = {
            'hydrophobic': hydrophobic.mean().item(),
            'hbond': hbond.mean().item(),
            'aromatic': aromatic.mean().item()
        }
        
        return output, metadata


class AllostericGene(StemGeneModule):
    """
    Stem gene that can differentiate based on allosteric site detection.
    Specializes in finding non-active site drug targets.
    """
    
    def __init__(
        self,
        variant_id: int,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        position: float = 0.5,  # Stem position
        differentiation_threshold: float = 0.7
    ):
        super().__init__(
            gene_types=['V', 'D', 'J', 'A']  # A for Allosteric
        )
        
        # Store the additional parameters as instance variables
        self.variant_id = variant_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.differentiation_threshold = differentiation_threshold
        
        # Allosteric site detector
        self.allosteric_detector = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Distance-based site analyzer
        self.distance_analyzer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                global_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Process with allosteric site detection.
        """
        # Base stem cell processing
        stem_output = super().forward(x, edge_index, batch, global_features=global_features)
        metadata = {}
        
        # Detect potential allosteric sites
        allosteric_features = self.allosteric_detector(x, edge_index)
        
        # Analyze distances from active site
        if hasattr(batch, 'active_site_mask'):
            active_site_center = x[batch.active_site_mask].mean(dim=0, keepdim=True)
            
            # Compute distance features
            distances = torch.cdist(x, active_site_center)
            distance_features = torch.cat([
                allosteric_features,
                distances.expand(-1, allosteric_features.shape[1])
            ], dim=-1)
            
            # Identify allosteric potential
            allosteric_scores = self.distance_analyzer(distance_features)
            
            # Modulate output based on allosteric detection
            output = stem_output * (1 + allosteric_scores)
            
            metadata['allosteric_score'] = allosteric_scores.mean().item()
            metadata['found_allosteric'] = (allosteric_scores > 0.5).any().item()
        else:
            output = stem_output
            metadata['allosteric_score'] = 0.0
            metadata['found_allosteric'] = False
            
        return output, metadata
    
    def sense_drug_discovery_needs(self, population_stats: Dict) -> torch.Tensor:
        """
        Sense what type of drug discovery gene is needed.
        """
        needs = torch.zeros(5)  # V, D, J, S, Q types
        
        # Need more pocket detectors if low affinity
        if population_stats.get('mean_fitness', 0) < 0.5:
            needs[0] = 0.8  # More V genes
            
        # Need pharmacophore genes if low diversity
        if population_stats.get('diversity', {}).get('shannon_index', 0) < 2.0:
            needs[1] = 0.7  # More D genes
            needs[4] = 0.5  # Some Q genes
            
        # Need allosteric genes if traditional binding saturated
        if population_stats.get('mean_fitness', 0) > 0.7 and \
           population_stats.get('fitness_variance', 1.0) < 0.1:
            needs[3] = 0.9  # More stem cells to find new sites
            
        return needs