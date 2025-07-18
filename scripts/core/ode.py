import matplotlib
matplotlib.use('Agg')


from threading import Thread
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import copy
import uuid
import json
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
import os
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hashlib
from scripts.config import cfg

# Removed circular import - StemGeneModule will be imported dynamically when needed
# Suppress warnings for clean output
warnings.filterwarnings('ignore')
from scripts.core.utils.detailed_logger import get_logger, trace

logger = get_logger()

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)






# ============================================================================
# True ODE-Based Continuous Depth Neural Module
# ============================================================================

class NeuralODEFunc(nn.Module):
    """Neural ODE dynamics function with proper implementation"""
    
    def __init__(self, hidden_dim: int, edge_index: torch.Tensor):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer('edge_index', edge_index)
        
        # Multiple GCN layers for richer dynamics
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
            for _ in range(3)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Residual connections
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute derivative dh/dt"""
        # Store original for residual
        h_orig = h

        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            if self.edge_index.max() >= h.size(0):
                self.edge_index = self.edge_index.clamp(max=h.size(0) - 1)

            h_new = gnn(h, self.edge_index)
            h_new = norm(h_new)
            
            if i > 0:
                # Gated residual connection
                gate_input = torch.cat([h, h_new], dim=-1)
                gate = self.gate(gate_input)
                h = gate * h_new + (1 - gate) * h
            else:
                h = h_new
        
        # Final activation
        h = torch.tanh(h)
        
        # Learnable residual connection to input
        dh = h + self.residual_weight * h_orig
        
        return dh

class ContinuousDepthGeneModule(nn.Module):
    """Gene module with true ODE-based continuous depth"""
    
    
    def __init__(self, gene_type: str, variant_id: int):
        super().__init__()
        self.gene_type = gene_type
        self.variant_id = variant_id
        self.gene_id = f"{gene_type}{variant_id}-{uuid.uuid4().hex[:8]}"
        
        # Input/output projections
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim)
        )
        
        # Learnable depth with constraints
        self.log_depth = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0
        
        # ODE solver settings
        self.ode_func = None  # Initialized per forward pass with edge_index
        
        # Gene properties
        self.position = random.random()
        self.is_active = True
        self.is_inverted = False
        self.fitness_contribution = 0.0
        
        # Epigenetic system
        self.methylation_state = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.histone_modifications = nn.Parameter(torch.zeros(4))  # H3K4me3, H3K27me3, etc.
        self.chromatin_accessibility = 1.0
        
        # History tracking
        self.transposition_history = []
        self.expression_history = deque(maxlen=100)
        self.activation_ema = 0.0
        self.is_cold = False
    
    
    def compute_depth(self) -> float:
        """Compute actual depth with constraints"""
        depth = torch.exp(self.log_depth)
        return torch.clamp(depth, cfg.min_depth, cfg.max_depth)
    
    
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Forward pass through continuous-depth ODE"""
        # --- MITIGATION 7: Skip if cold ---
        if self.is_cold:
            # Return a zero vector of the correct shape
            num_graphs = batch.max().item() + 1 if batch is not None else 1
            return torch.zeros(num_graphs, cfg.hidden_dim, device=x.device)    
    
        # Project input
        h = self.input_projection(x)
        
        # Apply epigenetic modulation
        h = self._apply_epigenetic_regulation(h)
        
        # Initialize ODE function with current edge structure
        if self.ode_func is None or self.ode_func.edge_index.shape != edge_index.shape:
            try:
                self.ode_func = NeuralODEFunc(cfg.hidden_dim, edge_index)
                # Transfer to same device
                self.ode_func = self.ode_func.to(h.device)
            except Exception as e:
                print(f"Error creating NeuralODEFunc: {e}")
                print(f"  hidden_dim: {cfg.hidden_dim}")
                print(f"  edge_index shape: {edge_index.shape}")
                print(f"  h shape: {h.shape}")
                print(f"  device: {h.device}")
                raise
        
        # Compute integration time based on learned depth
        depth = self.compute_depth()
        t = torch.linspace(0, depth.item(), cfg.ode_time_points).to(h.device)
        
        # Solve ODE using adjoint method for memory efficiency
        if self.training and cfg.gradient_checkpointing:
            h_trajectory = odeint(
                self.ode_func, h, t,
                method=cfg.ode_solver,
                rtol=cfg.ode_rtol,
                atol=cfg.ode_atol,
                adjoint_params=list(self.ode_func.parameters())
            )
        else:
            # Faster forward-mode for inference
            h_trajectory = odeint(
                self.ode_func, h, t,
                method=cfg.ode_solver,
                rtol=cfg.ode_rtol,
                atol=cfg.ode_atol
            )
        
        # Take final state
        h_final = h_trajectory[-1]
        
        # Apply inversion if needed
        if self.is_inverted:
            h_final = -h_final
        
        # Output projection
        h_out = self.output_projection(h_final)
        
        # Global pooling if needed
        if batch is not None:
            h_out = global_mean_pool(h_out, batch)
        else:
            h_out = h_out.mean(dim=0, keepdim=True)
        
        # Record expression level
        self.expression_history.append(h_out.detach().mean().item())
            
        with torch.no_grad():
            self.activation_ema = 0.95 * self.activation_ema + 0.05 * h_out.norm().item()
            
        return h_out

    
    def _apply_epigenetic_regulation(self, h: torch.Tensor) -> torch.Tensor:
        """Apply complex epigenetic regulation"""
        # Methylation silencing (CpG methylation effect)
        methylation_silencing = torch.sigmoid(self.methylation_state).mean()
        
        # Histone modification effects
        h3k4me3 = torch.sigmoid(self.histone_modifications[0])  # Activation mark
        h3k27me3 = torch.sigmoid(self.histone_modifications[1])  # Repression mark
        h3k9ac = torch.sigmoid(self.histone_modifications[2])    # Activation mark
        h3k9me3 = torch.sigmoid(self.histone_modifications[3])   # Repression mark
        
        # Compute chromatin state
        activation_marks = (h3k4me3 + h3k9ac) / 2
        repression_marks = (h3k27me3 + h3k9me3) / 2
        
        # Chromatin accessibility from histone state
        self.chromatin_accessibility = torch.clamp(
            activation_marks - repression_marks + 0.5, 0, 1
        ).item()
        
        # Apply regulation
        regulation_factor = self.chromatin_accessibility * (1 - methylation_silencing)
        h_regulated = h * regulation_factor
        
        return h_regulated
    
    
    def add_methylation(self, sites: torch.Tensor, level: float):
        logger.debug("Entering ContinuousDepthGeneModule.add_methylation")
        """Add methylation at specific sites"""
        with torch.no_grad():
            self.methylation_state.data[sites] += level
            self.methylation_state.data = torch.clamp(self.methylation_state.data, 0, 1)
    
    
    def modify_histones(self, modification_type: str, level: float):
        logger.debug("Entering ContinuousDepthGeneModule.modify_histones")
        """Modify histone marks"""
        histone_map = {
            'h3k4me3': 0, 'h3k27me3': 1, 
            'h3k9ac': 2, 'h3k9me3': 3
        }
        
        if modification_type in histone_map:
            idx = histone_map[modification_type]
            with torch.no_grad():
                self.histone_modifications.data[idx] += level
                self.histone_modifications.data = torch.clamp(
                    self.histone_modifications.data, -1, 1
                )
    
    def clone(self) -> 'ContinuousDepthGeneModule':
        """Efficient cloning method that preserves all gene state"""
        # Create new instance with same constructor args
        new_gene = self.__class__(self.gene_type, self.variant_id)
        
        # Copy state dict
        new_gene.load_state_dict(self.state_dict())
        
        # Generate new unique ID
        new_gene.gene_id = f"{self.gene_type}{self.variant_id}-{uuid.uuid4().hex[:8]}"
        
        # Copy non-parameter attributes
        attrs_to_copy = [
            'position', 'is_active', 'fitness_contribution',
            'transposition_history', 'activation_count',
            'total_gradient_norm', 'last_update_generation'
        ]
        
        for attr in attrs_to_copy:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if isinstance(value, torch.Tensor):
                    setattr(new_gene, attr, value.clone())
                elif isinstance(value, list):
                    setattr(new_gene, attr, value.copy())
                else:
                    setattr(new_gene, attr, value)
        
        return new_gene
    
# In the ContinuousDepthGeneModule class:

    
    def transpose(self, stress_level: float, population_diversity: float) -> Tuple[Optional['ContinuousDepthGeneModule'], Optional[str]]:
        """Intelligent transposition based on multiple factors"""
        # Base probability modulated by stress and diversity
        transpose_prob = cfg.base_transpose_prob * (1 + stress_level * cfg.stress_multiplier)
        
        # Reduce transposition if diversity is already high
        if population_diversity > cfg.shannon_entropy_target:
            transpose_prob *= 0.5
        
        # --- FIX APPLIED HERE ---
        # If no transposition occurs, return a tuple of (None, None) to maintain a consistent return type.
        if random.random() > transpose_prob:
            return None, None # Was returning a single `None`
        
        # Decide action based on gene performance and stress
        if self.fitness_contribution < 0.3:  # Poor performing gene
            action_weights = [0.1, 0.2, 0.2, 0.5]  # Favor deletion
        elif stress_level > 0.7:  # High stress
            action_weights = [0.2, 0.5, 0.2, 0.1]  # Favor duplication
        else:  # Normal conditions
            action_weights = [0.5, 0.2, 0.2, 0.1]  # Favor jumping
        
        action = random.choices(
            ['jump', 'duplicate', 'invert', 'delete'],
            weights=action_weights
        )[0]
        
        # Energy cost of transposition
        self.fitness_contribution -= cfg.transposition_energy_cost
        
        # This part is already correct as _execute_transposition returns a child or None
        child = self._execute_transposition(action, stress_level)
        return child, action
    
    
# In transposable_immune_ai_production_complete.py

# In the ContinuousDepthGeneModule class:
    
    def _execute_transposition(self, action: str, stress_level: float) -> Optional['ContinuousDepthGeneModule']:
        """Execute specific transposition action"""
        timestamp = datetime.now().isoformat()
        from scripts.core.quantum_gene import QuantumGeneModule

        if action == 'jump':
            old_pos = self.position
            # Biased jump based on gene type
            if self.gene_type == 'V':
                self.position = np.clip(np.random.normal(0.15, 0.1), 0, 0.3)
            elif self.gene_type == 'D':
                self.position = np.clip(np.random.normal(0.45, 0.1), 0.3, 0.6)
            else:  # J
                self.position = np.clip(np.random.normal(0.8, 0.1), 0.6, 1.0)
            
            self.transposition_history.append({
                'time': timestamp,
                'action': 'jump',
                'from': old_pos,
                'to': self.position,
                'stress': stress_level
            })
            
        elif action == 'duplicate':
            # ============================================================================
            # START OF FIX
            # ============================================================================
            # Determine the device of the parent gene to ensure the child is on the same device.
            parent_device = next(self.parameters()).device
            # ============================================================================
            # END OF FIX
            # ============================================================================

            if stress_level > 0.9 and random.random() < 0.2:
                print("    ‼️‼️ A high-stress event triggered a Quantum Leap!  ‼️‼️")
                # Instead of a normal duplicate, create and return a quantum version
                child = QuantumGeneModule(self.gene_type, self.variant_id)
                child.position = np.clip(self.position + np.random.normal(0, 0.05), 0, 1)
                
                self.transposition_history.append({
                    'time': timestamp,
                    'action': 'quantum_leap', # Log it as a special event
                    'child_id': child.gene_id,
                    'stress': stress_level
                })
                # ============================================================================
                # START OF FIX
                # ============================================================================
                # Move the newly created CPU-based module to the correct device before returning.
                return child.to(parent_device)
                # ============================================================================
                # END OF FIX
                # ============================================================================

            # If it's not a quantum leap, proceed with a normal duplication
            child = self.clone()
            child.position = np.clip(self.position + np.random.normal(0, 0.05), 0, 1)
            
            # Mutate child's parameters
            with torch.no_grad():
                for param in child.parameters():
                    param.data += torch.randn_like(param) * 0.1 * stress_level
                child.log_depth.data += np.random.normal(0, 0.2)
            
            # Partial epigenetic inheritance
            child.methylation_state.data *= cfg.methylation_inheritance
            child.histone_modifications.data *= cfg.methylation_inheritance
            
            self.transposition_history.append({
                'time': timestamp,
                'action': 'duplicate',
                'child_id': child.gene_id,
                'stress': stress_level
            })
            
            return child
        
                    
        elif action == 'invert':
            self.is_inverted = not self.is_inverted
            
            # Inversion affects regulatory elements
            with torch.no_grad():
                self.histone_modifications.data = -self.histone_modifications.data
            
            self.transposition_history.append({
                'time': timestamp,
                'action': 'invert',
                'state': self.is_inverted,
                'stress': stress_level
            })
            
        elif action == 'delete':
            self.is_active = False
            self.chromatin_accessibility = 0.0
            
            self.transposition_history.append({
                'time': timestamp,
                'action': 'delete',
                'stress': stress_level
            })



        
        return None







