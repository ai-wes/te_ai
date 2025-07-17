
import matplotlib
matplotlib.use('Agg')

import asyncio
import websockets
import json
import os
from threading import Thread
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from torch_geometric.utils import to_undirected, add_self_loops
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
from scripts.core.ode import NeuralODEFunc, ContinuousDepthGeneModule
from scripts.core.utils.detailed_logger import get_logger, trace

logger = get_logger()


warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)




# ============================================================================#
# QUANTUM GENE MODULE
# ============================================================================
# ============================================================================
# QUANTUM DREAM SYSTEM - PARALLEL REALITY ANTIGEN EXPLORATION
# ============================================================================





class QuantumGeneModule(ContinuousDepthGeneModule):
    """
    Quantum-inspired gene module that maintains superposition of multiple
    computational pathways until observation (evaluation).
    """
    @trace
    def __init__(self, gene_type: str, variant_id: int):
        super().__init__(gene_type, variant_id)
        
        # Quantum state represented as real amplitudes for two basis states
        # We use real numbers and handle phase separately for PyTorch compatibility
        self.alpha_amplitude = nn.Parameter(torch.tensor(1.0))  # |0⟩ amplitude
        self.beta_amplitude = nn.Parameter(torch.tensor(0.0))   # |1⟩ amplitude
        # Store phase as sin/cos pair for smooth gradients
        self.phase_sin = nn.Parameter(torch.tensor(0.0))  # sin(phase)
        self.phase_cos = nn.Parameter(torch.tensor(1.0))  # cos(phase)
        
        # Normalize sin/cos on init
        self._normalize_phase_components()
        
        # Coherence decay rate (how fast superposition collapses)
        self.decoherence_rate = nn.Parameter(torch.tensor(0.1))
        
        # Alternative computational pathways
        self.alt_projection = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),  # Different activation
            nn.Dropout(0.15)  # Different dropout
        )
        
        # Interference pathway (emerges from quantum interaction)
        self.interference_projection = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        
        # Measurement operator (collapses superposition)
        self.measurement_gate = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Track coherence over time
        self.coherence_steps = 0
        
        # Flag to track if normalization is needed after backward pass
        self._needs_normalization = False
    
    @trace
    def _normalize_phase_components(self):
        """Normalize sin/cos phase components"""
        with torch.no_grad():
            norm = torch.sqrt(self.phase_sin**2 + self.phase_cos**2 + 1e-8)
            self.phase_sin.data /= norm
            self.phase_cos.data /= norm
    
    @trace
    def get_phase(self):
        """Get phase angle from sin/cos components"""
        return torch.atan2(self.phase_sin, self.phase_cos)
    
    @trace
    def normalize_quantum_state(self):
        """Ensure quantum state is normalized (|α|² + |β|² = 1)"""
        # This should be called AFTER backward pass in a no_grad context
        norm = torch.sqrt(self.alpha_amplitude**2 + self.beta_amplitude**2 + 1e-8)
        if self.training:
            # During training, just flag that normalization is needed
            self._needs_normalization = True
            return norm  # Return norm for potential use
        else:
            # During evaluation, normalize immediately
            with torch.no_grad():
                self.alpha_amplitude.data = self.alpha_amplitude / norm
                self.beta_amplitude.data = self.beta_amplitude / norm
    
    @trace
    def post_backward_normalize(self):
        """Call this after backward pass to normalize quantum state"""
        if self._needs_normalization:
            with torch.no_grad():
                norm = torch.sqrt(self.alpha_amplitude**2 + self.beta_amplitude**2 + 1e-8)
                self.alpha_amplitude.data /= norm
                self.beta_amplitude.data /= norm
                self._needs_normalization = False
                
                # Also normalize phase components
                self._normalize_phase_components()
    
    @trace
    def compute_probabilities(self):
        """Compute measurement probabilities from amplitudes"""
        # Don't normalize during forward pass to maintain gradients
        norm_sq = self.alpha_amplitude**2 + self.beta_amplitude**2 + 1e-8
        prob_0 = self.alpha_amplitude ** 2 / norm_sq
        prob_1 = self.beta_amplitude ** 2 / norm_sq
        return prob_0, prob_1
    
    @trace
    def compute_interference(self, prob_0, prob_1):
        """Compute quantum interference term"""
        # Interference strength depends on amplitudes and phase
        amplitude_product = 2 * torch.sqrt(prob_0 * prob_1 + 1e-8)
        # Use phase_cos directly for interference (real part of e^{i*phase})
        interference = amplitude_product * self.phase_cos
        return interference
    
    @trace
    def apply_decoherence(self):
        """Apply environmental decoherence"""
        self.coherence_steps += 1
        
        # Clamp decoherence rate to [0, 1]
        with torch.no_grad():
            self.decoherence_rate.data = torch.clamp(self.decoherence_rate.data, 0.0, 1.0)
        
        # Exponential decay of coherence
        coherence = torch.exp(-self.decoherence_rate * self.coherence_steps)
        
        # As coherence decreases, state tends toward classical mixture
        if not self.is_cold:  # Apply decoherence even when cold
            with torch.no_grad():
                # Move toward measurement basis
                if self.alpha_amplitude.abs() > self.beta_amplitude.abs():
                    target_alpha = torch.sqrt(coherence + (1 - coherence))
                    target_beta = torch.sqrt(1 - target_alpha**2)
                else:
                    target_beta = torch.sqrt(coherence + (1 - coherence))
                    target_alpha = torch.sqrt(1 - target_beta**2)
                
                # Smooth transition
                decay_rate = 0.1
                self.alpha_amplitude.data = (1 - decay_rate) * self.alpha_amplitude.data + decay_rate * target_alpha
                self.beta_amplitude.data = (1 - decay_rate) * self.beta_amplitude.data + decay_rate * target_beta
                
                # Normalize after update
                norm = torch.sqrt(self.alpha_amplitude**2 + self.beta_amplitude**2 + 1e-8)
                self.alpha_amplitude.data /= norm
                self.beta_amplitude.data /= norm
    
    @trace
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with a more efficient, fused quantum pathway.
        OPTIMIZED: Runs a single ODE on a superposed state to prevent computational explosion.
        """
        if self.is_cold:
            # Still apply decoherence even when cold
            self.apply_decoherence()
            num_graphs = batch.max().item() + 1 if batch is not None else 1
            return torch.zeros(num_graphs, cfg.hidden_dim, device=x.device)
        
        # Get probabilities for each computational basis
        prob_0, prob_1 = self.compute_probabilities()
        
        # Create the two potential initial states
        h_0_initial = self.input_projection(x)
        h_1_initial = self.alt_projection(x)
        
        # Create a single, superposed initial state for the ODE
        # This blends the two pathways before the expensive computation
        h_superposed_initial = torch.sqrt(prob_0) * h_0_initial + torch.sqrt(prob_1) * h_1_initial
        
        # Apply epigenetic regulation to the combined state
        h_superposed_initial = self._apply_epigenetic_regulation(h_superposed_initial)
        
        # Initialize ODE function if needed
        if self.ode_func is None or self.ode_func.edge_index.shape != edge_index.shape:
            self.ode_func = NeuralODEFunc(cfg.hidden_dim, edge_index).to(h_superposed_initial.device)
        
        # ONLY ONE ODE CALL - Major optimization
        depth = self.compute_depth()
        t = torch.linspace(0, depth.item(), cfg.ode_time_points).to(h_superposed_initial.device)
        h_trajectory = odeint(
            self.ode_func, 
            h_superposed_initial, 
            t, 
            method=cfg.ode_solver,
            rtol=cfg.ode_rtol,
            atol=cfg.ode_atol
        )
        h_final_superposed = h_trajectory[-1]
        
        # Add quantum interference effects
        interference = self.compute_interference(prob_0, prob_1)
        if abs(interference) > 0.01:
            # Create interference features
            h_interference = self.interference_projection(
                torch.cat([h_final_superposed, h_final_superposed], dim=-1)
            )
            h_final_superposed = h_final_superposed + interference * h_interference
        
        # During evaluation, collapse to deterministic outcome
        if not self.training:
            if prob_0 > prob_1:
                # Collapse to |0⟩ basis
                h_final = h_final_superposed
            else:
                # Collapse to |1⟩ basis with phase modulation using sin/cos representation
                phase_rotation = self.phase_cos + 1j * self.phase_sin
                h_final = h_final_superposed * phase_rotation.real
        else:
            # During training, maintain superposition
            h_final = h_final_superposed
            self.apply_decoherence()
            
            # Randomly collapse with small probability (quantum Zeno effect)
            if random.random() < 0.05:
                outcome, _ = self.measure_quantum_state()
                if outcome == 1:
                    phase_rotation = self.phase_cos + 1j * self.phase_sin
                    h_final = h_final * phase_rotation.real
        
        # Apply inversion if needed
        if self.is_inverted:
            h_final = -h_final
        
        # Output projection and pooling
        h_out = self.output_projection(h_final)
        if batch is not None:
            h_out = global_mean_pool(h_out, batch)
        else:
            h_out = h_out.mean(dim=0, keepdim=True)
        
        # Record expression history
        self.expression_history.append(h_out.detach().mean().item())
        with torch.no_grad():
            self.activation_ema = 0.95 * self.activation_ema + 0.05 * h_out.norm().item()
        
        return h_out
    
    @trace
    def transpose(self, stress_level: float, population_diversity: float) -> Tuple[Optional['QuantumGeneModule'], Optional[str]]:
        """Quantum-enhanced transposition with entanglement effects"""
        # Get base transposition result
        child, action = super().transpose(stress_level, population_diversity)
        
        # If transposition occurred and child is quantum
        if child and isinstance(child, QuantumGeneModule):
            # Quantum leap under high stress
            if stress_level > 0.8 and random.random() < 0.1:
                print("    ‼️‼️ A high-stress event triggered a Quantum Leap!  ‼️‼️")
                
                # Create entangled pair
                with torch.no_grad():
                    # Parent and child become entangled
                    self.entangle_with(child)
                    
                    # Boost child's quantum properties
                    child.decoherence_rate.data *= 0.5  # Slower decoherence
                    # Set random phase using sin/cos
                    random_phase = np.random.uniform(-np.pi, np.pi)
                    child.phase_sin.data = torch.sin(torch.tensor(random_phase))
                    child.phase_cos.data = torch.cos(torch.tensor(random_phase))
                    child._normalize_phase_components()
                
                return child, "quantum_leap"
        
        return child, action
    
    @trace
    def entangle_with(self, other_gene: 'QuantumGeneModule'):
        """Create entanglement between two quantum genes"""
        if not isinstance(other_gene, QuantumGeneModule):
            return
        
        # Bell state preparation (maximally entangled)
        with torch.no_grad():
            # |Φ+⟩ = (|00⟩ + |11⟩) / √2
            self.alpha_amplitude.data = torch.tensor(1.0 / torch.sqrt(torch.tensor(2.0)))
            self.beta_amplitude.data = torch.tensor(1.0 / torch.sqrt(torch.tensor(2.0)))
            other_gene.alpha_amplitude.data = self.alpha_amplitude.data.clone()
            other_gene.beta_amplitude.data = self.beta_amplitude.data.clone()
            
            # Correlated phases - self has phase 0, other has phase pi
            self.phase_sin.data = torch.tensor(0.0)
            self.phase_cos.data = torch.tensor(1.0)
            other_gene.phase_sin.data = torch.tensor(0.0)
            other_gene.phase_cos.data = torch.tensor(-1.0)
    
    @trace
    def measure_quantum_state(self) -> Tuple[int, float]:
        """
        Perform measurement and return (outcome, probability)
        outcome: 0 or 1
        probability: probability of that outcome
        """
        prob_0, prob_1 = self.compute_probabilities()
        
        # Quantum measurement
        if random.random() < prob_0.item():
            outcome = 0
            probability = prob_0.item()
            # Collapse to |0⟩
            with torch.no_grad():
                self.alpha_amplitude.data = torch.tensor(1.0)
                self.beta_amplitude.data = torch.tensor(0.0)
        else:
            outcome = 1
            probability = prob_1.item()
            # Collapse to |1⟩
            with torch.no_grad():
                self.alpha_amplitude.data = torch.tensor(0.0)
                self.beta_amplitude.data = torch.tensor(1.0)
        
        # Reset coherence
        self.coherence_steps = 0
        
        return outcome, probability
    
    @trace
    def get_quantum_state_string(self) -> str:
        """Get human-readable quantum state"""
        prob_0, prob_1 = self.compute_probabilities()
        phase = self.get_phase().item()
        
        return (f"|ψ⟩ = {prob_0.sqrt():.2f}|0⟩ + "
                f"{prob_1.sqrt():.2f}e^(i{phase:.2f})|1⟩")
    
    @trace
    @staticmethod
    def normalize_all_quantum_states(population):
        """Call post_backward_normalize on all quantum genes in population"""
        for cell in population:
            if hasattr(cell, 'genes'):
                for gene in cell.genes:
                    if isinstance(gene, QuantumGeneModule):
                        gene.post_backward_normalize()








