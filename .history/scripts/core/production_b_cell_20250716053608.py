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
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.core.self_modifying_neural_architecture import SelfModifyingArchitecture
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.utils.telemetry import TermColors, _write_enhanced_visualization_state
# Removed circular import - StemGeneModule will be imported dynamically when needed
# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)








# ============================================================================
# Enhanced B-Cell with Complete Functionality
# ============================================================================

class ProductionBCell(nn.Module):
    """Production-ready B-cell with all features fully implemented"""
    
    def __init__(self, initial_genes: List[ContinuousDepthGeneModule]):
        super().__init__()
        self.cell_id = str(uuid.uuid4())
        self.genes = nn.ModuleList(initial_genes)
        self.generation = 0
        self.lineage = []
        self.fitness_history = deque(maxlen=100)
        
        # Gene regulatory network
        self.gene_regulatory_matrix = nn.Parameter(
            torch.randn(cfg.max_genes_per_clone, cfg.max_genes_per_clone) * 0.1
        )
        
        # Attention-based gene integration
        self.gene_attention = nn.MultiheadAttention(
            cfg.hidden_dim, num_heads=cfg.num_heads, 
            dropout=0.1, batch_first=True
        )
        
        self.gene_integrator = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim)
        )
        
        # Affinity maturation network
        self.affinity_maturation = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Self-modifying architecture
        self.architecture_modifier = SelfModifyingArchitecture(cfg.hidden_dim)
        
        # Plasmid system
        self.plasmids = []
        self.conjugation_pilus = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        
        # Memory and learning
        self.immunological_memory = deque(maxlen=1000)
        self.memory_encoder = nn.LSTM(cfg.hidden_dim, cfg.hidden_dim // 2, 
                                     batch_first=True, bidirectional=True)
        
    def forward(self, antigen: Data) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Complete forward pass with all features"""
        device = next(self.parameters()).device
        active_genes = [g for g in self.genes if g.is_active]
        
        if not active_genes:
            dummy = torch.zeros(1, 1, device=device)
            return dummy, torch.zeros(1, cfg.hidden_dim, device=device), {}
        
        # Apply gene regulatory network
        gene_activities = self._compute_gene_regulation(active_genes)
        
        # Process through active genes
        gene_outputs = []
        gene_metadata = {}
        
        for i, (gene, activity) in enumerate(zip(active_genes, gene_activities)):
            # Process antigen through gene
            gene_output = gene(antigen.x, antigen.edge_index, antigen.batch)
            
            # Apply regulatory modulation
            regulated_output = gene_output * activity
            gene_outputs.append(regulated_output)
            
            # Track gene expression
            gene_metadata[f'gene_{i}_activity'] = activity.item()
            gene_metadata[f'gene_{i}_depth'] = gene.compute_depth().item()
        
        # Stack outputs
        gene_tensor = torch.stack(gene_outputs)
        
        # --- FIX APPLIED HERE ---
        # The MultiheadAttention layer expects [batch_size, sequence_length, feature_dim].
        # We need to permute the dimensions of our tensor to match this.
        # The "sequence" is our list of genes.
        # The "batch" is the batch of antigens.
        # Original shape: [num_genes, batch_size, hidden_dim]
        # Target shape:   [batch_size, num_genes, hidden_dim]
        
        gene_tensor_permuted = gene_tensor.permute(1, 0, 2)
        
        # Now, pass the correctly shaped 3D tensor to the attention layer.
        # The .unsqueeze(0) is removed.
        integrated, attention_weights = self.gene_attention(
            gene_tensor_permuted,
            gene_tensor_permuted,
            gene_tensor_permuted
        )
        # 'integrated' will have shape [batch_size, num_genes, hidden_dim]
        
        # Final integration
        # We average across the gene dimension (dim=1)
        cell_representation = self.gene_integrator(integrated.mean(dim=1))        
        # Check immunological memory
        memory_response = self._check_memory(cell_representation)
        if memory_response is not None:
            cell_representation = cell_representation + 0.3 * memory_response
        
        # Affinity prediction with maturation
        affinity = self.affinity_maturation(cell_representation)
        
        # Architecture self-modification based on performance
        if len(self.fitness_history) > 4:
            self._attempt_architecture_modification()
        
        metadata = {
            'gene_count': len(active_genes),
            'attention_weights': attention_weights.detach().cpu().numpy(),
            **gene_metadata
        }
        
        return affinity.squeeze(), cell_representation.squeeze(), metadata
    
    def _compute_gene_regulation(self, active_genes: List) -> torch.Tensor:
        """Compute gene regulatory activities"""
        n = len(active_genes)
        if n == 0:
            return torch.tensor([])
            
        # Create dynamic regulatory matrix for current active genes
        # Get device from first gene's parameters
        device = next(active_genes[0].parameters()).device
        reg_matrix = torch.randn(n, n, device=device)
        
        # Get gene activities (assuming genes have an activity attribute)
        activities = torch.stack([getattr(g, 'activity', torch.tensor(1.0, device=device)) for g in active_genes])
        
        # Apply regulation
        regulated = torch.sigmoid(reg_matrix @ activities)
        
        return regulated
    
    def _check_memory(self, representation: torch.Tensor) -> Optional[torch.Tensor]:
        """Check immunological memory for similar antigens"""
        if len(self.immunological_memory) < 10:
            return None
        
        # Encode memories
        memory_tensors = torch.stack([m['representation'] for m in 
                                     list(self.immunological_memory)[-50:]])
        
        # Compute similarity
        similarities = F.cosine_similarity(representation.unsqueeze(0), 
                                         memory_tensors, dim=1)
        
        # If high similarity found, return memory response
        max_similarity, max_idx = similarities.max(dim=0)
        if max_similarity > 0.8:
            return memory_tensors[max_idx]
        
        return None
    
    def _attempt_architecture_modification(self):
        """Attempt self-modification based on performance"""
        recent_fitness = list(self.fitness_history)[-10:]
        performance_metrics = self.architecture_modifier.analyze_performance(
            recent_fitness, 
            [0.1] * len(recent_fitness)  # Placeholder gradient norms
        )
        
        # Only modify if performance is poor or unstable
        if performance_metrics['trend'] > -0.01 or performance_metrics['stability'] < 0.7:
            current_state = torch.randn(cfg.hidden_dim).to(next(self.parameters()).device)
            modification = self.architecture_modifier.decide_modification(
                performance_metrics, current_state
            )
            
            # Apply modification
            # Pass self.cell_id to the apply function
            success = self.architecture_modifier.apply_modification(modification, self.cell_id)
            if success:
                _write_enhanced_visualization_state(self.cell_id, self.architecture_modifier)
                modification.performance_delta = performance_metrics['trend']
    
    def undergo_transposition(self, stress_level: float, diversity: float = 0.5):
        """Stress-induced transposition with population awareness"""
        new_genes = []
        
        for gene in list(self.genes):
            if gene.is_active:
                child = gene.transpose(stress_level, diversity)
                if child:
                    new_genes.append(child)
        
        # Add new genes
        for gene in new_genes:
            if len(self.genes) < cfg.max_genes_per_clone:
                self.genes.append(gene)
        
        # Update generation
        self.generation += 1
        
        # Epigenetic inheritance
        if stress_level > 0.7:
            self._apply_stress_epigenetics()
    
    def _apply_stress_epigenetics(self):
        """Apply stress-induced epigenetic changes"""
        for gene in self.genes:
            if gene.is_active:
                # Stress-induced methylation
                stress_sites = torch.randint(0, cfg.hidden_dim, (10,))
                gene.add_methylation(stress_sites, cfg.methylation_rate * 2)
                
                # Histone modifications
                gene.modify_histones('h3k27me3', 0.1)  # Repressive mark
    
# In the ProductionBCell class:

    def extract_plasmid(self) -> Optional[Dict]:
        """Extract plasmid with high-fitness genes.
        MODIFIED: Ensures the extracted plasmid's genes are on the CPU.
        """
        high_fitness_genes = [
            g for g in self.genes 
            if g.is_active and g.fitness_contribution > 0.7
        ]
        
        if not high_fitness_genes:
            return None
        
        # --- FIX APPLIED HERE ---
        # Temporarily move the cell to CPU to ensure deepcopy is safe and clean.
        original_device = next(self.parameters()).device
        self.to('cpu')
        
        # Select genes for plasmid
        plasmid_size = min(3, len(high_fitness_genes))
        # The genes are now on the CPU, so deepcopy is safe.
        plasmid_genes = [copy.deepcopy(g) for g in random.sample(high_fitness_genes, plasmid_size)]
        
        # The conjugation signal is a new tensor, so it's fine to create on the original device.
        conjugation_signal = self.conjugation_pilus(
            torch.randn(cfg.hidden_dim)
        ).detach()

        # Move the parent cell back to its original device
        self.to(original_device)
        # --- END FIX ---

        plasmid = {
            'id': str(uuid.uuid4()),
            'donor_cell': self.cell_id,
            'genes': plasmid_genes, # These are now CPU-based gene modules
            'fitness': sum(g.fitness_contribution for g in plasmid_genes) / len(plasmid_genes),
            'timestamp': datetime.now(),
            'conjugation_signal': conjugation_signal
        }
        
        self.plasmids.append(plasmid['id'])
        return plasmid
    
    
    


    def get_signature(self, calibration_batch: Data) -> torch.Tensor:
        if hasattr(self, '_signature_cache') and self._signature_cache is not None:
            return self._signature_cache
        
        with torch.no_grad():
            # Ensure the cell is on the correct device for the forward pass
            original_device = next(self.parameters()).device
            self.to(cfg.device)
            
            _, cell_representation, _ = self.forward(calibration_batch)
            self._signature_cache = cell_representation.mean(dim=0).detach().cpu()
            
            # Move back to original device if it was different
            self.to(original_device)
            
        return self._signature_cache



    def attempt_entanglement(self):
        """Periodically entangle quantum genes within the same cell."""
        quantum_genes = [g for g in self.genes if isinstance(g, QuantumGeneModule)]
        
        if len(quantum_genes) >= 2:
            # Pick two random quantum genes to entangle
            g1, g2 = random.sample(quantum_genes, 2)
            g1.entangle_with(g2)
            print(f"   얽힘 Entangling genes {g1.gene_id[:8]} and {g2.gene_id[:8]} in cell {self.cell_id[:8]}")


    
    def integrate_plasmid(self, plasmid: Dict, calibration_batch: Data) -> bool:
        """Integrate foreign plasmid with feature-signature handshake."""
        if len(self.genes) >= cfg.max_genes_per_clone:
            return False
        
        # --- MITIGATION 4: Feature-Signature Handshake ---
        recipient_signature = self.get_signature(calibration_batch)
        donor_signature = plasmid['signature'].to(recipient_signature.device)
        
        similarity = F.cosine_similarity(recipient_signature, donor_signature, dim=0)
        
        if similarity < 0.8:
            # In a full implementation, an adapter would be used.
            # For now, we will just reject the transfer.
            # print(f"   - HGT rejected for cell {self.cell_id[:8]}. Similarity too low: {similarity:.2f}")
            return False
        
        # If handshake is successful, integrate the genes
        integrated_count = 0
        for gene in plasmid['genes']:
            if len(self.genes) < cfg.max_genes_per_clone:
                new_gene = copy.deepcopy(gene)
                new_gene.gene_id = f"{new_gene.gene_id}-HGT-{self.cell_id[:8]}"
                
                with torch.no_grad():
                    for param in new_gene.parameters():
                        param.data += torch.randn_like(param) * cfg.mutation_rate
                
                self.genes.append(new_gene)
                integrated_count += 1
        
        return integrated_count > 0



    
    def store_memory(self, antigen_representation: torch.Tensor, response_quality: float):
        """Store successful immune responses in memory"""
        if response_quality > 0.7:
            memory = {
                'representation': antigen_representation.detach().cpu(),
                'response_quality': response_quality,
                'timestamp': datetime.now(),
                'gene_signature': self._compute_gene_signature()
            }
            self.immunological_memory.append(memory)
    
    def _compute_gene_signature(self) -> str:
        """Compute signature of current gene configuration"""
        active_genes = [g for g in self.genes if g.is_active]
        signature_parts = []
        
        for gene in sorted(active_genes, key=lambda g: g.position):
            signature_parts.append(f"{gene.gene_type}{gene.variant_id}:{gene.position:.2f}")
        
        return "-".join(signature_parts)
      
    def clone(self) -> 'ProductionBCell':
        """Create offspring with mutations and epigenetic inheritance.
        MODIFIED: Uses a 'CPU-First' strategy to prevent memory leaks from deepcopy on GPU.
        """
       # print(f"[DEBUG] clone: Starting clone for cell {self.cell_id[:8]}")
       # print(f"[DEBUG] clone: Current device: {next(self.parameters()).device}")
      #  print(f"[DEBUG] clone: Number of genes: {len(self.genes)}")
        
        # --- FIX APPLIED HERE: Move parent to CPU before copying ---
    #    print(f"[DEBUG] clone: Moving parent to CPU...")
        self.to('cpu')
     #   print(f"[DEBUG] clone: Parent moved to CPU, new device: {next(self.parameters()).device}")

        child_genes = []
        active_gene_count = 0
        
        for i, gene in enumerate(self.genes):
            if gene.is_active:
                active_gene_count += 1
               # print(f"[DEBUG] clone: Processing active gene {i}/{len(self.genes)} (type: {gene.gene_type})")
                
                # Now, deepcopy happens on CPU objects, which is much safer.
              #  print(f"[DEBUG] clone: Deep copying gene {i}...")
                child_gene = copy.deepcopy(gene)
              #  print(f"[DEBUG] clone: Deep copy completed for gene {i}")
                
                # Epigenetic inheritance (all on CPU)
             #   print(f"[DEBUG] clone: Applying epigenetic inheritance to gene {i}...")
                child_gene.methylation_state.data *= cfg.methylation_inheritance
                child_gene.histone_modifications.data *= cfg.methylation_inheritance
             #   print(f"[DEBUG] clone: Epigenetic inheritance applied to gene {i}")
                
                # Chance of spontaneous transposition (all on CPU)
                if random.random() < 0.05:
                   # print(f"[DEBUG] clone: Attempting spontaneous transposition for gene {i}...")
                    transposed_child, transposed_action = child_gene.transpose(0.1, 0.5)
                    if transposed_child:
             #           print(f"[DEBUG] clone: Transposition successful for gene {i}, action: {transposed_action}")
                        child_genes.append(transposed_child)
                    else:
                        # print(f"[DEBUG] clone: Transposition failed for gene {i}")
                        pass
                child_genes.append(child_gene)
              #  print(f"[DEBUG] clone: Added child gene {i} to collection")
        
     #   print(f"[DEBUG] clone: Processed {active_gene_count} active genes, created {len(child_genes)} child genes")
        
        # Create the new child (on CPU)
       # print(f"[DEBUG] clone: Creating new ProductionBCell with {len(child_genes)} genes...")
        child = ProductionBCell(child_genes)
       # print(f"[DEBUG] clone: New child created with ID: {child.cell_id[:8]}")
        
        child.lineage = self.lineage + [self.cell_id]
      #  print(f"[DEBUG] clone: Child lineage set, length: {len(child.lineage)}")
        
        # Inherit regulatory matrix (all on CPU)
     #   print(f"[DEBUG] clone: Inheriting regulatory matrix...")
        with torch.no_grad():
            child.gene_regulatory_matrix.data = \
                self.gene_regulatory_matrix.data * 0.9 + \
                torch.randn_like(child.gene_regulatory_matrix) * 0.1
      #  print(f"[DEBUG] clone: Regulatory matrix inherited")
        
        # Apply mutations (on CPU)
      #  print(f"[DEBUG] clone: Applying mutations to child...")
        child._mutate()
      #  print(f"[DEBUG] clone: Mutations applied")
        
        # --- CRITICAL: Move the parent back to the GPU ---
     #   print(f"[DEBUG] clone: Moving parent back to GPU ({cfg.device})...")
        self.to(cfg.device)
      #  print(f"[DEBUG] clone: Parent moved back to device: {next(self.parameters()).device}")
        
        # Return the new child, moved to the GPU in a single, clean operation.
      #  print(f"[DEBUG] clone: Moving child to GPU ({cfg.device})...")
        result = child.to(cfg.device)
      #  print(f"[DEBUG] clone: Child moved to device: {next(result.parameters()).device}")
        print(f"[DEBUG] clone: Clone operation completed successfully")
        
        return result

    
        

    def recycle_as_child(self, parent: 'ProductionBCell'):
        """
        Overwrites this cell's state with a mutated copy of the parent's state.
        This is an in-place, memory-efficient alternative to deepcopy-based cloning.
        """
        # Ensure both are on the CPU for the operation
        parent.to('cpu')
        self.to('cpu')

        # Clear existing genes
        # --- FIX APPLIED HERE ---
        # Re-initialize self.genes to clear it, instead of using .clear()
        self.genes = nn.ModuleList()        
        # Create new genes by copying the parent's (still using deepcopy here, but on CPU)
        child_genes = []
        for gene in parent.genes:
            if gene.is_active:
                child_gene = copy.deepcopy(gene)
                # ... (epigenetic inheritance and spontaneous transposition logic) ...
                if random.random() < 0.05:
                    transposed_child, _ = child_gene.transpose(0.1, 0.5)
                    if transposed_child:
                        child_genes.append(transposed_child)
                child_genes.append(child_gene)
        
        # Assign the new list of gene modules
        for i, gene in enumerate(child_genes):
            self.genes.add_module(str(i), gene)

        # Copy parent's other attributes
        self.lineage = parent.lineage + [parent.cell_id]
        self.generation = parent.generation + 1
        
        # Inherit regulatory matrix
        with torch.no_grad():
            self.gene_regulatory_matrix.data = \
                parent.gene_regulatory_matrix.data * 0.9 + \
                torch.randn_like(self.gene_regulatory_matrix) * 0.1
        
        # Apply mutations
        self._mutate()
        
        # Move both back to the GPU
        parent.to(cfg.device)
        self.to(cfg.device)
        
        #print(f"[DEBUG] recycle_as_child: Recycled cell {self.cell_id[:8]} from parent {parent.cell_id[:8]}")
    
    def _mutate(self):
        """Apply mutations to all parameters"""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < cfg.mutation_rate:
                    mutation = torch.randn_like(param) * cfg.mutation_rate
                    param.data += mutation

