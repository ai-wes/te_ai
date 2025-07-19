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
from scripts.core.advanced.global_workspace import GlobalWorkspace
import inspect
from scripts.config import cfg
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.core.self_modifying_neural_architecture import SelfModifyingArchitecture
from scripts.core.utils.detailed_logger import get_logger, trace
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.utils.telemetry import TermColors
# Removed circular import - StemGeneModule will be imported dynamically when needed
# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



from scripts.core.utils.detailed_logger import get_logger, trace

logger = get_logger()





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
            dropout=cfg.attention_dropout, batch_first=True
        )
        
        self.gene_integrator = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout for regularization
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(0.2)  # Additional dropout layer
        )
        
        # Affinity maturation network
        self.affinity_maturation = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout before final layer
            nn.Linear(cfg.hidden_dim // 2, 1)
            # Removed Sigmoid - will apply in prediction for better numerical stability
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
        
    
    def forward(self, antigen: Data, global_workspace: Optional[GlobalWorkspace] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Complete forward pass with all features"""
        # Validate all genes are on correct device before processing
        self._validate_device_consistency()
        
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
                # --- START OF FIX ---
        for i, (gene, activity) in enumerate(zip(active_genes, gene_activities)):
            # Inspect the gene's forward method signature
            forward_params = inspect.signature(gene.forward).parameters
            
            # Prepare arguments based on the signature
            call_args = {
                'x': antigen.x,
                'edge_index': antigen.edge_index
            }
            if 'batch' in forward_params:
                call_args['batch'] = antigen.batch
            if 'global_features' in forward_params:
                # You might want to pass some global features here in the future
                call_args['global_features'] = None

            # Process antigen through gene using the prepared arguments
            gene_output = gene(**call_args)
            
            # The specialized genes return a tuple (output, metadata)
            # The base gene returns only the output tensor. We need to handle both cases.
            if isinstance(gene_output, tuple):
                gene_output, metadata_from_gene = gene_output
                gene_metadata.update(metadata_from_gene)
            # --- END OF FIX ---

            
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
            # memory_response has same shape as cell_representation
            cell_representation = cell_representation + 0.3 * memory_response
        
        # Affinity prediction with maturation
        affinity = self.affinity_maturation(cell_representation)
        
        # *** NEW: Broadcast to Global Workspace ***
        if self.training and global_workspace is not None:
            # Create a hash of the input data to identify the challenge
            antigen_hash = hashlib.sha256(antigen.x.cpu().numpy().tobytes()).hexdigest()
            self.broadcast_to_workspace(global_workspace, antigen_hash, affinity, cell_representation)
        
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
        # Get device from the cell, not individual genes
        device = next(self.parameters()).device
        reg_matrix = torch.randn(n, n, device=device)
        
        # Get gene activities, ensuring all are on the same device
        activities = []
        for g in active_genes:
            activity = getattr(g, 'activity', torch.tensor(1.0))
            if not isinstance(activity, torch.Tensor):
                activity = torch.tensor(activity)
            activities.append(activity.to(device))
        
        activities = torch.stack(activities)
        
        # Apply regulation
        regulated = torch.sigmoid(reg_matrix @ activities)
        
        return regulated
    
    
    def _check_memory(self, representation: torch.Tensor) -> Optional[torch.Tensor]:
        """Check immunological memory for similar antigens"""
        if len(self.immunological_memory) < 10:
            return None
        
        # Get device from representation
        device = representation.device
        
        # Encode memories and ensure they're on the correct device
        memory_list = []
        for m in list(self.immunological_memory)[-50:]:
            mem_tensor = m['representation']
            if mem_tensor.device != device:
                mem_tensor = mem_tensor.to(device)
            memory_list.append(mem_tensor)
        
        memory_tensors = torch.stack(memory_list)
        
        # Ensure tensor dimensions match
        if representation.dim() == 1:
            representation = representation.unsqueeze(0)
        if memory_tensors.dim() == 1:
            memory_tensors = memory_tensors.unsqueeze(0)
            
        # Handle dimension mismatch
        if representation.size(-1) != memory_tensors.size(-1):
            # Project both to common dimension using adaptive pooling
            common_dim = min(representation.size(-1), memory_tensors.size(-1))
            if representation.size(-1) != common_dim:
                # Use adaptive pooling to resize
                representation = F.adaptive_avg_pool1d(
                    representation.unsqueeze(1), common_dim
                ).squeeze(1)
            if memory_tensors.size(-1) != common_dim:
                memory_tensors = F.adaptive_avg_pool1d(
                    memory_tensors.transpose(0, 1).unsqueeze(0), common_dim
                ).squeeze(0).transpose(0, 1)
        
        # Compute similarity - handle batch dimension properly
        if representation.dim() == 2 and memory_tensors.dim() == 2:
            # representation: [B, D], memory_tensors: [N, D]
            # We need to compute similarity for each item in batch
            batch_size = representation.size(0)
            n_memories = memory_tensors.size(0)
            
            # Expand for broadcasting: [B, 1, D] and [1, N, D]
            rep_expanded = representation.unsqueeze(1)  # [B, 1, D]
            mem_expanded = memory_tensors.unsqueeze(0)  # [1, N, D]
            
            # Compute cosine similarity for all pairs
            similarities = F.cosine_similarity(rep_expanded, mem_expanded, dim=2)  # [B, N]
            
            # Find best match for each item in batch
            max_similarities, max_indices = similarities.max(dim=1)  # [B]
            
            # Return memory response for items with high similarity
            memory_responses = []
            for b in range(batch_size):
                if max_similarities[b] > 0.8:
                    memory_responses.append(memory_tensors[max_indices[b]])
                else:
                    memory_responses.append(None)
            
            # If any memory responses found, stack them
            if any(r is not None for r in memory_responses):
                # Create a tensor with zeros for items without memory response
                result = torch.zeros_like(representation)
                for i, resp in enumerate(memory_responses):
                    if resp is not None:
                        result[i] = resp
                return result
            else:
                return None
        else:
            # Fallback for unexpected dimensions
            return None
    
    
    def _attempt_architecture_modification(self):
        logger.debug("Entering ProductionBCell._attempt_architecture_modification")
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
                modification.performance_delta = performance_metrics['trend']
    
    
    def undergo_transposition(self, stress_level: float, diversity: float = 0.5):
        logger.debug("Entering ProductionBCell.undergo_transposition")
        """Stress-induced transposition with population awareness"""
        new_genes = []
        
        for gene in list(self.genes):
            if gene.is_active:
                child = gene.transpose(stress_level, diversity)
                if child:
                    new_genes.append(child)
        
        # Add new genes
        device = next(self.parameters()).device
        for gene in new_genes:
            if len(self.genes) < cfg.max_genes_per_clone:
                # Ensure gene is on the same device as the cell
                gene = gene.to(device)
                self.genes.append(gene)
        
        # Update generation
        self.generation += 1
        
        # Epigenetic inheritance
        if stress_level > 0.7:
            self._apply_stress_epigenetics()
    
    
    def _apply_stress_epigenetics(self):
        logger.debug("Entering ProductionBCell._apply_stress_epigenetics")
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
        OPTIMIZED: Uses state_dict copying, stays on GPU.
        """
        from scripts.core.stem_gene_module import StemGeneModule

        device = next(self.parameters()).device
        high_fitness_genes = [
            g for g in self.genes 
            if g.is_active and g.fitness_contribution > 0.7
        ]
        
        if not high_fitness_genes:
            return None
        
        # Select genes for plasmid
        plasmid_size = min(3, len(high_fitness_genes))
        selected_genes = random.sample(high_fitness_genes, plasmid_size)
        
        # Clone genes efficiently and ensure they stay on the correct device
        plasmid_genes = []
        for gene in selected_genes:
            try:
                # Use clone method if available, otherwise fall back to deepcopy
                if hasattr(gene, 'clone'):
                    new_gene = gene.clone()
                else:
                    # Fallback for genes without clone method
                    new_gene = copy.deepcopy(gene)
                
                # CRITICAL: Ensure the cloned gene is on the same device
                new_gene = new_gene.to(device)
                plasmid_genes.append(new_gene)
            except Exception as e:
                logger.warning(f"Could not clone gene of type {type(gene).__name__}: {e}")
                continue
        
        # Generate conjugation signal on correct device
        conjugation_signal = self.conjugation_pilus(
            torch.randn(cfg.hidden_dim, device=device)
        ).detach()

        plasmid = {
            'id': str(uuid.uuid4()),
            'donor_cell': self.cell_id,
            'genes': plasmid_genes,
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
            # No device transfers - assume everything is on the correct device
            _, cell_representation, _ = self.forward(calibration_batch)
            self._signature_cache = cell_representation.mean(dim=0).detach()
            
        return self._signature_cache



    
    def attempt_entanglement(self):
        logger.debug("Entering ProductionBCell.attempt_entanglement")
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
                
                # Ensure new gene is on the same device as the cell
                device = next(self.parameters()).device
                new_gene = new_gene.to(device)
                
                with torch.no_grad():
                    for param in new_gene.parameters():
                        param.data += torch.randn_like(param) * cfg.mutation_rate
                
                self.genes.append(new_gene)
                integrated_count += 1
        
        return integrated_count > 0



    
    
    def store_memory(self, antigen_representation: torch.Tensor, response_quality: float):
        logger.debug("Entering ProductionBCell.store_memory")
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
    
    def _validate_device_consistency(self):
        """Ensure all genes are on the same device as the cell"""
        cell_device = next(self.parameters()).device
        for i, gene in enumerate(self.genes):
            gene_device = next(gene.parameters()).device
            if gene_device != cell_device:
                logger.warning(f"Gene {i} ({gene.gene_id}) on {gene_device}, moving to {cell_device}")
                self.genes[i] = gene.to(cell_device)
      
    
    def clone(self) -> 'ProductionBCell':
        """Create offspring with mutations and epigenetic inheritance.
        OPTIMIZED: Uses state_dict copying instead of deepcopy, stays on GPU.
        """
        # *** CRITICAL FIX: Determine the device from the parent cell itself ***
        device = next(self.parameters()).device
        
        child_genes = []
        for gene in self.genes:
            if gene.is_active:
                # Use the gene's own clone method if it exists, as it's the most reliable
                if hasattr(gene, 'clone'):
                    child_gene = gene.clone()
                else:
                    # Fallback to deepcopy for genes without a custom clone method
                    child_gene = copy.deepcopy(gene)
                
                # *** CRITICAL FIX: Ensure the new gene is on the correct device ***
                child_genes.append(child_gene.to(device))
        
        # Create the new child directly on the correct device
        child = ProductionBCell(child_genes).to(device)
        
        # Inherit attributes
        child.lineage = self.lineage + [self.cell_id]
        
        # Inherit regulatory matrix with noise
        with torch.no_grad():
            child.gene_regulatory_matrix.data = \
                self.gene_regulatory_matrix.data * 0.9 + \
                torch.randn_like(child.gene_regulatory_matrix, device=device) * 0.1
        
        # Apply mutations
        child._mutate()
        
        return child

    
    
    def recycle_as_child(self, parent: 'ProductionBCell'):
        logger.debug("Entering ProductionBCell.recycle_as_child")
        """
        Overwrites this cell's state with a mutated copy of the parent's state.
        OPTIMIZED: In-place operation without CPU transfers.
        """
        device = next(self.parameters()).device
        
        # Clear existing genes
        self.genes = nn.ModuleList()
        
        # Create new genes using state_dict copying
        child_genes = []
        for gene in parent.genes:
            if gene.is_active:
                # Create new gene instance with required arguments
                gene_class = type(gene)
                
                # Get required arguments from the original gene
                if hasattr(gene, 'gene_type') and hasattr(gene, 'variant_id'):
                    child_gene = gene_class(
                        gene_type=gene.gene_type,
                        variant_id=gene.variant_id
                    )
                elif hasattr(gene, 'variant_id'):
                    # For genes that only need variant_id
                    child_gene = gene_class(variant_id=gene.variant_id)
                else:
                    # Fallback for genes with no required arguments
                    child_gene = gene_class()
                
                # Copy state efficiently
                try:
                    child_gene.load_state_dict(gene.state_dict())
                except RuntimeError as e:
                    # If state dict doesn't match, skip copying and continue
                    logger.warning(f"State dict mismatch for {gene_class.__name__}: {e}")
                    # Continue with the new gene without copying state
                
                # Copy non-parameter attributes
                for attr in ['gene_id', 'gene_type', 'variant_id', 'position', 'is_active', 
                            'fitness_contribution']:
                    if hasattr(gene, attr):
                        value = getattr(gene, attr)
                        if isinstance(value, torch.Tensor):
                            setattr(child_gene, attr, value.clone())
                        else:
                            setattr(child_gene, attr, copy.copy(value))
                
                # Handle parameter attributes separately
                if hasattr(gene, 'methylation_state') and hasattr(child_gene, 'methylation_state'):
                    if isinstance(gene.methylation_state, nn.Parameter):
                        child_gene.methylation_state.data.copy_(gene.methylation_state.data)
                    else:
                        setattr(child_gene, 'methylation_state', gene.methylation_state.clone())
                        
                if hasattr(gene, 'histone_modifications') and hasattr(child_gene, 'histone_modifications'):
                    if isinstance(gene.histone_modifications, nn.Parameter):
                        child_gene.histone_modifications.data.copy_(gene.histone_modifications.data)
                    else:
                        setattr(child_gene, 'histone_modifications', gene.histone_modifications.clone())
                
                # Epigenetic inheritance
                if hasattr(child_gene, 'methylation_state'):
                    child_gene.methylation_state.data *= cfg.methylation_inheritance
                if hasattr(child_gene, 'histone_modifications'):
                    child_gene.histone_modifications.data *= cfg.methylation_inheritance
                
                # Spontaneous transposition
                if random.random() < 0.05:
                    transposed_child, _ = child_gene.transpose(0.1, 0.5)
                    if transposed_child:
                        child_genes.append(transposed_child)
                child_genes.append(child_gene)
        
        # Assign the new list of gene modules
        for i, gene in enumerate(child_genes):
            self.genes.add_module(str(i), gene.to(device))

        # Copy parent's other attributes
        self.lineage = parent.lineage + [parent.cell_id]
        self.generation = parent.generation + 1
        
        # Inherit regulatory matrix with noise
        with torch.no_grad():
            # Use pre-allocated tensor pool for efficiency
            from scripts.core.tensor_pool import get_pooled_random
            noise = get_pooled_random(self.gene_regulatory_matrix.shape, 
                                    dtype=self.gene_regulatory_matrix.dtype)
            self.gene_regulatory_matrix.data = \
                parent.gene_regulatory_matrix.data * 0.9 + noise * 0.1
        
        # Apply mutations
        self._mutate()
    
    
    def _mutate(self):
        logger.debug("Entering ProductionBCell._mutate")
        """Apply mutations to all parameters"""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < cfg.mutation_rate:
                    mutation = torch.randn_like(param) * cfg.mutation_rate
                    param.data += mutation



    def broadcast_to_workspace(self, global_workspace, antigen_hash, affinity, representation):
        """Broadcasts cell state to the global workspace based on confidence."""
        if global_workspace is None:
            return

        affinity_val = affinity.item()
        
        # Check for confusion (low confidence)
        if 0.45 < affinity_val < 0.55:
            # *** ADDED LOGGING ***
            logger.info(f"   🧠 Cell {self.cell_id[:8]} is CONFUSED (affinity {affinity_val:.3f}), broadcasting to Workspace.")
            global_workspace.broadcast(
                antigen_hash=antigen_hash,
                cell_id=self.cell_id,
                message_type='CONFUSED',
                state_tensor=representation
            )
        # Check for high confidence on a potentially difficult problem
        elif affinity_val > 0.95 or affinity_val < 0.05:
            challenge_state = global_workspace.query(antigen_hash)
            if len(challenge_state['confused_states']) > 0:
                # *** ADDED LOGGING ***
                logger.info(f"   🧠 Cell {self.cell_id[:8]} is CONFIDENT (affinity {affinity_val:.3f}) on a difficult problem, broadcasting to Workspace.")
                global_workspace.broadcast(
                    antigen_hash=antigen_hash,
                    cell_id=self.cell_id,
                    message_type='CONFIDENT',
                    state_tensor=representation
                )