
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

# ============================================================================
# Learning-Based Dream Consolidation System
# ============================================================================

class DreamMemory:
    """Structured memory storage for dream consolidation"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.priority_queue = []
        self.memory_index = {}
        
    def store(self, state: torch.Tensor, action: str, reward: float, 
              next_state: torch.Tensor, metadata: Dict):
        """Store experience with priority"""
        memory_id = str(uuid.uuid4())
        
        memory = {
            'id': memory_id,
            'state': state.detach().cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.detach().cpu(),
            'metadata': metadata,
            'timestamp': datetime.now(),
            'access_count': 0
        }
        
        self.memories.append(memory)
        self.memory_index[memory_id] = memory
        
        # Priority based on reward magnitude (surprising experiences)
        priority = abs(reward - 0.5)  # Distance from neutral
        self.priority_queue.append((priority, memory_id))
        self.priority_queue.sort(reverse=True)
        
        # Limit priority queue size
        if len(self.priority_queue) > self.capacity // 10:
            self.priority_queue = self.priority_queue[:self.capacity // 10]
    
    def sample_batch(self, batch_size: int, prioritized: bool = True) -> List[Dict]:
        """Sample batch of memories for replay"""
        if len(self.memories) < batch_size:
            return list(self.memories)
        
        if prioritized and self.priority_queue:
            # 50% from priority queue, 50% random
            n_priority = batch_size // 2
            n_random = batch_size - n_priority
            
            priority_ids = [pid for _, pid in self.priority_queue[:n_priority]]
            priority_memories = [self.memory_index.get(pid) for pid in priority_ids 
                               if pid in self.memory_index]
            
            random_memories = random.sample(self.memories, n_random)
            
            batch = priority_memories + random_memories
        else:
            batch = random.sample(self.memories, batch_size)
        
        # Update access counts
        for memory in batch:
            if memory:
                memory['access_count'] += 1
        
        return [m for m in batch if m is not None]

class DreamConsolidationEngine(nn.Module):
    """Complete dream-based learning system"""
    
    def __init__(self, input_dim: int = CFG.hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Dream generation network (VAE-style)
        self.dream_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2)
        )
        
        # Reparameterization for VAE
        self.mu_layer = nn.Linear(input_dim * 2, input_dim)
        self.logvar_layer = nn.Linear(input_dim * 2, input_dim)
        
        # Dream decoder
        self.dream_decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Nightmare generator (adversarial component)
        self.nightmare_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # Dream critic (evaluates dream quality)
        self.dream_critic = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )
        
        # Memory systems
        self.episodic_memory = DreamMemory(capacity=10000)
        self.semantic_memory = DreamMemory(capacity=5000)
        
        # Consolidation networks
        self.consolidation_gru = nn.GRU(
            input_dim, input_dim, 
            num_layers=3, batch_first=True, dropout=0.1
        )
        
        self.consolidation_attention = nn.MultiheadAttention(
            input_dim, num_heads=8, batch_first=True
        )
        
        # Meta-learning components
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh()
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate_dream_batch(self, num_dreams: int) -> Tuple[torch.Tensor, Dict]:
        """Generate batch of dream experiences"""
        # Sample from episodic memory
        memories = self.episodic_memory.sample_batch(num_dreams * 2)
        
        if len(memories) < 2:
            return None, {}
        
        dream_states = []
        dream_metadata = {
            'vae_loss': [],
            'diversity': [],
            'novelty': []
        }
        
        for i in range(num_dreams):
            # Encode memory
            memory = random.choice(memories)
            state = memory['state'].to(CFG.device).unsqueeze(0)
            
            encoded = self.dream_encoder(state)
            mu = self.mu_layer(encoded)
            logvar = self.logvar_layer(encoded)
            
            # Generate dream variation
            z = self.reparameterize(mu, logvar)
            dream_state = self.dream_decoder(z)
            
            # VAE loss for quality monitoring
            recon_loss = F.mse_loss(dream_state, state)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + 0.01 * kl_loss
            
            dream_metadata['vae_loss'].append(vae_loss.item())
            
            # Add controlled noise for diversity
            if i % 3 == 0:  # Every third dream is a nightmare
                nightmare = self.nightmare_generator(dream_state)
                dream_state = dream_state + CFG.nightmare_adversarial_strength * nightmare
            
            dream_states.append(dream_state)
        
        if dream_states:
            dream_batch = torch.cat(dream_states, dim=0)
            
            # Compute diversity metrics
            dream_numpy = dream_batch.detach().cpu().numpy()
            pairwise_distances = np.linalg.norm(
                dream_numpy[:, np.newaxis] - dream_numpy[np.newaxis, :], 
                axis=2
            )
            dream_metadata['diversity'] = pairwise_distances.mean()
            
            # Compute novelty vs memories
            memory_states = torch.stack([m['state'] for m in memories[:num_dreams]]).to(CFG.device)
            novelty = torch.norm(dream_batch - memory_states, dim=1).mean()
            dream_metadata['novelty'] = novelty.item()
            
            return dream_batch, dream_metadata
        
        return None, dream_metadata
    
    def consolidate_learning(self, dream_batch: torch.Tensor, 
                           gene_states: List[torch.Tensor]) -> torch.Tensor:
        """Consolidate dream experiences into improved parameters"""
        if len(gene_states) == 0:
            return None
        
        # Stack gene states
        gene_tensor = torch.stack(gene_states).to(CFG.device)
        
        # Process dreams through GRU
        dream_hidden, _ = self.consolidation_gru(dream_batch.unsqueeze(0))
        dream_repr = dream_hidden.mean(dim=1)
        
        # Attention between dreams and current genes
        attended, attention_weights = self.consolidation_attention(
            gene_tensor.unsqueeze(0),
            dream_batch.unsqueeze(0),
            dream_batch.unsqueeze(0)
        )
        
        # Meta-learning: learn how to learn from dreams
        combined = torch.cat([
            gene_tensor.mean(dim=0),
            dream_repr.squeeze(0),
            attended.squeeze(0).mean(dim=0)
        ])
        
        meta_update = self.meta_learner(combined)
        
        return meta_update, attention_weights
    
    def evaluate_dream_quality(self, dream_batch: torch.Tensor, 
                             real_batch: torch.Tensor) -> float:
        """Evaluate quality and usefulness of dreams"""
        combined = torch.cat([dream_batch, real_batch], dim=1)
        quality_scores = self.dream_critic(combined)
        return quality_scores.mean().item()
    
    def dream_phase(self, population: Dict[str, Any], num_cycles: int = 5):
        """Complete dream consolidation phase"""
        print(f"\n💤 Dream Consolidation Phase ({num_cycles} cycles)")
        
        for cycle in range(num_cycles):
            cycle_start = time.time()
            
            # Generate dreams
            dream_batch, dream_meta = self.generate_dream_batch(
                CFG.memory_replay_batch_size
            )
            
            if dream_batch is None:
                continue
            
            # Process each cell
            consolidation_count = 0
            total_improvement = 0.0
            
            for cell_id, cell in list(population.items())[:100]:  # Limit for efficiency
                if not hasattr(cell, 'genes'):
                    continue
                
                # Extract gene states
                gene_states = []
                for gene in cell.genes:
                    if gene.is_active and hasattr(gene, 'output_projection'):
                        with torch.no_grad():
                            # Get representative state
                            weight = gene.output_projection[0].weight.data
                            state = weight.mean(dim=0)
                            gene_states.append(state)
                
                if len(gene_states) < 2:
                    continue
                
                # Consolidate learning
                meta_update, attention = self.consolidate_learning(
                    dream_batch, gene_states
                )
                
                if meta_update is not None:
                    # Apply consolidated learning to genes
                    with torch.no_grad():
                        for i, gene in enumerate(cell.genes):
                            if gene.is_active and i < len(gene_states):
                                # Selective update based on attention
                                if attention is not None and i < attention.shape[-1]:
                                    update_strength = attention[0, i, :].mean().item()
                                else:
                                    update_strength = 0.1
                                
                                # Apply meta-learned update
                                update_strength *= (1.0 - gene.chromatin_accessibility)
                                
                                for param in gene.parameters():
                                    param.data += update_strength * torch.randn_like(param) * \
                                                meta_update.norm().item() * 0.01
                    
                    consolidation_count += 1
                    total_improvement += meta_update.norm().item()
            
            # Log cycle results
            cycle_time = time.time() - cycle_start
            print(f"  Cycle {cycle+1}: {consolidation_count} cells consolidated, "
                  f"avg improvement: {total_improvement/max(consolidation_count, 1):.4f}, "
                  f"time: {cycle_time:.2f}s")
            
            if dream_meta:
                print(f"    Dream quality - VAE loss: {np.mean(dream_meta['vae_loss']):.4f}, "
                      f"diversity: {dream_meta['diversity']:.4f}, "
                      f"novelty: {dream_meta.get('novelty', 0):.4f}")

# ============================================================================
# Fully Parallel GPU Population Processing
# ============================================================================

class ParallelCellBatch(nn.Module):
    """Wrapper for processing multiple cells in parallel on GPU"""
    
    def __init__(self, cells: List[nn.Module]):
        super().__init__()
        self.num_cells = len(cells)
        
        # Create shared parameter groups for efficiency
        self.gene_modules = nn.ModuleList()
        self.gene_mapping = {}  # Maps (cell_idx, gene_idx) to module index
        
        module_idx = 0
        for cell_idx, cell in enumerate(cells):
            for gene_idx, gene in enumerate(cell.genes):
                if gene.is_active:
                    self.gene_modules.append(gene)
                    self.gene_mapping[(cell_idx, gene_idx)] = module_idx
                    module_idx += 1
        
        # Shared attention and integration layers
        self.shared_attention = nn.MultiheadAttention(
            CFG.hidden_dim, num_heads=CFG.num_heads, batch_first=True
        )
        
        self.shared_integrator = nn.Sequential(
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(CFG.hidden_dim * 2, CFG.hidden_dim),
            nn.LayerNorm(CFG.hidden_dim)
        )
        
        # Affinity prediction heads (one per cell)
        self.affinity_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(CFG.hidden_dim, CFG.hidden_dim),
                nn.ReLU(),
                nn.Linear(CFG.hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(self.num_cells)
        ])
    
    def forward(self, antigen_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process all cells on antigen batch in parallel"""
        batch_size = antigen_batch.num_graphs
        device = antigen_batch.x.device
        
        # Process all genes in parallel
        all_gene_outputs = []
        gene_cell_mapping = []
        
        for (cell_idx, gene_idx), module_idx in self.gene_mapping.items():
            gene = self.gene_modules[module_idx]
            
            # Process gene on entire antigen batch
            gene_output = gene(antigen_batch.x, antigen_batch.edge_index, 
                             antigen_batch.batch)
            
            all_gene_outputs.append(gene_output)
            gene_cell_mapping.extend([cell_idx] * gene_output.shape[0])
        
        if not all_gene_outputs:
            # No active genes
            dummy = torch.zeros(self.num_cells, batch_size, 1, device=device)
            return dummy.squeeze(-1), dummy.squeeze(-1)
        
        # Stack all gene outputs
        all_gene_tensor = torch.cat(all_gene_outputs, dim=0)
        
        # Group by cell using scatter operations
        cell_outputs = []
        cell_hidden = []
        
        for cell_idx in range(self.num_cells):
            # Get indices for this cell's genes
            cell_mask = torch.tensor(gene_cell_mapping) == cell_idx
            cell_genes = all_gene_tensor[cell_mask]
            
            if cell_genes.shape[0] == 0:
                # No genes for this cell
                cell_outputs.append(torch.zeros(batch_size, 1, device=device))
                cell_hidden.append(torch.zeros(batch_size, CFG.hidden_dim, device=device))
                continue
            
            # Apply attention across genes
            if cell_genes.shape[0] > 1:
                attended, _ = self.shared_attention(
                    cell_genes.unsqueeze(0),
                    cell_genes.unsqueeze(0),
                    cell_genes.unsqueeze(0)
                )
                integrated = self.shared_integrator(attended.mean(dim=1))
            else:
                integrated = self.shared_integrator(cell_genes)
            
            # Predict affinity
            affinity = self.affinity_heads[cell_idx](integrated)
            
            cell_outputs.append(affinity.squeeze(-1))
            cell_hidden.append(integrated.squeeze(0) if integrated.dim() > 2 else integrated)
        
        # Stack results
        affinities = torch.stack(cell_outputs, dim=0)  # [num_cells, batch_size]
        hiddens = torch.stack(cell_hidden, dim=0)      # [num_cells, batch_size, hidden_dim]
        
        return affinities, hiddens

def create_parallel_batch(population: Dict[str, nn.Module], 
                         batch_indices: List[str]) -> ParallelCellBatch:
    """Create parallel batch from population subset"""
    cells = [population[idx] for idx in batch_indices if idx in population]
    return ParallelCellBatch(cells).to(CFG.device)

# ============================================================================
# Continue in part 3...
# ============================================================================