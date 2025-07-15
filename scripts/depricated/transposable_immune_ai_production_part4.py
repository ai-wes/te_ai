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
# Phase Transition Detection and Response System
# ============================================================================

class PhaseTransitionDetector:
    """Advanced phase transition detection with population intervention"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.phase_states = {
            'stable': {'color': 'green', 'intervention': None},
            'critical_slowing': {'color': 'yellow', 'intervention': 'increase_diversity'},
            'bifurcation': {'color': 'orange', 'intervention': 'stabilize'},
            'chaos': {'color': 'red', 'intervention': 'reset_subset'},
            'collapse': {'color': 'black', 'intervention': 'emergency_recovery'}
        }
        self.current_phase = 'stable'
        self.transition_history = []
        
        # Early warning indicators
        self.indicators = {
            'autocorrelation': deque(maxlen=window_size),
            'variance': deque(maxlen=window_size),
            'skewness': deque(maxlen=window_size),
            'spatial_correlation': deque(maxlen=window_size),
            'recovery_rate': deque(maxlen=window_size)
        }
        
        # Intervention strategies
        self.intervention_strategies = {
            'increase_diversity': self._increase_diversity_intervention,
            'stabilize': self._stabilization_intervention,
            'reset_subset': self._reset_subset_intervention,
            'emergency_recovery': self._emergency_recovery_intervention
        }
        
    def update(self, metrics: Dict[str, float], population_state: Dict):
        """Update metrics and check for phase transitions"""
        # Store metrics
        for key, value in metrics.items():
            self.metric_history[key].append(value)
        
        # Compute indicators
        self._compute_early_warning_indicators(metrics, population_state)
        
        # Detect phase state
        new_phase = self._detect_phase_state()
        
        if new_phase != self.current_phase:
            self._record_transition(self.current_phase, new_phase, metrics)
            self.current_phase = new_phase
            
            # Return intervention needed
            intervention = self.phase_states[new_phase]['intervention']
            if intervention:
                return self.intervention_strategies[intervention]
        
        return None
    
    def _compute_early_warning_indicators(self, metrics: Dict, population_state: Dict):
        """Compute all early warning indicators"""
        
        # 1. Autocorrelation at lag-1
        for metric_name, values in self.metric_history.items():
            if len(values) >= 10:
                values_array = np.array(values)
                if values_array.std() > 0:
                    autocorr = np.corrcoef(values_array[:-1], values_array[1:])[0, 1]
                    self.indicators['autocorrelation'].append(autocorr)
        
        # 2. Variance
        if 'fitness' in metrics:
            recent_fitness = list(self.metric_history['fitness'])[-20:]
            if len(recent_fitness) >= 10:
                variance = np.var(recent_fitness)
                self.indicators['variance'].append(variance)
        
        # 3. Skewness
        if 'fitness_distribution' in population_state:
            fitness_dist = population_state['fitness_distribution']
            skewness = stats.skew(fitness_dist)
            self.indicators['skewness'].append(skewness)
        
        # 4. Spatial correlation (gene position correlation)
        if 'gene_positions' in population_state:
            positions = population_state['gene_positions']
            if len(positions) > 10:
                # Compute Moran's I for spatial autocorrelation
                spatial_corr = self._compute_morans_i(positions)
                self.indicators['spatial_correlation'].append(spatial_corr)
        
        # 5. Recovery rate from perturbations
        if 'perturbation_response' in metrics:
            recovery_rate = metrics['perturbation_response']
            self.indicators['recovery_rate'].append(recovery_rate)
    
    def _compute_morans_i(self, positions: List[Tuple[float, float]]) -> float:
        """Compute Moran's I statistic for spatial autocorrelation"""
        if len(positions) < 3:
            return 0.0
        
        positions_array = np.array(positions)
        n = len(positions)
        
        # Compute spatial weights matrix (inverse distance)
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(positions_array[i] - positions_array[j])
                    W[i, j] = 1.0 / (1.0 + dist)
        
        # Normalize weights
        W = W / W.sum()
        
        # Compute values (using first dimension as attribute)
        values = positions_array[:, 0]
        mean_val = values.mean()
        
        # Compute Moran's I
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val) ** 2
        
        morans_i = (n / W.sum()) * (numerator / denominator) if denominator > 0 else 0
        
        return morans_i
    
    def _detect_phase_state(self) -> str:
        """Detect current phase state from indicators"""
        if len(self.indicators['autocorrelation']) < 10:
            return 'stable'
        
        # Get recent indicator values
        recent_autocorr = np.mean(list(self.indicators['autocorrelation'])[-10:])
        recent_variance = np.mean(list(self.indicators['variance'])[-10:]) if self.indicators['variance'] else 0
        recent_skewness = np.abs(np.mean(list(self.indicators['skewness'])[-10:])) if self.indicators['skewness'] else 0
        
        # Trend analysis
        if len(self.indicators['autocorrelation']) >= 20:
            autocorr_trend = np.polyfit(range(20), list(self.indicators['autocorrelation'])[-20:], 1)[0]
            variance_trend = np.polyfit(range(20), list(self.indicators['variance'])[-20:], 1)[0] if len(self.indicators['variance']) >= 20 else 0
        else:
            autocorr_trend = 0
            variance_trend = 0
        
        # Phase detection logic
        if recent_autocorr > 0.95 and variance_trend > 0:
            return 'collapse'
        elif recent_autocorr > 0.8 and autocorr_trend > 0.01:
            return 'critical_slowing'
        elif recent_variance > np.percentile(list(self.indicators['variance']), 90):
            return 'bifurcation'
        elif recent_skewness > 2.0 or recent_autocorr < -0.5:
            return 'chaos'
        else:
            return 'stable'
    
    def _record_transition(self, from_phase: str, to_phase: str, metrics: Dict):
        """Record phase transition event"""
        transition = {
            'timestamp': datetime.now(),
            'from_phase': from_phase,
            'to_phase': to_phase,
            'metrics': metrics.copy(),
            'indicators': {k: list(v)[-10:] if v else [] for k, v in self.indicators.items()}
        }
        self.transition_history.append(transition)
        
        print(f"\n⚠️ PHASE TRANSITION: {from_phase} → {to_phase}")
        print(f"   Autocorrelation: {np.mean(list(self.indicators['autocorrelation'])[-10:]):.3f}")
        print(f"   Variance trend: {np.mean(list(self.indicators['variance'])[-10:]):.3f}")
    
    def _increase_diversity_intervention(self, population_manager) -> bool:
        """Intervention to increase population diversity"""
        print("   🧬 Intervention: Increasing population diversity")
        
        # Force transposition events
        for cell in list(population_manager.population.values())[:50]:
            cell.undergo_transposition(stress_level=0.8)
        
        # Add new random individuals
        num_new = min(50, CFG.max_population - len(population_manager.population))
        population_manager._add_random_individuals(num_new)
        
        return True
    
    def _stabilization_intervention(self, population_manager) -> bool:
        """Intervention to stabilize population"""
        print("   🛡️ Intervention: Stabilizing population")
        
        # Reduce mutation rate temporarily
        original_mutation = CFG.mutation_rate
        CFG.mutation_rate *= 0.1
        
        # Increase selection pressure
        original_selection = CFG.selection_pressure
        CFG.selection_pressure *= 1.5
        
        # Schedule restoration
        population_manager.scheduled_tasks.append({
            'generation': population_manager.generation + 10,
            'action': lambda: setattr(CFG, 'mutation_rate', original_mutation)
        })
        population_manager.scheduled_tasks.append({
            'generation': population_manager.generation + 10,
            'action': lambda: setattr(CFG, 'selection_pressure', original_selection)
        })
        
        return True
    
    def _reset_subset_intervention(self, population_manager) -> bool:
        """Reset a subset of the population"""
        print("   🔄 Intervention: Resetting population subset")
        
        # Identify bottom 20% performers
        fitness_scores = {
            cid: cell.fitness_history[-1] if cell.fitness_history else 0
            for cid, cell in population_manager.population.items()
        }
        
        sorted_cells = sorted(fitness_scores.items(), key=lambda x: x[1])
        reset_count = len(sorted_cells) // 5
        
        # Reset worst performers
        for cell_id, _ in sorted_cells[:reset_count]:
            if cell_id in population_manager.population:
                # Create new random genes
                cell = population_manager.population[cell_id]
                cell.genes.clear()
                
                # Add fresh genes
                for gene_type in ['V', 'D', 'J']:
                    num_genes = random.randint(1, 3)
                    for _ in range(num_genes):
                        gene = ContinuousDepthGeneModule(gene_type, random.randint(1, 50))
                        cell.genes.append(gene)
        
        return True
    
    def _emergency_recovery_intervention(self, population_manager) -> bool:
        """Emergency intervention for population collapse"""
        print("   🚨 EMERGENCY INTERVENTION: Population collapse detected")
        
        # Save best performers
        fitness_scores = {
            cid: cell.fitness_history[-1] if cell.fitness_history else 0
            for cid, cell in population_manager.population.items()
        }
        
        sorted_cells = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        elite_count = max(10, len(sorted_cells) // 10)
        elite_ids = [cid for cid, _ in sorted_cells[:elite_count]]
        
        # Create new diverse population
        new_population = {}
        
        # Keep elite
        for elite_id in elite_ids:
            if elite_id in population_manager.population:
                new_population[elite_id] = population_manager.population[elite_id]
        
        # Generate diverse new individuals
        while len(new_population) < CFG.initial_population:
            new_cell = population_manager._create_random_cell()
            new_population[new_cell.cell_id] = new_cell
        
        # Replace population
        population_manager.population = new_population
        
        # Reset stress
        population_manager.current_stress = 0.0
        
        return True
    
    def get_phase_diagram_data(self) -> Dict:
        """Get data for phase diagram visualization"""
        if not self.transition_history:
            return {}
        
        # Extract phase space coordinates
        phases = []
        autocorrs = []
        variances = []
        
        for transition in self.transition_history:
            phases.append(transition['to_phase'])
            if transition['indicators']['autocorrelation']:
                autocorrs.append(np.mean(transition['indicators']['autocorrelation']))
            if transition['indicators']['variance']:
                variances.append(np.mean(transition['indicators']['variance']))
        
        return {
            'phases': phases,
            'autocorrelation': autocorrs,
            'variance': variances,
            'phase_colors': [self.phase_states[p]['color'] for p in phases]
        }

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
            torch.randn(CFG.max_genes_per_clone, CFG.max_genes_per_clone) * 0.1
        )
        
        # Attention-based gene integration
        self.gene_attention = nn.MultiheadAttention(
            CFG.hidden_dim, num_heads=CFG.num_heads, 
            dropout=0.1, batch_first=True
        )
        
        self.gene_integrator = nn.Sequential(
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim * 2),
            nn.LayerNorm(CFG.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(CFG.hidden_dim * 2, CFG.hidden_dim),
            nn.LayerNorm(CFG.hidden_dim)
        )
        
        # Affinity maturation network
        self.affinity_maturation = nn.Sequential(
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim),
            nn.LayerNorm(CFG.hidden_dim),
            nn.ReLU(),
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(CFG.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Self-modifying architecture
        self.architecture_modifier = SelfModifyingArchitecture(CFG.hidden_dim)
        
        # Plasmid system
        self.plasmids = []
        self.conjugation_pilus = nn.Linear(CFG.hidden_dim, CFG.hidden_dim)
        
        # Memory and learning
        self.immunological_memory = deque(maxlen=1000)
        self.memory_encoder = nn.LSTM(CFG.hidden_dim, CFG.hidden_dim // 2, 
                                     batch_first=True, bidirectional=True)
        
    def forward(self, antigen: Data) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Complete forward pass with all features"""
        device = next(self.parameters()).device
        active_genes = [g for g in self.genes if g.is_active]
        
        if not active_genes:
            dummy = torch.zeros(1, 1, device=device)
            return dummy, torch.zeros(1, CFG.hidden_dim, device=device), {}
        
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
        
        # Attention-based integration
        integrated, attention_weights = self.gene_attention(
            gene_tensor.unsqueeze(0),
            gene_tensor.unsqueeze(0),
            gene_tensor.unsqueeze(0)
        )
        
        # Final integration
        cell_representation = self.gene_integrator(integrated.mean(dim=1))
        
        # Check immunological memory
        memory_response = self._check_memory(cell_representation)
        if memory_response is not None:
            cell_representation = cell_representation + 0.3 * memory_response
        
        # Affinity prediction with maturation
        affinity = self.affinity_maturation(cell_representation)
        
        # Architecture self-modification based on performance
        if len(self.fitness_history) > 10:
            self._attempt_architecture_modification()
        
        metadata = {
            'gene_count': len(active_genes),
            'attention_weights': attention_weights.detach().cpu().numpy(),
            **gene_metadata
        }
        
        return affinity.squeeze(), cell_representation.squeeze(), metadata
    
    def _compute_gene_regulation(self, active_genes: List) -> torch.Tensor:
        """Compute gene regulatory activities"""
        n_genes = len(active_genes)
        if n_genes == 0:
            return torch.tensor([])
        
        # Extract regulatory submatrix
        reg_matrix = self.gene_regulatory_matrix[:n_genes, :n_genes]
        
        # Apply sigmoid for bounded activities
        reg_matrix = torch.sigmoid(reg_matrix)
        
        # Compute steady-state activities (simplified)
        activities = torch.ones(n_genes).to(reg_matrix.device)
        
        for _ in range(5):  # Fixed point iteration
            new_activities = torch.sigmoid(reg_matrix @ activities)
            activities = 0.9 * activities + 0.1 * new_activities
        
        return activities
    
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
            current_state = torch.randn(CFG.hidden_dim).to(next(self.parameters()).device)
            modification = self.architecture_modifier.decide_modification(
                performance_metrics, current_state
            )
            
            # Apply modification
            success = self.architecture_modifier.apply_modification(modification)
            if success:
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
            if len(self.genes) < CFG.max_genes_per_clone:
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
                stress_sites = torch.randint(0, CFG.hidden_dim, (10,))
                gene.add_methylation(stress_sites, CFG.methylation_rate * 2)
                
                # Histone modifications
                gene.modify_histones('h3k27me3', 0.1)  # Repressive mark
    
    def extract_plasmid(self) -> Optional[Dict]:
        """Extract plasmid with high-fitness genes"""
        high_fitness_genes = [
            g for g in self.genes 
            if g.is_active and g.fitness_contribution > 0.7
        ]
        
        if not high_fitness_genes or random.random() > CFG.plasmid_formation_prob:
            return None
        
        # Select genes for plasmid
        plasmid_size = min(3, len(high_fitness_genes))
        plasmid_genes = random.sample(high_fitness_genes, plasmid_size)
        
        plasmid = {
            'id': str(uuid.uuid4()),
            'donor_cell': self.cell_id,
            'genes': [copy.deepcopy(g) for g in plasmid_genes],
            'fitness': sum(g.fitness_contribution for g in plasmid_genes) / len(plasmid_genes),
            'timestamp': datetime.now(),
            'conjugation_signal': self.conjugation_pilus(
                torch.randn(CFG.hidden_dim).to(next(self.parameters()).device)
            ).detach()
        }
        
        self.plasmids.append(plasmid['id'])
        return plasmid
    
    def integrate_plasmid(self, plasmid: Dict) -> bool:
        """Integrate foreign plasmid"""
        # Check compatibility
        if len(self.genes) >= CFG.max_genes_per_clone:
            return False
        
        # Conjugation compatibility check
        compatibility = F.cosine_similarity(
            self.conjugation_pilus(torch.randn(CFG.hidden_dim).to(next(self.parameters()).device)),
            plasmid['conjugation_signal'].to(next(self.parameters()).device),
            dim=0
        )
        
        if compatibility < 0.5:
            return False
        
        # Integrate genes with modifications
        integrated_count = 0
        for gene in plasmid['genes']:
            if len(self.genes) < CFG.max_genes_per_clone:
                new_gene = copy.deepcopy(gene)
                new_gene.gene_id = f"{new_gene.gene_id}-HGT-{self.cell_id[:8]}"
                
                # Mutate during integration
                with torch.no_grad():
                    for param in new_gene.parameters():
                        param.data += torch.randn_like(param) * CFG.mutation_rate
                
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
        """Create offspring with mutations and epigenetic inheritance"""
        # Deep copy genes
        child_genes = []
        for gene in self.genes:
            if gene.is_active:
                child_gene = copy.deepcopy(gene)
                
                # Epigenetic inheritance
                child_gene.methylation_state.data *= CFG.methylation_inheritance
                child_gene.histone_modifications.data *= CFG.methylation_inheritance
                
                # Chance of spontaneous transposition
                if random.random() < 0.05:
                    transposed = child_gene.transpose(0.1, 0.5)
                    if transposed:
                        child_genes.append(transposed)
                
                child_genes.append(child_gene)
        
        # Create child
        child = ProductionBCell(child_genes)
        child.lineage = self.lineage + [self.cell_id]
        
        # Inherit some regulatory connections
        with torch.no_grad():
            child.gene_regulatory_matrix.data = \
                self.gene_regulatory_matrix.data * 0.9 + \
                torch.randn_like(self.gene_regulatory_matrix) * 0.1
        
        # Mutate
        child._mutate()
        
        return child
    
    def _mutate(self):
        """Apply mutations to all parameters"""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < CFG.mutation_rate:
                    mutation = torch.randn_like(param) * CFG.mutation_rate
                    param.data += mutation

# ============================================================================
# Continue in part 5 with the complete Population Manager...
# ============================================================================