"""
Transposable Element Adaptive Immune System AI
==============================================
A production-ready implementation of jumping gene neural networks for immune system modeling.

Features:
- Gene modules that can jump, duplicate, invert, and delete
- Stress-triggered transposition bursts
- Dynamic genome reorganization
- Viral escape simulation
- Real-time visualization of gene arrangements

Requirements:
pip install torch torch_geometric matplotlib networkx seaborn

Run: python transposable_immune_ai.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import copy
import uuid
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class Config:
    """Configuration for the transposable immune system"""
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Neural architecture
    feature_dim: int = 64
    hidden_dim: int = 128
    
    # Transposon parameters
    base_transpose_prob: float = 0.01  # Base probability of transposition
    stress_multiplier: float = 10.0    # How much stress increases transposition
    duplication_cost: float = 0.1      # Fitness penalty for each duplication
    max_genes_per_clone: int = 20      # Maximum genes before forced pruning
    
    # Population dynamics
    initial_population: int = 100
    max_population: int = 2500
    selection_pressure: float = 0.3    # Top 30% reproduce
    mutation_rate: float = 0.01
    
    # Training
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 0.001
    
    # Stress detection
    stress_window: int = 10
    stress_threshold: float = 0.5
    
    # Visualization
    plot_interval: int = 10
    save_dir: str = "transposon_results"

CFG = Config()
os.makedirs(CFG.save_dir, exist_ok=True)

# ============================================================================
# Antigen Generation
# ============================================================================

def generate_antigen_graph(num_nodes: int = 20, mutation_sites: List[int] = None) -> Data:
    """Generate an antigen as a graph with optional mutations"""
    # Create a random geometric graph
    positions = torch.rand(num_nodes, 2)
    distances = torch.cdist(positions, positions)
    adj_matrix = (distances < 0.3).float()
    edge_index = adj_matrix.nonzero().t()
    edge_index = to_undirected(edge_index)
    
    # Node features represent amino acid properties
    features = torch.randn(num_nodes, CFG.feature_dim)
    
    # Apply mutations if specified
    if mutation_sites:
        for site in mutation_sites:
            if site < num_nodes:
                features[site] += torch.randn(CFG.feature_dim) * 2.0
    
    # Binding affinity (ground truth)
    binding_affinity = torch.rand(1).item()
    
    return Data(x=features, edge_index=edge_index, affinity=binding_affinity,
                num_nodes=num_nodes)

# ============================================================================
# Transposable Gene Module
# ============================================================================

class TransposableGene(nn.Module):
    """A gene module that can jump, duplicate, and invert"""
    
    def __init__(self, gene_type: str, variant_id: int):
        super().__init__()
        self.gene_type = gene_type  # 'V', 'D', or 'J'
        self.variant_id = variant_id
        self.gene_id = f"{gene_type}{variant_id}-{uuid.uuid4().hex[:6]}"
        
        # Neural components
        self.conv1 = GCNConv(CFG.feature_dim, CFG.hidden_dim)
        self.conv2 = GCNConv(CFG.hidden_dim, CFG.hidden_dim)
        self.norm1 = nn.LayerNorm(CFG.hidden_dim)
        self.norm2 = nn.LayerNorm(CFG.hidden_dim)
        
        # Transposon properties
        self.position = random.random()  # Position in genome [0, 1]
        self.is_active = True
        self.is_inverted = False
        self.copy_number = 1
        self.fitness_contribution = 0.0
        
        # History tracking
        self.transposition_history = []
        self.parent_gene = None
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process antigen features"""
        # First layer
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        
        # Second layer
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        
        # Inversion flips the representation
        if self.is_inverted:
            h = -h
        
        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)
            
        return h
    
    def transpose(self, stress_level: float) -> Optional['TransposableGene']:
        """Perform transposition based on stress level"""
        transpose_prob = CFG.base_transpose_prob * (1 + stress_level * CFG.stress_multiplier)
        
        if random.random() > transpose_prob:
            return None
        
        # Choose transposition type based on stress
        if stress_level > 0.7:
            # High stress: more likely to duplicate or invert
            weights = [0.2, 0.4, 0.3, 0.1]  # jump, duplicate, invert, delete
        else:
            # Low stress: mostly jumping
            weights = [0.6, 0.2, 0.1, 0.1]
        
        action = random.choices(['jump', 'duplicate', 'invert', 'delete'], 
                               weights=weights)[0]
        
        timestamp = datetime.now().isoformat()
        
        if action == 'jump':
            old_pos = self.position
            self.position = random.random()
            self.transposition_history.append({
                'time': timestamp,
                'action': 'jump',
                'from_position': old_pos,
                'to_position': self.position,
                'stress_level': stress_level
            })
            print(f"  🦘 Gene {self.gene_id} jumped: {old_pos:.3f} → {self.position:.3f}")
            
        elif action == 'duplicate':
            # Create a mutated copy
            new_gene = copy.deepcopy(self)
            new_gene.gene_id = f"{self.gene_type}{self.variant_id}-{uuid.uuid4().hex[:6]}"
            new_gene.parent_gene = self.gene_id
            new_gene.position = self.position + random.uniform(-0.1, 0.1)
            new_gene.position = max(0, min(1, new_gene.position))
            new_gene.copy_number = self.copy_number + 1
            
            # Mutate the duplicate
            with torch.no_grad():
                for param in new_gene.parameters():
                    param.data += torch.randn_like(param) * 0.1
            
            self.transposition_history.append({
                'time': timestamp,
                'action': 'duplicate',
                'child_id': new_gene.gene_id,
                'stress_level': stress_level
            })
            print(f"  🧬 Gene {self.gene_id} duplicated → {new_gene.gene_id}")
            return new_gene
            
        elif action == 'invert':
            self.is_inverted = not self.is_inverted
            self.transposition_history.append({
                'time': timestamp,
                'action': 'invert',
                'inverted_state': self.is_inverted,
                'stress_level': stress_level
            })
            print(f"  🔄 Gene {self.gene_id} inverted: {not self.is_inverted} → {self.is_inverted}")
            
        elif action == 'delete':
            self.is_active = False
            self.transposition_history.append({
                'time': timestamp,
                'action': 'delete',
                'stress_level': stress_level
            })
            print(f"  ❌ Gene {self.gene_id} deleted (silenced)")
            
        return None

# ============================================================================
# B-Cell Clone with Transposable Genome
# ============================================================================

class TransposableBCell(nn.Module):
    """B-cell receptor with dynamic gene arrangement"""
    
    def __init__(self, initial_genes: List[TransposableGene]):
        super().__init__()
        self.cell_id = uuid.uuid4().hex[:8]
        self.genes = nn.ModuleList(initial_genes)
        self.generation = 0
        self.lineage = []
        self.fitness_history = []
        
        # Learned integration of multiple genes
        self.gene_integrator = nn.Sequential(
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(CFG.hidden_dim * 2, CFG.hidden_dim),
            nn.LayerNorm(CFG.hidden_dim)
        )
        
        # Binding affinity predictor
        self.affinity_head = nn.Sequential(
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim),
            nn.ReLU(),
            nn.Linear(CFG.hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, antigen: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process antigen and return binding affinity"""
        active_genes = [g for g in self.genes if g.is_active]
        
        if not active_genes:
            # Dead cell
            device = next(self.parameters()).device
            dummy = torch.zeros(1, CFG.hidden_dim, device=device)
            return self.affinity_head(dummy), dummy
        
        # Sort genes by position for ordered processing
        active_genes.sort(key=lambda g: g.position)
        
        # Process through each gene
        gene_outputs = []
        device = next(self.parameters()).device  # Get the device of this cell
        
        for gene in active_genes:
            output = gene(antigen.x, antigen.edge_index)
            # Ensure output is on the correct device
            output = output.to(device)
            
            # Weight by position (early genes have more influence on binding)
            position_weight = 1.0 - gene.position * 0.3
            gene_outputs.append(output * position_weight)
        
        # Combine gene outputs
        if len(gene_outputs) == 1:
            combined = gene_outputs[0]
        else:
            stacked = torch.cat(gene_outputs, dim=0)
            combined = self.gene_integrator(stacked.mean(dim=0, keepdim=True))
        
        # Predict binding affinity
        affinity = self.affinity_head(combined)
        
        return affinity, combined
    
    def undergo_transposition(self, stress_level: float):
        """Trigger transposition events based on stress"""
        print(f"\n🧫 Cell {self.cell_id} undergoing transposition (stress={stress_level:.2f})")
        
        new_genes = []
        deleted_indices = []
        
        for i, gene in enumerate(self.genes):
            if not gene.is_active:
                continue
                
            result = gene.transpose(stress_level)
            if isinstance(result, TransposableGene):
                new_genes.append(result)
        
        # Add new genes
        for gene in new_genes:
            self.genes.append(gene)
        
        # Prune if too many genes
        if len([g for g in self.genes if g.is_active]) > CFG.max_genes_per_clone:
            self._prune_genes()
        
        self.generation += 1
    
    def _prune_genes(self):
        """Remove least fit genes"""
        active_genes = [(i, g) for i, g in enumerate(self.genes) if g.is_active]
        if len(active_genes) <= CFG.max_genes_per_clone:
            return
        
        # Sort by fitness contribution
        active_genes.sort(key=lambda x: x[1].fitness_contribution)
        
        # Deactivate worst genes
        num_to_remove = len(active_genes) - CFG.max_genes_per_clone
        for i, gene in active_genes[:num_to_remove]:
            gene.is_active = False
            print(f"  🗑️  Pruned low-fitness gene {gene.gene_id}")
    
    def mutate(self):
        """Standard mutation of weights"""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < CFG.mutation_rate:
                    param.data += torch.randn_like(param) * 0.01
    
    def clone(self) -> 'TransposableBCell':
        """Create offspring with mutations"""
        child_genes = []
        for gene in self.genes:
            if gene.is_active:
                child_gene = copy.deepcopy(gene)
                # Small chance of spontaneous transposition during reproduction
                if random.random() < 0.1:
                    result = child_gene.transpose(0.1)
                    if isinstance(result, TransposableGene):
                        child_genes.append(result)
                child_genes.append(child_gene)
        
        child = TransposableBCell(child_genes).to(CFG.device)  # Move to device
        child.lineage = self.lineage + [self.cell_id]
        child.mutate()
        
        return child

# ============================================================================
# Germinal Center Evolution
# ============================================================================

class TransposableGerminalCenter:
    """Population manager with stress-induced transposition"""
    
    def __init__(self):
        self.population: Dict[str, TransposableBCell] = {}
        self.generation = 0
        self.stress_history = deque(maxlen=CFG.stress_window)
        self.current_stress = 0.0
        self.transposition_events = []
        self.fitness_landscape = []
        
        # Initialize population
        self._seed_population()
        
    def _seed_population(self):
        """Create initial naive B-cell population"""
        print(f"Seeding population with {CFG.initial_population} naive B-cells...")
        
        for _ in range(CFG.initial_population):
            # Each cell gets random V, D, J genes
            genes = []
            
            # V genes (1-3 copies)
            num_v = random.randint(1, 3)
            for i in range(num_v):
                v_gene = TransposableGene('V', random.randint(1, 50))
                v_gene.position = random.uniform(0, 0.3)
                genes.append(v_gene)
            
            # D genes (1-2 copies)  
            num_d = random.randint(1, 2)
            for i in range(num_d):
                d_gene = TransposableGene('D', random.randint(1, 30))
                d_gene.position = random.uniform(0.3, 0.6)
                genes.append(d_gene)
            
            # J genes (1-2 copies)
            num_j = random.randint(1, 2)
            for i in range(num_j):
                j_gene = TransposableGene('J', random.randint(1, 6))
                j_gene.position = random.uniform(0.6, 1.0)
                genes.append(j_gene)
            
            cell = TransposableBCell(genes).to(CFG.device)
            self.population[cell.cell_id] = cell

    def compute_population_fitness(self, antigens: List[Data]) -> Dict[str, float]:
        """Evaluate population against antigens"""
        fitness_scores = {}
        
        for cell_id, cell in self.population.items():
            total_fitness = 0.0
            
            for antigen in antigens:
                antigen = antigen.to(CFG.device)
                pred_affinity, _ = cell(antigen)
                true_affinity = antigen.affinity
                
                # Fitness is inverse of prediction error
                error = abs(pred_affinity.item() - true_affinity)
                fitness = 1.0 / (1.0 + error)
                
                # Penalty for too many genes
                num_active = len([g for g in cell.genes if g.is_active])
                complexity_penalty = max(0, num_active - 10) * CFG.duplication_cost
                fitness -= complexity_penalty
                
                total_fitness += fitness
            
            fitness_scores[cell_id] = total_fitness / len(antigens)
            
            # Update gene fitness contributions
            for gene in cell.genes:
                if gene.is_active:
                    gene.fitness_contribution = fitness_scores[cell_id]
        
        return fitness_scores
    
    def detect_stress(self, fitness_scores: Dict[str, float]) -> float:
        """Compute population stress level"""
        mean_fitness = np.mean(list(fitness_scores.values()))
        self.stress_history.append(mean_fitness)
        
        if len(self.stress_history) < CFG.stress_window:
            return 0.0
        
        # Stress increases if:
        # 1. Fitness is declining
        # 2. Fitness variance is high (population struggling)
        fitness_trend = np.array(self.stress_history)
        fitness_decline = (fitness_trend[0] - fitness_trend[-1]) / (fitness_trend[0] + 1e-6)
        fitness_variance = np.std(list(fitness_scores.values()))
        
        stress = max(0, fitness_decline) + fitness_variance
        stress = min(1.0, stress)  # Clamp to [0, 1]
        
        return stress
    
    def evolve(self, antigens: List[Data]):
        """One generation of evolution"""
        import time
        generation_start = time.time()
        
        self.generation += 1
        print(f"\n{'='*60}")
        print(f"Generation {self.generation}")
        print(f"{'='*60}")
        
        # Evaluate fitness
        fitness_start = time.time()
        fitness_scores = self.compute_population_fitness(antigens)
        fitness_time = time.time() - fitness_start
        
        mean_fitness = np.mean(list(fitness_scores.values()))
        print(f"Mean fitness: {mean_fitness:.4f} (computed in {fitness_time:.2f}s)")
        
        # Detect stress
        stress_start = time.time()
        self.current_stress = self.detect_stress(fitness_scores)
        stress_time = time.time() - stress_start
        print(f"Population stress: {self.current_stress:.4f} (computed in {stress_time:.3f}s)")
        
        # High stress triggers transposition
        transposition_time = 0
        if self.current_stress > CFG.stress_threshold:
            print(f"\n⚠️  HIGH STRESS DETECTED! Triggering transposition cascade...")
            transposition_start = time.time()
            self._transposition_phase()
            transposition_time = time.time() - transposition_start
            print(f"Transposition completed in {transposition_time:.2f}s")
        
        # Selection and reproduction
        selection_start = time.time()
        self._selection_phase(fitness_scores)
        selection_time = time.time() - selection_start
        
        generation_time = time.time() - generation_start
        
        print(f"Selection completed in {selection_time:.2f}s")
        print(f"🕒 Total generation time: {generation_time:.2f}s")
        
        # Record fitness landscape with timing info
        self.fitness_landscape.append({
            'generation': self.generation,
            'mean_fitness': mean_fitness,
            'stress_level': self.current_stress,
            'population_size': len(self.population),
            'transposition_events': len(self.transposition_events),
            'generation_time': generation_time,
            'fitness_time': fitness_time,
            'selection_time': selection_time,
            'transposition_time': transposition_time
        })
    
    def _transposition_phase(self):
        """Stress-induced transposition across population"""
        for cell_id, cell in list(self.population.items()):
            cell.undergo_transposition(self.current_stress)
            
            # Track events
            for gene in cell.genes:
                if gene.transposition_history:
                    self.transposition_events.append({
                        'generation': self.generation,
                        'cell_id': cell_id,
                        'gene_id': gene.gene_id,
                        'events': gene.transposition_history[-1]
                    })
    
    def _selection_phase(self, fitness_scores: Dict[str, float]):
        """Natural selection and reproduction"""
        # Sort by fitness
        sorted_cells = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Keep top performers
        num_survivors = int(len(sorted_cells) * (1 - CFG.selection_pressure))
        # Fix: survivors should be a dict of cell_id -> cell, not cell_id -> fitness
        survivor_ids = [cell_id for cell_id, _ in sorted_cells[:num_survivors]]
        survivors = {cell_id: self.population[cell_id] for cell_id in survivor_ids}
        
        # Best cells reproduce
        num_reproducers = int(len(survivors) * 0.5)
        new_cells = {}
        
        for cell_id in survivor_ids[:num_reproducers]:
            parent = self.population[cell_id]
            fitness = fitness_scores[cell_id]
            
            # More fit cells get more offspring
            num_offspring = int(1 + fitness * 3)
            
            for _ in range(num_offspring):
                child = parent.clone()
                new_cells[child.cell_id] = child
        
        # Update population
        self.population = {**survivors, **new_cells}
        
        # Cap population size
        if len(self.population) > CFG.max_population:
            # Sort by fitness, but only keep those in self.population
            sorted_all = sorted(self.population.items(), 
                              key=lambda x: fitness_scores.get(x[0], 0), 
                              reverse=True)
            self.population = dict(sorted_all[:CFG.max_population])
        
        print(f"Population after selection: {len(self.population)} cells")

# ============================================================================
# Visualization
# ============================================================================

def visualize_genome_evolution(center: TransposableGerminalCenter, save_path: str):
    """Visualize gene arrangements across population"""
    n_cells = min(10, len(center.population))
    fig, axes = plt.subplots(n_cells, 1, 
                           figsize=(12, n_cells * 0.8))
    
    if n_cells == 1:
        axes = [axes]
    
    gene_colors = {'V': 'red', 'D': 'green', 'J': 'blue'}
    
    # Only use the first n_cells actual (cell_id, cell) pairs
    cell_items = list(center.population.items())[:n_cells]
    for ax, (cell_id, cell) in zip(axes, cell_items):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        
        # Plot each gene as a rectangle
        for gene in cell.genes:
            if not gene.is_active:
                continue
            
            color = gene_colors.get(gene.gene_type, 'gray')
            alpha = 0.3 if gene.is_inverted else 1.0
            
            rect = plt.Rectangle((gene.position - 0.02, -0.4), 0.04, 0.8,
                               facecolor=color, alpha=alpha, edgecolor='black')
            ax.add_patch(rect)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'Cell {cell_id[:6]} (Gen {cell.generation})', fontsize=10)
        ax.set_yticks([])
        
    axes[-1].set_xlabel('Genomic Position')
    plt.suptitle(f'Transposable Gene Arrangements (Generation {center.generation})', 
                fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_fitness_landscape(center: TransposableGerminalCenter, save_path: str):
    """Plot fitness and stress over time"""
    if not center.fitness_landscape:
        return
    
    data = center.fitness_landscape
    generations = [d['generation'] for d in data]
    fitness = [d['mean_fitness'] for d in data]
    stress = [d['stress_level'] for d in data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Fitness plot
    ax1.plot(generations, fitness, 'b-', linewidth=2)
    ax1.fill_between(generations, fitness, alpha=0.3)
    ax1.set_ylabel('Mean Population Fitness', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Stress plot with transposition events
    ax2.plot(generations, stress, 'r-', linewidth=2)
    ax2.axhline(y=CFG.stress_threshold, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(generations, stress, alpha=0.3, color='red')
    ax2.set_ylabel('Population Stress Level', fontsize=12)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Mark transposition events
    transposition_gens = [e['generation'] for e in center.transposition_events]
    for gen in set(transposition_gens):
        ax2.axvline(x=gen, color='orange', alpha=0.3, linestyle=':')
    
    plt.suptitle('Adaptive Evolution via Transposable Elements', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Main Simulation
# ============================================================================

def simulate_viral_escape():
    """Simulate immune adaptation to evolving virus"""
    import time
    simulation_start = time.time()
    
    print("\n🦠 Starting Viral Escape Simulation with Transposable Elements\n")
    
    # Initialize germinal center
    init_start = time.time()
    center = TransposableGerminalCenter()
    init_time = time.time() - init_start
    print(f"Initialization completed in {init_time:.2f}s\n")
    
    # Define viral mutations that appear over time
    viral_timeline = [
        (0, [], "Wild Type"),
        (10, [5], "Alpha Variant"),
        (20, [5, 12], "Beta Variant"),
        (30, [5, 12, 18], "Delta Variant"),
        (40, [3, 7, 12, 15, 18], "Omicron Variant"),
        (50, [1, 3, 5, 7, 9, 12, 15, 17, 18], "Hypothetical Escape Variant")
    ]
    
    current_variant_idx = 0
    total_evolution_time = 0
    total_visualization_time = 0
    
    # Training loop
    for epoch in range(CFG.epochs):
        # Check if virus mutates
        if current_variant_idx < len(viral_timeline) - 1:
            if epoch >= viral_timeline[current_variant_idx + 1][0]:
                current_variant_idx += 1
                _, mutations, variant_name = viral_timeline[current_variant_idx]
                print(f"\n🚨 VIRUS MUTATED TO {variant_name}! Sites: {mutations}")
                print(f"{'='*60}\n")
                
                # Spike stress to trigger transposition
                center.current_stress = 1.0
        
        # Generate antigens for current variant
        antigen_start = time.time()
        _, mutations, variant_name = viral_timeline[current_variant_idx]
        antigens = []
        for _ in range(CFG.batch_size):
            antigen = generate_antigen_graph(mutation_sites=mutations)
            antigens.append(antigen)
        antigen_time = time.time() - antigen_start
        
        # Evolve population
        evolution_start = time.time()
        center.evolve(antigens)
        evolution_time = time.time() - evolution_start
        total_evolution_time += evolution_time
        
        print(f"Antigen generation: {antigen_time:.3f}s")
        
        # Visualize periodically
        if epoch % CFG.plot_interval == 0:
            viz_start = time.time()
            visualize_genome_evolution(center, 
                                     f"{CFG.save_dir}/genomes_gen_{epoch:03d}.png")
            viz_time = time.time() - viz_start
            total_visualization_time += viz_time
            print(f"Visualization saved in {viz_time:.2f}s")
        
        # Progress update
        progress = (epoch + 1) / CFG.epochs * 100
        avg_gen_time = total_evolution_time / (epoch + 1)
        estimated_remaining = avg_gen_time * (CFG.epochs - epoch - 1)
        
        print(f"📈 Progress: {progress:.1f}% | Avg gen time: {avg_gen_time:.2f}s | ETA: {estimated_remaining/60:.1f}m")
    
    # Final timing summary
    simulation_time = time.time() - simulation_start
    
    # Final plots
    final_viz_start = time.time()
    plot_fitness_landscape(center, f"{CFG.save_dir}/fitness_landscape.png")
    final_viz_time = time.time() - final_viz_start
    
    # Save transposition events
    save_start = time.time()
    with open(f"{CFG.save_dir}/transposition_events.json", 'w') as f:
        json.dump(center.transposition_events, f, indent=2)
    save_time = time.time() - save_start
    
    # Print comprehensive timing summary
    print(f"\n{'='*60}")
    print(f"🏁 SIMULATION COMPLETE - TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Total simulation time: {simulation_time/60:.2f} minutes")
    print(f"Initialization time: {init_time:.2f}s")
    print(f"Total evolution time: {total_evolution_time/60:.2f}m ({total_evolution_time/simulation_time*100:.1f}%)")
    print(f"Total visualization time: {(total_visualization_time + final_viz_time):.2f}s")
    print(f"Data saving time: {save_time:.3f}s")
    print(f"Average time per generation: {total_evolution_time/CFG.epochs:.2f}s")
    
    if center.fitness_landscape:
        # Get timing stats from recorded data
        gen_times = [d.get('generation_time', 0) for d in center.fitness_landscape]
        fitness_times = [d.get('fitness_time', 0) for d in center.fitness_landscape]
        selection_times = [d.get('selection_time', 0) for d in center.fitness_landscape]
        transposition_times = [d.get('transposition_time', 0) for d in center.fitness_landscape]
        
        print(f"\nDetailed timing breakdown:")
        print(f"  Fitness evaluation: {sum(fitness_times):.1f}s total, {np.mean(fitness_times):.3f}s avg")
        print(f"  Selection/reproduction: {sum(selection_times):.1f}s total, {np.mean(selection_times):.3f}s avg")
        print(f"  Transposition: {sum(transposition_times):.1f}s total")
        
        # Find slowest generation
        if gen_times:
            slowest_idx = np.argmax(gen_times)
            print(f"  Slowest generation: #{slowest_idx + 1} ({gen_times[slowest_idx]:.2f}s)")
    
    print(f"\n✅ Results saved to {CFG.save_dir}/")
    print(f"Total transposition events: {len(center.transposition_events)}")
    
    # Analyze successful adaptations
    print("\n📊 Transposition Statistics:")
    event_types = defaultdict(int)
    for event in center.transposition_events:
        event_types[event['events']['action']] += 1
    
    for action, count in event_types.items():
        print(f"  {action}: {count} events")

if __name__ == "__main__":
    simulate_viral_escape()