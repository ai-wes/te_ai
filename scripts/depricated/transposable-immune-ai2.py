"""
Transposable Element Adaptive Immune System AI - FIXED VERSION
============================================================================
Fixed issues with CUDA streams and torch.compile compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean
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

import time

# --- SPEED INIT: cuDNN/TF32/AMP/Seed (Section 1) ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")


def enable_cuda_fast_math(seed=42):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
enable_cuda_fast_math()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
cuda = torch.cuda

USE_AMP = True
# Disable torch.compile due to incompatibility with scatter operations
USE_COMPILE = False  # Changed from torch.__version__ >= "2.0.0"
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from contextlib import nullcontext
from functools import partial
from torch import amp  # Updated import (removes deprecation warning)
amp_ctx = partial(amp.autocast, 'cuda') if USE_AMP else nullcontext

print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim: int = 128
    hidden_dim: int = 512
    num_gcn_layers: int = 4
    eval_antigens_subset: int = 128
    base_transpose_prob: float = 0.01
    stress_multiplier: float = 10.0
    duplication_cost: float = 0.1
    max_genes_per_clone: int = 50
    initial_population: int = 100
    max_population: int = 5000
    selection_pressure: float = 0.3
    mutation_rate: float = 0.01
    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 0.001
    stress_window: int = 10
    stress_threshold: float = 0.5
    plot_interval: int = 10
    save_dir: str = "transposon_results"
    verbose_transposition: bool = True
    # Removed num_streams as we'll use simpler approach
    batch_cells: int = 32  # Process cells in smaller batches

CFG = Config()
os.makedirs(CFG.save_dir, exist_ok=True)

# ============================================================================
# Antigen Generation (larger graphs)
# ============================================================================

def generate_antigen_graph(num_nodes: int = 256, mutation_sites: List[int] = None) -> Data:
    positions = torch.rand(num_nodes, 2)
    distances = torch.cdist(positions, positions)
    adj_matrix = (distances < 0.3).float()
    edge_index = adj_matrix.nonzero().t()
    edge_index = to_undirected(edge_index)
    features = torch.randn(num_nodes, CFG.feature_dim)
    if mutation_sites:
        for site in mutation_sites:
            if site < num_nodes:
                features[site] += torch.randn(CFG.feature_dim) * 2.0
    binding_affinity = torch.rand(1).item()
    return Data(x=features, edge_index=edge_index, affinity=binding_affinity, num_nodes=num_nodes)

# ============================================================================
# Transposable Gene Module (deeper)
# ============================================================================

class TransposableGene(nn.Module):
    def __init__(self, gene_type: str, variant_id: int):
        super().__init__()
        self.gene_type = gene_type
        self.variant_id = variant_id
        self.gene_id = f"{gene_type}{variant_id}-{uuid.uuid4().hex[:6]}"
        self.convs = nn.ModuleList([GCNConv(CFG.feature_dim if i == 0 else CFG.hidden_dim, CFG.hidden_dim) for i in range(CFG.num_gcn_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(CFG.hidden_dim) for _ in range(CFG.num_gcn_layers)])
        self.position = random.random()
        self.is_active = True
        self.is_inverted = False
        self.copy_number = 1
        self.fitness_contribution = 0.0
        self.transposition_history = []
        self.parent_gene = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
        if self.is_inverted:
            h = -h
        if batch is not None:
            h = scatter_mean(h, batch, dim=0)
        else:
            h = h.mean(dim=0, keepdim=True)
        return h

    def transpose(self, stress_level: float) -> Optional['TransposableGene']:
        transpose_prob = CFG.base_transpose_prob * (1 + stress_level * CFG.stress_multiplier)
        if random.random() > transpose_prob:
            return None
        if stress_level > 0.7:
            weights = [0.2, 0.4, 0.3, 0.1]
        else:
            weights = [0.6, 0.2, 0.1, 0.1]
        action = random.choices(['jump', 'duplicate', 'invert', 'delete'], weights=weights)[0]
        timestamp = datetime.now().isoformat()
        if action == 'jump':
            old_pos = self.position
            self.position = random.random()
            self.transposition_history.append({
                'time': timestamp, 'action': 'jump',
                'from_position': old_pos, 'to_position': self.position,
                'stress_level': stress_level
            })
            if CFG.verbose_transposition:
                print(f"  🦘 Gene {self.gene_id} jumped: {old_pos:.3f} → {self.position:.3f}")
        elif action == 'duplicate':
            new_gene = copy.deepcopy(self)
            new_gene.gene_id = f"{self.gene_type}{self.variant_id}-{uuid.uuid4().hex[:6]}"
            new_gene.parent_gene = self.gene_id
            new_gene.position = self.position + random.uniform(-0.1, 0.1)
            new_gene.position = max(0, min(1, new_gene.position))
            new_gene.copy_number = self.copy_number + 1
            device = next(self.parameters()).device
            new_gene = new_gene.to(device)
            with torch.no_grad():
                for param in new_gene.parameters():
                    param.data += torch.randn_like(param) * 0.1
            self.transposition_history.append({
                'time': timestamp, 'action': 'duplicate',
                'child_id': new_gene.gene_id, 'stress_level': stress_level
            })
            if CFG.verbose_transposition:
                print(f"  🧬 Gene {self.gene_id} duplicated → {new_gene.gene_id}")
            return new_gene
        elif action == 'invert':
            self.is_inverted = not self.is_inverted
            self.transposition_history.append({
                'time': timestamp, 'action': 'invert',
                'inverted_state': self.is_inverted, 'stress_level': stress_level
            })
            if CFG.verbose_transposition:
                print(f"  🔄 Gene {self.gene_id} inverted: {not self.is_inverted} → {self.is_inverted}")
        elif action == 'delete':
            self.is_active = False
            self.transposition_history.append({
                'time': timestamp, 'action': 'delete', 'stress_level': stress_level
            })
            if CFG.verbose_transposition:
                print(f"  ❌ Gene {self.gene_id} deleted (silenced)")
        return None

# ============================================================================
# B-Cell Clone (deeper integrator)
# ============================================================================

class TransposableBCell(nn.Module):
    def __init__(self, initial_genes: List[TransposableGene]):
        super().__init__()
        self.cell_id = uuid.uuid4().hex[:8]
        self.genes = nn.ModuleList(initial_genes)
        self.generation = 0
        self.lineage = []
        self.fitness_history = []
        self.gene_integrator = nn.Sequential(
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(CFG.hidden_dim * 2, CFG.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(CFG.hidden_dim * 2, CFG.hidden_dim),
            nn.LayerNorm(CFG.hidden_dim)
        )
        self.affinity_head = nn.Sequential(
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim),
            nn.ReLU(),
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(CFG.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, antigen_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_vec = antigen_batch.batch
        active_genes = [g for g in self.genes if g.is_active]
        if not active_genes:
            dummy = torch.zeros(antigen_batch.num_graphs, CFG.hidden_dim, device=CFG.device)
            return self.affinity_head(dummy), dummy
        active_genes.sort(key=lambda g: g.position)
        gene_outputs = []
        for g in active_genes:
            out = g(antigen_batch.x, antigen_batch.edge_index, batch_vec)
            gene_outputs.append(out * (1.0 - g.position * 0.3))
        stacked = torch.stack(gene_outputs)
        combined = stacked.mean(dim=0)
        combined = self.gene_integrator(combined)
        affinity = self.affinity_head(combined)
        return affinity, combined

    def undergo_transposition(self, stress_level: float):
        if CFG.verbose_transposition:
            print(f"\n🧫 Cell {self.cell_id} undergoing transposition (stress={stress_level:.2f})")
        new_genes = []
        for i, gene in enumerate(self.genes):
            if not gene.is_active:
                continue
            result = gene.transpose(stress_level)
            if isinstance(result, TransposableGene):
                new_genes.append(result)
        for gene in new_genes:
            self.genes.append(gene)
        if len([g for g in self.genes if g.is_active]) > CFG.max_genes_per_clone:
            self._prune_genes()
        self.generation += 1

    def _prune_genes(self):
        active_genes = [(i, g) for i, g in enumerate(self.genes) if g.is_active]
        if len(active_genes) <= CFG.max_genes_per_clone:
            return
        active_genes.sort(key=lambda x: x[1].fitness_contribution)
        num_to_remove = len(active_genes) - CFG.max_genes_per_clone
        for i, gene in active_genes[:num_to_remove]:
            gene.is_active = False
            if CFG.verbose_transposition:
                print(f"  🗑️  Pruned low-fitness gene {gene.gene_id}")

    def mutate(self):
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < CFG.mutation_rate:
                    param.data += torch.randn_like(param) * 0.01

    def clone(self) -> 'TransposableBCell':
        child_genes = []
        for gene in self.genes:
            if gene.is_active:
                child_gene = copy.deepcopy(gene)
                if random.random() < 0.1:
                    result = child_gene.transpose(0.1)
                    if isinstance(result, TransposableGene):
                        child_genes.append(result)
                child_genes.append(child_gene)
        child = TransposableBCell(child_genes)
        child.lineage = self.lineage + [self.cell_id]
        device = next(self.parameters()).device
        child = child.to(device)
        child.mutate()
        return child

# Don't compile due to incompatibility with scatter operations
if USE_COMPILE:
    # Only compile if explicitly enabled
    TransposableGene.forward = torch.compile(TransposableGene.forward, mode="default")
    TransposableBCell.forward = torch.compile(TransposableBCell.forward, mode="default")

# ============================================================================
# Germinal Center Evolution (FIXED)
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
        self._seed_population()

    def _seed_population(self):
        print(f"🌱 Seeding population with {CFG.initial_population} naive B-cells...")
        for _ in range(CFG.initial_population):
            genes = []
            num_v = random.randint(1, 3)
            for i in range(num_v):
                v_gene = TransposableGene('V', random.randint(1, 50))
                v_gene.position = random.uniform(0, 0.3)
                genes.append(v_gene)
            num_d = random.randint(1, 2)
            for i in range(num_d):
                d_gene = TransposableGene('D', random.randint(1, 30))
                d_gene.position = random.uniform(0.3, 0.6)
                genes.append(d_gene)
            num_j = random.randint(1, 2)
            for i in range(num_j):
                j_gene = TransposableGene('J', random.randint(1, 6))
                j_gene.position = random.uniform(0.6, 1.0)
                genes.append(j_gene)
            cell = TransposableBCell(genes).to(CFG.device)
            self.population[cell.cell_id] = cell

    def compute_population_fitness(self, antigens: List[Data]) -> Dict[str, float]:
        """FIXED: Simplified approach without complex CUDA streams"""
        antigens_subset = antigens[:CFG.eval_antigens_subset]
        if not antigens_subset:
            return {cell_id: 0.0 for cell_id in self.population}
        
        antigen_batch = Batch.from_data_list(antigens_subset).to(CFG.device)
        fitness_scores = {}
        cells_list = list(self.population.items())
        
        # Process cells in smaller batches to avoid memory issues
        with torch.no_grad():  # Add no_grad for evaluation
            with amp_ctx():
                for i in range(0, len(cells_list), CFG.batch_cells):
                    batch_cells = cells_list[i:i+CFG.batch_cells]
                    
                    for cell_id, cell in batch_cells:
                        try:
                            pred_aff, _ = cell(antigen_batch)
                            true_aff = antigen_batch.affinity.unsqueeze(1)
                            error = torch.abs(pred_aff - true_aff)
                            fitness = 1. / (1. + error)
                            mean_fit = fitness.mean()
                            
                            # Penalty for too many genes
                            num_active = len([g for g in cell.genes if g.is_active])
                            mean_fit -= max(0, num_active - 10) * CFG.duplication_cost
                            
                            fitness_scores[cell_id] = mean_fit.item()
                            
                            # Update gene fitness contributions
                            for g in cell.genes:
                                if g.is_active:
                                    g.fitness_contribution = fitness_scores[cell_id]
                                    
                        except Exception as e:
                            print(f"⚠️  Error evaluating cell {cell_id}: {str(e)}")
                            fitness_scores[cell_id] = 0.0
                    
                    # Clear cache periodically to avoid memory buildup
                    if i % (CFG.batch_cells * 4) == 0:
                        torch.cuda.empty_cache()
        
        return fitness_scores

    def detect_stress(self, fitness_scores: Dict[str, float]) -> float:
        mean_fitness = np.mean(list(fitness_scores.values()))
        self.stress_history.append(mean_fitness)
        if len(self.stress_history) < CFG.stress_window:
            return 0.0
        fitness_trend = np.array(self.stress_history)
        fitness_decline = (fitness_trend[0] - fitness_trend[-1]) / (fitness_trend[0] + 1e-6)
        fitness_variance = np.std(list(fitness_scores.values()))
        stress = max(0, fitness_decline) + fitness_variance
        stress = min(1.0, stress)
        return stress

    def evolve(self, antigens: List[Data]):
        self.generation += 1
        gen_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🧬 Generation {self.generation}")
        print(f"{'='*60}")
        
        fitness_scores = self.compute_population_fitness(antigens)
        mean_fitness = np.mean(list(fitness_scores.values()))
        print(f"📈 Mean fitness: {mean_fitness:.4f}")
        
        self.current_stress = self.detect_stress(fitness_scores)
        print(f"🔥 Population stress: {self.current_stress:.4f}")
        
        if self.current_stress > CFG.stress_threshold:
            print(f"\n⚠️  HIGH STRESS DETECTED! Triggering transposition cascade...")
            self._transposition_phase()
            transposition_summary = defaultdict(int)
            for event in self.transposition_events:
                if event['generation'] == self.generation:
                    transposition_summary[event['events']['action']] += 1
            if transposition_summary:
                print(f"  🧬 Transposition summary:")
                for action, count in transposition_summary.items():
                    emoji = {'jump': '🦘', 'duplicate': '🧬', 'invert': '🔄', 'delete': '❌'}.get(action, '❓')
                    print(f"    {emoji} {action}: {count} events")
        
        self._selection_phase(fitness_scores)
        
        self.fitness_landscape.append({
            'generation': self.generation,
            'mean_fitness': mean_fitness,
            'stress_level': self.current_stress,
            'population_size': len(self.population),
            'transposition_events': len(self.transposition_events)
        })
        
        gen_end_time = time.time()
        elapsed = gen_end_time - gen_start_time
        print(f"⏱️  Generation {self.generation} completed in {elapsed:.2f} seconds.")
        
        # Clear CUDA cache after each generation
        torch.cuda.empty_cache()

    def _transposition_phase(self):
        original_verbose = CFG.verbose_transposition
        CFG.verbose_transposition = False
        for cell_id, cell in list(self.population.items()):
            cell.undergo_transposition(self.current_stress)
            for gene in cell.genes:
                if gene.transposition_history:
                    self.transposition_events.append({
                        'generation': self.generation,
                        'cell_id': cell_id,
                        'gene_id': gene.gene_id,
                        'events': gene.transposition_history[-1]
                    })
        CFG.verbose_transposition = original_verbose

    def _selection_phase(self, fitness_scores: Dict[str, float]):
        sorted_cells = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        num_survivors = int(len(sorted_cells) * (1 - CFG.selection_pressure))
        survivor_ids = [cell_id for cell_id, fitness in sorted_cells[:num_survivors]]
        survivors = {cell_id: self.population[cell_id] for cell_id in survivor_ids}
        
        num_reproducers = int(len(survivors) * 0.5)
        new_cells = {}
        for cell_id, fitness in sorted_cells[:num_reproducers]:
            parent = self.population[cell_id]
            num_offspring = int(1 + fitness * 3)
            for _ in range(num_offspring):
                child = parent.clone()
                new_cells[child.cell_id] = child
        
        self.population = {**survivors, **new_cells}
        
        if len(self.population) > CFG.max_population:
            current_fitness = {}
            for cell_id, cell in self.population.items():
                if cell_id in fitness_scores:
                    current_fitness[cell_id] = fitness_scores[cell_id]
                else:
                    current_fitness[cell_id] = 0.5
            sorted_all = sorted(self.population.items(),
                                key=lambda x: current_fitness.get(x[0], 0),
                                reverse=True)
            self.population = dict(sorted_all[:CFG.max_population])
        
        print(f"🧪 Population after selection: {len(self.population)} cells")

# ============================================================================
# Visualization (unchanged)
# ============================================================================

def visualize_genome_evolution(center: TransposableGerminalCenter, save_path: str):
    valid_cells = [(cell_id, cell) for cell_id, cell in center.population.items()
                   if hasattr(cell, 'genes') and hasattr(cell, 'cell_id')]
    if not valid_cells:
        print("No valid cells found for visualization")
        return
    num_cells_to_plot = min(10, len(valid_cells))
    fig, axes = plt.subplots(num_cells_to_plot, 1,
                             figsize=(12, num_cells_to_plot * 0.8))
    if num_cells_to_plot == 1:
        axes = [axes]
    gene_colors = {'V': 'red', 'D': 'green', 'J': 'blue'}
    for i, (ax, (cell_id, cell)) in enumerate(zip(axes, valid_cells[:num_cells_to_plot])):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_fitness_landscape(center: TransposableGerminalCenter, save_path: str):
    if not center.fitness_landscape:
        return
    data = center.fitness_landscape
    generations = [d['generation'] for d in data]
    fitness = [d['mean_fitness'] for d in data]
    stress = [d['stress_level'] for d in data]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(generations, fitness, 'b-', linewidth=2)
    ax1.fill_between(generations, fitness, alpha=0.3)
    ax1.set_ylabel('Mean Population Fitness', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax2.plot(generations, stress, 'r-', linewidth=2)
    ax2.axhline(y=CFG.stress_threshold, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(generations, stress, alpha=0.3, color='red')
    ax2.set_ylabel('Population Stress Level', fontsize=12)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.grid(True, alpha=0.3)
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
    print("\n🦠 Starting Viral Escape Simulation with Transposable Elements\n")
    sim_start_time = time.time()
    center = TransposableGerminalCenter()
    viral_timeline = [
        (0, [], "Wild Type"),
        (10, [5], "Alpha Variant"),
        (20, [5, 12], "Beta Variant"),
        (30, [5, 12, 18], "Delta Variant"),
        (40, [3, 7, 12, 15, 18], "Omicron Variant"),
        (50, [1, 3, 5, 7, 9, 12, 15, 17, 18], "Hypothetical Escape Variant")
    ]

    current_variant_idx = 0
    for epoch in range(CFG.epochs):
        epoch_start_time = time.time()
        if current_variant_idx < len(viral_timeline) - 1:
            if epoch >= viral_timeline[current_variant_idx + 1][0]:
                current_variant_idx += 1
                _, mutations, variant_name = viral_timeline[current_variant_idx]
                print(f"\n🚨 VIRUS MUTATED TO {variant_name}! Sites: {mutations}")
                print(f"{'='*60}\n")
                center.current_stress = 1.0
        _, mutations, variant_name = viral_timeline[current_variant_idx]
        antigens = []
        for _ in range(CFG.batch_size):
            antigen = generate_antigen_graph(mutation_sites=mutations)
            antigens.append(antigen)
        center.evolve(antigens)
        epoch_end_time = time.time()
        epoch_elapsed = epoch_end_time - epoch_start_time
        print(f"⏳ Epoch {epoch+1}/{CFG.epochs} took {epoch_elapsed:.2f} seconds.")
        if epoch % CFG.plot_interval == 0:
            vis_start = time.time()
            visualize_genome_evolution(center,
                                       f"{CFG.save_dir}/genomes_gen_{epoch:03d}.png")
            vis_end = time.time()
            print(f"🖼️  Visualization for epoch {epoch+1} took {vis_end - vis_start:.2f} seconds.")
    plot_start = time.time()
    plot_fitness_landscape(center, f"{CFG.save_dir}/fitness_landscape.png")
    plot_end = time.time()
    print(f"📈 Fitness landscape plotting took {plot_end - plot_start:.2f} seconds.")
    with open(f"{CFG.save_dir}/transposition_events.json", 'w') as f:
        json.dump(center.transposition_events, f, indent=2)
    sim_end_time = time.time()
    total_elapsed = sim_end_time - sim_start_time
    print(f"\n✅ Simulation complete! Results saved to {CFG.save_dir}/")
    print(f"🧬 Total transposition events: {len(center.transposition_events)}")
    print(f"🕒 Total simulation time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print("\n📊 Transposition Statistics:")
    event_types = defaultdict(int)
    for event in center.transposition_events:
        event_types[event['events']['action']] += 1
    for action, count in event_types.items():
        emoji = {'jump': '🦘', 'duplicate': '🧬', 'invert': '🔄', 'delete': '❌'}.get(action, '❓')
        print(f"  {emoji} {action}: {count} events")

if __name__ == "__main__":
    enable_cuda_fast_math()
    simulate_viral_escape()