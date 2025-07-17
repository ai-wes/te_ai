"""
Transposable Element AI - Temporal Aging Rejuvenation Sim
============================================================================
Temporal sim: Generations = time/age; antigens accumulate decay (hallmarks like telomere/epigenetic drift). Evolves reversal via transpositions (Yamanaka-inspired rewind).
Maps to real: Outputs seq combos for lab (e.g., SB000-like single factors, EPOCH cocktails).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean
import numpy as np
import random
import copy
import uuid
import os
from datetime import datetime
import torch.cuda as cuda
from collections import defaultdict, deque

# Fast Math Setup (unchanged)
def enable_cuda_fast_math(seed=42):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
enable_cuda_fast_math()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

USE_AMP = True
AMP_DTYPE = torch.bfloat16
from torch.amp import autocast
amp_ctx = lambda: autocast(device_type='cuda', dtype=AMP_DTYPE) if USE_AMP else nullcontext

@dataclass
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 256
    hidden_dim = 1024
    num_gcn_layers = 4
    eval_antigens_subset = 128
    base_transpose_prob = 0.01
    stress_multiplier = 10.0
    duplication_cost = 0.1
    max_genes_per_clone = 50
    initial_population = 100
    max_population = 20000
    selection_pressure = 0.3
    mutation_rate = 0.01
    epochs = 300  # Temporal "years"
    batch_size = 256
    learning_rate = 0.001
    stress_window = 10
    stress_threshold = 0.5
    plot_interval = 10
    save_dir = "aging_rejuv_results"
    verbose_transposition = True
    num_streams = 8
    # Aging params (plausible from hallmarks: decay ramps over time)
    decay_rate = 0.005  # Accumulating damage per gen
    rewind_efficacy = 0.2  # Reversal per successful transposition (Yamanaka-like)

CFG = Config()
os.makedirs(CFG.save_dir, exist_ok=True)

# Antigen as Time/Age: Mutations accumulate over gens (damage/decay)
def generate_aging_antigen(num_nodes=256, age_level=0.0, mutation_sites=None):
    positions = torch.rand(num_nodes, 2)
    distances = torch.cdist(positions, positions)
    adj_matrix = (distances < 0.3).float()
    edge_index = adj_matrix.nonzero().t()
    edge_index = to_undirected(edge_index)
    features = torch.randn(num_nodes, CFG.feature_dim)
    # Age decay: Modulate features by time (epigenetic drift sim)
    features *= (1 - age_level * 0.5)  # Weaken with age
    if mutation_sites:
        for site in mutation_sites:
            if site < num_nodes:
                features[site] += torch.randn(CFG.feature_dim) * (2.0 * age_level)  # More damage over time
    affinity = torch.rand(1).item() * (1 - age_level)  # "Youth affinity" declines
    return Data(x=features, edge_index=edge_index, affinity=affinity, num_nodes=num_nodes)

# Base TransposableGene (with real seq mutation for molecular plausibility)
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
        self.dna_seq = "ATG" + "".join(random.choice("ATGC") for _ in range(50))  # Longer for realism

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

    def mutate_seq(self):
        # Plausible point mutation (aging effect)
        idx = random.randint(0, len(self.dna_seq)-1)
        self.dna_seq = self.dna_seq[:idx] + random.choice("ATGC") + self.dna_seq[idx+1:]

    def transpose(self, stress_level: float) -> Optional['TransposableGene']:
        transpose_prob = CFG.base_transpose_prob * (1 + stress_level * CFG.stress_multiplier)
        if random.random() > transpose_prob:
            return None
        weights = [0.2, 0.4, 0.3, 0.1] if stress_level > 0.7 else [0.6, 0.2, 0.1, 0.1]
        action = random.choices(['jump', 'duplicate', 'invert', 'delete'], weights=weights)[0]
        timestamp = datetime.now().isoformat()
        if action == 'jump':
            old_pos = self.position
            self.position = random.random()
            self.transposition_history.append({'action': 'jump', 'from': old_pos, 'to': self.position})
            if CFG.verbose_transposition:
                print(f"  🦘 {self.gene_id} jumped: {old_pos:.3f} → {self.position:.3f}")
        elif action == 'duplicate':
            new_gene = copy.deepcopy(self)
            new_gene.gene_id = f"{self.gene_type}{self.variant_id}-{uuid.uuid4().hex[:6]}"
            new_gene.parent_gene = self.gene_id
            new_gene.position = self.position + random.uniform(-0.1, 0.1)
            new_gene.position = max(0, min(1, new_gene.position))
            new_gene.copy_number = self.copy_number + 1
            new_gene = new_gene.to(next(self.parameters()).device)
            with torch.no_grad():
                for param in new_gene.parameters():
                    param.data += torch.randn_like(param) * 0.1
            new_gene.mutate_seq()  # Diverge for neofunctionalization
            self.transposition_history.append({'action': 'duplicate', 'child': new_gene.gene_id})
            if CFG.verbose_transposition:
                print(f"  🧬 {self.gene_id} duplicated → {new_gene.gene_id} (seq: {new_gene.dna_seq[:20]}...)")
            return new_gene
        elif action == 'invert':
            self.is_inverted = not self.is_inverted
            self.dna_seq = self.dna_seq[::-1]  # Histone flip sim
            self.transposition_history.append({'action': 'invert', 'state': self.is_inverted})
            if CFG.verbose_transposition:
                print(f"  🔄 {self.gene_id} inverted (seq: {self.dna_seq[:20]}...)")
        elif action == 'delete':
            self.is_active = False
            self.transposition_history.append({'action': 'delete'})
            if CFG.verbose_transposition:
                print(f"  ❌ {self.gene_id} deleted")
        return None

# Molecular Building Blocks: Genes for real aging effects (telomere, epigenetic, senescence)
class TelomereGene(TransposableGene):
    def __init__(self, variant_id: int):
        super().__init__('TEL', variant_id)
        self.length = 1.0  # Starts full, decays with time

    def decay(self, decay_level: float):
        self.length = max(0.0, self.length - decay_level * random.uniform(0.01, 0.03))  # Attrition like hallmarks

    def forward(self, x, edge_index, batch=None):
        h = super().forward(x, edge_index, batch)
        return h * self.length  # Short telomeres weaken

    def transpose(self, stress_level):
        new = super().transpose(stress_level)
        if new and 'duplicate' in new.transposition_history[-1]['action']:
            self.length = min(1.0, self.length + CFG.rewind_efficacy)  # Telomerase-like reversal
        return new

class EpigeneticGene(TransposableGene):
    def __init__(self, variant_id: int):
        super().__init__('EPI', variant_id)
        self.clock = 0.0  # Methylation clock

    def decay(self, decay_level: float):
        self.clock = min(1.0, self.clock + decay_level * random.uniform(0.005, 0.01))  # Drift

    def forward(self, x, edge_index, batch=None):
        h = super().forward(x, edge_index, batch)
        return h * (1 - self.clock)  # Aged epi reduces

    def transpose(self, stress_level):
        new = super().transpose(stress_level)
        if self.clock > 0.5 and random.random() < stress_level:
            self.clock = max(0.0, self.clock - CFG.rewind_efficacy)  # Yamanaka rewind
            print(f"  ⏪ EPI {self.gene_id} clock rewound: {self.clock:.3f}")
        return new

class SenescenceGene(TransposableGene):
    def __init__(self, variant_id: int):
        super().__init__('SEN', variant_id)
        self.sasp_level = 0.0  # Senescence-associated secretory phenotype

    def decay(self, decay_level: float):
        self.sasp_level = min(1.0, self.sasp_level + decay_level * random.uniform(0.015, 0.025))  # Inflammation build

    def forward(self, x, edge_index, batch=None):
        h = super().forward(x, edge_index, batch)
        return h * (1 - self.sasp_level)  # SASP penalizes

    def transpose(self, stress_level):
        new = super().transpose(stress_level)
        if self.sasp_level > 0.4 and 'delete' in (new.transposition_history[-1]['action'] if new else ''):
            self.sasp_level = max(0.0, self.sasp_level - CFG.rewind_efficacy * 1.5)  # Senolytic clearance
        return new

# TemporalCell: Evolves against time decay
class TemporalCell(TransposableBCell):
    def __init__(self, initial_genes):
        super().__init__(initial_genes)
        self.decay_accum = 0.0  # Overall aging load

    @torch.no_grad()
    def forward(self, antigen_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        affinity, combined = super().forward(antigen_batch)
        tel_avg = np.mean([g.length for g in self.genes if isinstance(g, TelomereGene)] or [1.0])
        epi_avg = np.mean([g.clock for g in self.genes if isinstance(g, EpigeneticGene)] or [0.0])
        sen_avg = np.mean([g.sasp_level for g in self.genes if isinstance(g, SenescenceGene)] or [0.0])
        youth_mod = tel_avg * (1 - epi_avg) * (1 - sen_avg)
        combined *= youth_mod
        return affinity, combined

    def apply_decay(self, decay_level: float):
        self.decay_accum += decay_level
        for gene in self.genes:
            gene.decay(decay_level)
            if random.random() < CFG.mutation_rate:
                gene.mutate_seq()  # Aging mutations

    def undergo_transposition(self, stress_level: float):
        super().undergo_transposition(stress_level)
        if self.decay_accum > 0.5:
            # Cascade for reversal (real rejuvenation sim)
            for gene in random.sample(self.genes, min(5, len(self.genes))):
                gene.transpose(stress_level * 1.2)

    def export_discovery(self):
        # Real-map output: Evolved blocks as interventions
        active = [g for g in self.genes if g.is_active]
        combo = {'tel': [g.dna_seq for g in active if isinstance(g, TelomereGene)],
                 'epi': [g.dna_seq for g in active if isinstance(g, EpigeneticGene)],
                 'sen': [g.dna_seq for g in active if isinstance(g, SenescenceGene)]}
        print(f"Reversal discovery: {combo} (decay reversed: {1 - self.decay_accum:.3f})")
        return combo

# TemporalGerminalCenter: Time as stressor
class TemporalGerminalCenter(TransposableGerminalCenter):
    def _seed_population(self):
        print(f"Seeding {CFG.initial_population} temporal cells...")
        for _ in range(CFG.initial_population):
            genes = []
            # Building blocks: Mix molecular genes
            for _ in range(random.randint(1, 3)):
                genes.append(TelomereGene(random.randint(1,10)))
                genes.append(EpigeneticGene(random.randint(1,10)))
                genes.append(SenescenceGene(random.randint(1,10)))
            # Base V/D/J for structure
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
            cell = TemporalCell(genes).to(CFG.device)
            self.population[cell.cell_id] = cell

    def evolve(self, antigens):
        self.generation += 1
        decay_level = self.generation * CFG.decay_rate / CFG.epochs  # Time-ramped decay
        print(f"\n=== Gen {self.generation} (Age Decay: {decay_level:.4f}) ===")
        for cell in self.population.values():
            cell.apply_decay(decay_level)  # Apply time stressor
        fitness_scores = self.compute_population_fitness(antigens)
        self.current_stress = self.detect_stress(fitness_scores) + decay_level  # Age boosts stress
        if self.current_stress > CFG.stress_threshold:
            print("⚠️ Age stress high! Evolving reversals...")
            self._transposition_phase()
        self._selection_phase(fitness_scores)
        if self.generation % CFG.plot_interval == 0:
            top_cell = max(self.population.values(), key=lambda c: 1 - c.decay_accum)
            top_cell.export_discovery()

# Compile (unchanged)
TransposableGene.forward = torch.compile(TransposableGene.forward, mode="max-autotune")
TemporalCell.forward = torch.compile(TemporalCell.forward, mode="max-autotune")

# Main Temporal Sim
def simulate_temporal_rejuv():
    print("\n🧬 Temporal Aging Rejuvenation Sim: Time as Stressor\n")
    center = TemporalGerminalCenter()
    for epoch in range(CFG.epochs):
        # Antigens age with time (accumulating sites)
        num_sites = int(epoch / 10)  # Ramp damage
        sites = random.sample(range(256), num_sites)
        antigens = [generate_aging_antigen(age_level=epoch/CFG.epochs, mutation_sites=sites) for _ in range(CFG.batch_size)]
        center.evolve(antigens)
    print("\n✅ Sim complete! Top reversals exported above.")

if __name__ == "__main__":
    enable_cuda_fast_math()
    simulate_temporal_rejuv()