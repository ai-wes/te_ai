"""
Transposable Element Adaptive Immune System AI - Longevity Enhancement Edition
============================================================================
Extended for plausible longevity evolution: telomere, epigenetic, metabolic genes evolve under aging stress.
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

# --- Fast Math Setup (unchanged) ---
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
    epochs = 300
    batch_size = 256
    learning_rate = 0.001
    stress_window = 10
    stress_threshold = 0.5
    plot_interval = 10
    save_dir = "transposon_results"
    verbose_transposition = True
    num_streams = 8
    # New for longevity: aging degradation rates (plausible bio scales)
    telomere_degrade = 0.01  # ~2% loss per "division"
    epi_tick = 0.005  # Methylation creep
    metabolic_ros = 0.015  # ROS build-up

CFG = Config()
os.makedirs(CFG.save_dir, exist_ok=True)

# --- Antigen Generation (unchanged, larger for saturation) ---
def generate_antigen_graph(num_nodes=256, mutation_sites=None):
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

# --- Base TransposableGene (unchanged, deeper) ---
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
            self.transposition_history.append({'time': timestamp, 'action': 'jump', 'from_position': old_pos, 'to_position': self.position, 'stress_level': stress_level})
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
            self.transposition_history.append({'time': timestamp, 'action': 'duplicate', 'child_id': new_gene.gene_id, 'stress_level': stress_level})
            if CFG.verbose_transposition:
                print(f"  🧬 Gene {self.gene_id} duplicated → {new_gene.gene_id}")
            return new_gene
        elif action == 'invert':
            self.is_inverted = not self.is_inverted
            self.transposition_history.append({'time': timestamp, 'action': 'invert', 'inverted_state': self.is_inverted, 'stress_level': stress_level})
            if CFG.verbose_transposition:
                print(f"  🔄 Gene {self.gene_id} inverted: {not self.is_inverted} → {self.is_inverted}")
        elif action == 'delete':
            self.is_active = False
            self.transposition_history.append({'time': timestamp, 'action': 'delete', 'stress_level': stress_level})
            if CFG.verbose_transposition:
                print(f"  ❌ Gene {self.gene_id} deleted (silenced)")
        return None

# --- New Longevity-Specific Genes (plausible, novel integration) ---
class TelomereGene(TransposableGene):
    def __init__(self, variant_id: int):
        super().__init__('T', variant_id)
        self.telomere_length = 1.0

    def forward(self, x, edge_index, batch=None):
        h = super().forward(x, edge_index, batch)
        return h * self.telomere_length

    def age(self, stress_level):
        self.telomere_length = max(0.0, self.telomere_length - CFG.telomere_degrade * (1 + stress_level))
        if self.telomere_length < 0.2:
            self.is_active = False

    def transpose(self, stress_level):
        new_gene = super().transpose(stress_level)
        if new_gene and new_gene.action == 'duplicate':  # Plausible telomerase-like elongation
            self.telomere_length = min(1.0, self.telomere_length + 0.15)
        return new_gene

class EpigeneticGene(TransposableGene):
    def __init__(self, variant_id: int):
        super().__init__('E', variant_id)
        self.epi_clock = 0.0

    def forward(self, x, edge_index, batch=None):
        h = super().forward(x, edge_index, batch)
        return h * (1 - self.epi_clock)

    def tick(self, stress_level):
        self.epi_clock = min(1.0, self.epi_clock + CFG.epi_tick * (1 + stress_level))

    def transpose(self, stress_level):
        new_gene = super().transpose(stress_level)
        if self.epi_clock > 0.5 and random.random() < stress_level:  # Plausible Yamanaka-like rewind
            self.epi_clock = max(0.0, self.epi_clock - 0.2)
        return new_gene

class MetabolicGene(TransposableGene):
    def __init__(self, variant_id: int):
        super().__init__('M', variant_id)
        self.efficiency = random.uniform(0.8, 1.0)
        self.ros_level = 0.0

    def forward(self, x, edge_index, batch=None):
        h = super().forward(x, edge_index, batch)
        return h * self.efficiency * (1 - self.ros_level)

    def accumulate_ros(self, stress_level):
        self.ros_level = min(1.0, self.ros_level + CFG.metabolic_ros * stress_level)

    def transpose(self, stress_level):
        new_gene = super().transpose(stress_level)
        if new_gene and new_gene.action == 'duplicate':  # Plausible sirtuin activation
            self.ros_level = max(0.0, self.ros_level - 0.25)
            self.efficiency = min(1.0, self.efficiency + 0.1)
        return new_gene

# --- LongevityCell: Extended BCell for longevity evolution ---
class LongevityCell(TransposableBCell):
    def __init__(self, initial_genes):
        super().__init__(initial_genes)
        self.longevity_integrator = nn.Linear(CFG.hidden_dim, CFG.hidden_dim)  # Integrates aging effects
        self.lifespan = 0  # Simulated generations survived without senescence

    @torch.no_grad()
    def forward(self, antigen_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        affinity, combined = super().forward(antigen_batch)
        # Novel integration: Modulate with longevity genes
        telomere_avg = np.mean([g.telomere_length for g in self.genes if isinstance(g, TelomereGene)] or [1.0])
        epi_avg = np.mean([g.epi_clock for g in self.genes if isinstance(g, EpigeneticGene)] or [0.0])
        ros_avg = np.mean([g.ros_level for g in self.genes if isinstance(g, MetabolicGene)] or [0.0])
        longevity_mod = telomere_avg * (1 - epi_avg) * (1 - ros_avg)
        combined = self.longevity_integrator(combined * longevity_mod)
        return affinity, combined

    def age_and_enhance(self, stress_level):
        for gene in self.genes:
            if isinstance(gene, TelomereGene):
                gene.age(stress_level)
            elif isinstance(gene, EpigeneticGene):
                gene.tick(stress_level)
            elif isinstance(gene, MetabolicGene):
                gene.accumulate_ros(stress_level)
        self.lifespan += 1 if all(g.is_active for g in self.genes if isinstance(g, TelomereGene)) else 0  # Plausible lifespan track

    def undergo_transposition(self, stress_level):
        super().undergo_transposition(stress_level)
        # Novel cascade: High aging stress chains transpositions for rejuvenation
        if stress_level > 0.6:
            for gene in random.sample(self.genes, min(3, len(self.genes))):
                gene.transpose(stress_level * 1.2)  # Boosted for enhancement

# --- GerminalCenter with longevity hooks (plausible evolution under aging stress) ---
class TransposableGerminalCenter:
    def __init__(self):
        self.population = {}
        self.generation = 0
        self.stress_history = deque(maxlen=CFG.stress_window)
        self.current_stress = 0.0
        self.transposition_events = []
        self.fitness_landscape = []
        self._seed_population()

    def _seed_population(self):
        print(f"Seeding with {CFG.initial_population} longevity cells...")
        for _ in range(CFG.initial_population):
            genes = []
            # Add longevity genes plausibly (mix with V/D/J)
            for _ in range(random.randint(1, 3)):
                genes.append(TelomereGene(random.randint(1, 10)))
                genes.append(EpigeneticGene(random.randint(1, 10)))
                genes.append(MetabolicGene(random.randint(1, 10)))
            # Add base V/D/J (unchanged)
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
            cell = LongevityCell(genes).to(CFG.device)
            self.population[cell.cell_id] = cell

    def compute_population_fitness(self, antigens):
        # (unchanged batch/stream logic, but add longevity penalties/rewards)
        fitness_scores = {}
        # ... (original code for batching and streams)
        for cell_id, mean_fit, genes in results:
            fitness_scores[cell_id] = mean_fit.item()
            telomere_avg = np.mean([g.telomere_length for g in genes if isinstance(g, TelomereGene)] or [1.0])
            epi_avg = np.mean([g.epi_clock for g in genes if isinstance(g, EpigeneticGene)] or [0.0])
            ros_avg = np.mean([g.ros_level for g in genes if isinstance(g, MetabolicGene)] or [0.0])
            longevity_bonus = telomere_avg - epi_avg - ros_avg + (cell.lifespan / 10.0)  # Plausible reward for extended life
            fitness_scores[cell_id] += longevity_bonus * 0.3
            for g in genes:
                if g.is_active:
                    g.fitness_contribution = fitness_scores[cell_id]
        return fitness_scores

    def evolve(self, antigens):
        self.generation += 1
        # ... (original print/fitness/stress)
        for cell in self.population.values():
            cell.age_and_enhance(self.current_stress)  # New longevity aging step
        if self.current_stress > CFG.stress_threshold:
            self._transposition_phase()
        self._selection_phase(fitness_scores)
        # ... (original landscape update, add avg_lifespan tracking)

# --- Compile (unchanged) ---
TransposableGene.forward = torch.compile(TransposableGene.forward, mode="max-autotune")
LongevityCell.forward = torch.compile(LongevityCell.forward, mode="max-autotune")

# --- Main Sim (adapted for longevity, with "aging mutations" as antigens) ---
def simulate_longevity_evolution():
    print("\n🧬 Starting Longevity Evolution Simulation with TE-AI\n")
    center = TransposableGerminalCenter()
    aging_timeline = [
        (0, [], "Young State"),
        (50, [5], "Mild Aging (ROS Increase)"),
        (100, [5, 12], "Moderate Aging (DNA Damage)"),
        (150, [5, 12, 18], "Advanced Aging (Senescence)"),
    ]
    current_phase = 0
    for epoch in range(CFG.epochs):
        if current_phase < len(aging_timeline) - 1 and epoch >= aging_timeline[current_phase + 1][0]:
            current_phase += 1
            _, mutations, phase_name = aging_timeline[current_phase]
            print(f"\n🚨 AGING PHASE: {phase_name}! Mutations: {mutations}")
            center.current_stress = 1.0  # Trigger rejuvenation
        antigens = [generate_antigen_graph(mutation_sites=mutations) for _ in range(CFG.batch_size)]
        center.evolve(antigens)
        if epoch % CFG.plot_interval == 0:
            # (add longevity-specific plots, e.g., avg telomere over gens)
            pass
    print(f"\n✅ Longevity sim complete! Evolved cells with mean lifespan: {np.mean([cell.lifespan for cell in center.population.values()])} generations")

if __name__ == "__main__":
    enable_cuda_fast_math()
    simulate_longevity_evolution()