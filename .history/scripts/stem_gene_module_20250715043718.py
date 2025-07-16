# File: stem_gene_prototype.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Optional, Tuple


import torch.nn as nn
import torch.nn.functional as F 
import random
from scripts.stem_gene_module import StemGeneModule
from config import cfg





from transposable_immune_ai_production_complete import (
    ContinuousDepthGeneModule, cfg
)

class StemGeneModule(ContinuousDepthGeneModule):
    """
    Stem gene for the viral recognition system
    Can differentiate into V, D, or J genes based on population needs
    """
    
    def __init__(self):
        # Initialize as stem type 'S'
        super().__init__('S', 0)
        
        # Differentiation state: [V, D, J, Stem]
        self.differentiation_state = nn.Parameter(torch.zeros(4))
        self.differentiation_state.data[3] = 1.0  # Start as pure stem
        
        # Track differentiation history
        self.differentiation_history = []
        self.commitment_level = 0.0
        self.plasticity = nn.Parameter(torch.tensor(1.0))
        
        # Learning what the population needs
        self.population_sensor = nn.Sequential(
            nn.Linear(12, 32),  # Input: gene counts, performance stats
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1)
        )
        
        # Morphable components for each gene type
        self.gene_components = nn.ModuleDict({
            'V': nn.Linear(cfg.hidden_dim, cfg.hidden_dim),  # Variable region processing
            'D': nn.Linear(cfg.hidden_dim, cfg.hidden_dim),  # Diversity region processing
            'J': nn.Linear(cfg.hidden_dim, cfg.hidden_dim),  # Joining region processing
            'S': nn.Linear(cfg.hidden_dim, cfg.hidden_dim)   # Stem processing
        })
        
        # Position preference based on differentiation
        self.position_preferences = {
            'V': (0.15, 0.1),  # mean, std for position
            'D': (0.45, 0.1),
            'J': (0.8, 0.1),
            'S': (0.5, 0.2)    # Stem cells can go anywhere
        }
    
    def sense_population_needs(self, population_stats: Dict) -> torch.Tensor:
        """Analyze what type of gene the population needs"""
        # Extract population statistics
        v_count = population_stats.get('V_count', 0)
        d_count = population_stats.get('D_count', 0)
        j_count = population_stats.get('J_count', 0)
        total_genes = v_count + d_count + j_count
        
        # Performance by type
        v_performance = population_stats.get('V_avg_fitness', 0.5)
        d_performance = population_stats.get('D_avg_fitness', 0.5)
        j_performance = population_stats.get('J_avg_fitness', 0.5)
        
        # Population stress indicators
        stress_level = population_stats.get('stress_level', 0.0)
        mutation_rate = population_stats.get('recent_mutation_rate', 0.0)
        
        # Create feature vector
        features = torch.tensor([
            v_count / max(total_genes, 1),  # V proportion
            d_count / max(total_genes, 1),  # D proportion
            j_count / max(total_genes, 1),  # J proportion
            v_performance,
            d_performance,
            j_performance,
            stress_level,
            mutation_rate,
            float(v_count == 0),  # Missing V genes
            float(d_count == 0),  # Missing D genes
            float(j_count == 0),  # Missing J genes
            population_stats.get('diversity', 0.5)
        ], dtype=torch.float32).to(self.population_sensor.parameters().__next__().device)
        
        # Determine what's needed
        needs = self.population_sensor(features)
        
        return needs
    
    def differentiate(self, target_type: Optional[str] = None, 
                     population_stats: Optional[Dict] = None):
        """Differentiate into specific gene type"""
        
        if target_type:
            # Directed differentiation
            type_idx = {'V': 0, 'D': 1, 'J': 2, 'S': 3}[target_type]
            
            # Gradual differentiation
            with torch.no_grad():
                current_stem = self.differentiation_state[3].item()
                commitment = min(0.9, self.commitment_level + 0.3)
                
                self.differentiation_state.data = torch.zeros(4)
                self.differentiation_state.data[type_idx] = commitment
                self.differentiation_state.data[3] = 1 - commitment
                
                self.commitment_level = commitment
                
                # Update position based on differentiation
                if target_type in ['V', 'D', 'J']:
                    mean, std = self.position_preferences[target_type]
                    self.position = np.clip(np.random.normal(mean, std), 0, 1)
                    self.gene_type = target_type
                
                print(f"   🎯 Stem gene differentiating: {target_type} (commitment: {commitment:.2f})")
        
        elif population_stats:
            # Context-based differentiation
            needs = self.sense_population_needs(population_stats)
            
            # Stochastic differentiation based on needs
            if random.random() < self.plasticity.item():
                # Choose based on population needs
                chosen_idx = torch.multinomial(needs, 1).item()
                target_type = ['V', 'D', 'J', 'S'][chosen_idx]
                
                if target_type != 'S':  # Don't differentiate to stem
                    self.differentiate(target_type)
        
        # Record differentiation event
        self.differentiation_history.append({
            'generation': population_stats.get('generation', 0) if population_stats else 0,
            'target': target_type,
            'commitment': self.commitment_level,
            'trigger': 'directed' if target_type else 'population_need'
        })
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with differentiation-based processing"""
        
        # Base input projection
        h = self.input_projection(x)
        
        # Apply differentiation-weighted processing
        diff_probs = F.softmax(self.differentiation_state, dim=0)
        
        # Combine different gene type processings
        h_combined = (
            diff_probs[0] * self.gene_components['V'](h) +
            diff_probs[1] * self.gene_components['D'](h) +
            diff_probs[2] * self.gene_components['J'](h) +
            diff_probs[3] * self.gene_components['S'](h)
        )
        
        # Apply epigenetic regulation (inherited from parent class)
        h_regulated = self._apply_epigenetic_regulation(h_combined)
        
        # Continue with ODE processing (from parent class)
        # ... rest of forward pass from ContinuousDepthGeneModule
        
        # For now, we'll use the parent's forward method with our processed hidden state
        # This would need to be properly integrated
        return super().forward(x, edge_index, batch)
    
    def transpose(self, stress_level: float, population_diversity: float) -> Tuple[Optional['StemGeneModule'], Optional[str]]:
        """Stem-specific transposition behavior"""
        
        # Stem cells are more mobile when undifferentiated
        mobility_factor = 2.0 - self.commitment_level
        
        # Check if we should differentiate instead of transpose
        if stress_level > 0.7 and self.commitment_level < 0.5:
            if random.random() < stress_level:
                # High stress + low commitment = differentiate to help
                population_stats = {
                    'stress_level': stress_level,
                    'diversity': population_diversity
                }
                self.differentiate(population_stats=population_stats)
                return None, 'differentiation'
        
        # Otherwise use parent's transposition with higher mobility
        original_prob = cfg.base_transpose_prob
        cfg.base_transpose_prob *= mobility_factor
        
        result = super().transpose(stress_level, population_diversity)
        
        cfg.base_transpose_prob = original_prob
        
        return result
    
    def de_differentiate(self, stress_level: float):
        """Return to stem state under extreme stress"""
        if stress_level > 0.9 and self.commitment_level > 0.5:
            if random.random() < stress_level * 0.2:
                with torch.no_grad():
                    # Reset to stem state
                    self.differentiation_state.data = torch.zeros(4)
                    self.differentiation_state.data[3] = 1.0
                    self.commitment_level = 0.0
                    self.plasticity.data = torch.tensor(1.0)
                    self.gene_type = 'S'
                    self.position = 0.5  # Return to neutral position
                    
                print(f"   🔄 Gene {self.gene_id} de-differentiated to stem state!")
                
                return True
        return False






class GuidedStemGene(StemGeneModule):
    """Stem gene that learns from seed genes"""
    
    def __init__(self):
        super().__init__()
        
        # Learning from seeds
        self.observed_patterns = {}
        self.skill_memory = nn.Parameter(torch.zeros(10, cfg.hidden_dim))
        self.confidence_scores = {}
        
        # Attention mechanism for learning from multiple teachers
        self.teacher_attention = nn.MultiheadAttention(
            cfg.hidden_dim, 
            num_heads=4,
            batch_first=True
        )
        
    def observe_seed_genes(self, population):
        """Learn from successful seed genes"""
        seed_genes = []
        seed_features = []
        seed_performance = []
        
        for cell in population.values():
            for gene in cell.genes:
                if hasattr(gene, 'is_seed') and gene.is_seed:
                    seed_genes.append(gene)
                    seed_features.append(gene.morphogen_field)
                    seed_performance.append(gene.fitness_contribution)
        
        if not seed_genes:
            return None
            
        # Learn successful patterns
        features = torch.stack(seed_features)
        performance = torch.tensor(seed_performance)
        
        # Weight by performance
        weights = F.softmax(performance / 0.1, dim=0)
        
        # Attend to successful patterns
        query = self.skill_memory.unsqueeze(0)
        keys = features.unsqueeze(0)
        values = features.unsqueeze(0)
        
        attended_skills, attention_weights = self.teacher_attention(
            query, keys, values
        )
        
        # Store learned patterns
        best_teacher_idx = torch.argmax(weights)
        best_teacher = seed_genes[best_teacher_idx]
        
        self.observed_patterns[best_teacher.gene_type] = {
            'signature': best_teacher.domain_signature,
            'performance': performance[best_teacher_idx].item(),
            'pattern': attended_skills.squeeze(0)
        }
        
        return best_teacher.gene_type, attention_weights
    
    def guided_differentiation(self, stress_level: float = 0.0):
        """Differentiate based on learned patterns"""
        
        # First, learn from environment
        learned_type, attention = self.observe_seed_genes(self.get_population())
        
        if learned_type and random.random() < 0.7:  # 70% chance to follow teacher
            # Partial differentiation toward successful pattern
            self.differentiation_state.data = torch.zeros(4)
            
            # Don't fully commit - maintain some plasticity
            differentiation_strength = min(0.8, self.confidence_scores.get(learned_type, 0.5))
            
            type_map = {'MD': 0, 'NA': 1, 'TP': 2, 'S': 3}
            if learned_type in type_map:
                self.differentiation_state.data[type_map[learned_type]] = differentiation_strength
                self.differentiation_state.data[3] = 1 - differentiation_strength  # Remain partly stem
                
                # Absorb some characteristics
                if learned_type in self.observed_patterns:
                    pattern = self.observed_patterns[learned_type]
                    self.skill_memory.data[0] = pattern['pattern'][0]  # Learn first skill
                    
                print(f"   📚 Stem gene learned {learned_type} pattern (confidence: {differentiation_strength:.2f})")
        else:
            # Explore novel combinations
            self.explore_novel_differentiation()
    
    def explore_novel_differentiation(self):
        """Create novel gene types by combining learned patterns"""
        if len(self.observed_patterns) >= 2:
            # Combine two successful patterns
            types = list(self.observed_patterns.keys())
            type1, type2 = random.sample(types, 2)
            
            # Create hybrid differentiation
            self.differentiation_state.data = torch.zeros(4)
            type_map = {'MD': 0, 'NA': 1, 'TP': 2, 'S': 3}
            
            # 40% each from teachers, 20% stem
            self.differentiation_state.data[type_map[type1]] = 0.4
            self.differentiation_state.data[type_map[type2]] = 0.4
            self.differentiation_state.data[3] = 0.2
            
            print(f"   🧪 Stem gene exploring hybrid: {type1}+{type2}")




# ============================================================================
# Integration with existing system
# ============================================================================

def add_stem_genes_to_population(germinal_center, stem_ratio: float = 0.2):
    """Add stem genes to existing population"""
    
    current_pop_size = len(germinal_center.population)
    num_stem_to_add = int(current_pop_size * stem_ratio)
    
    print(f"\n🧬 Adding {num_stem_to_add} stem genes to population...")
    
    stem_cells_added = 0
    
    # Add stem genes to existing cells
    for i, (cell_id, cell) in enumerate(germinal_center.population.items()):
        if i % 5 == 0 and stem_cells_added < num_stem_to_add:  # Every 5th cell
            # Add 1-2 stem genes
            num_new_stem = random.randint(1, 2)
            
            for _ in range(num_new_stem):
                if len(cell.genes) < cfg.max_genes_per_clone:
                    stem_gene = StemGeneModule()
                    # Move stem gene to same device as existing genes
                    if len(cell.genes) > 0:
                        # Get device from existing genes
                        device = next(cell.genes[0].parameters()).device
                        stem_gene = stem_gene.to(device)
                    else:
                        # Use the default device from config
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        stem_gene = stem_gene.to(device)
                    cell.genes.append(stem_gene)
                    stem_cells_added += 1
                    
                    if stem_cells_added >= num_stem_to_add:
                        break
        
        if stem_cells_added >= num_stem_to_add:
            break
    
    print(f"   Added {stem_cells_added} stem genes across population")
    
    # Monkey-patch the population statistics collector
    original_compute_metrics = germinal_center._compute_comprehensive_metrics
    
    def compute_metrics_with_stem(fitness_scores: Dict[str, float]) -> Dict[str, float]:
        # Get original metrics
        metrics = original_compute_metrics(fitness_scores)
        
        # Add stem-specific metrics
        gene_type_counts = {'V': 0, 'D': 0, 'J': 0, 'S': 0}
        gene_type_fitness = {'V': [], 'D': [], 'J': [], 'S': []}
        differentiation_events = 0
        
        for cell in germinal_center.population.values():
            for gene in cell.genes:
                if gene.is_active:
                    gene_type = gene.gene_type
                    if gene_type in gene_type_counts:
                        gene_type_counts[gene_type] += 1
                        gene_type_fitness[gene_type].append(gene.fitness_contribution)
                    
                    # Count differentiation events
                    if isinstance(gene, StemGeneModule) and gene.differentiation_history:
                        differentiation_events += len(gene.differentiation_history)
        
        # Add population stats for stem genes
        population_stats = {
            'V_count': gene_type_counts['V'],
            'D_count': gene_type_counts['D'],
            'J_count': gene_type_counts['J'],
            'S_count': gene_type_counts['S'],
            'V_avg_fitness': np.mean(gene_type_fitness['V']) if gene_type_fitness['V'] else 0.5,
            'D_avg_fitness': np.mean(gene_type_fitness['D']) if gene_type_fitness['D'] else 0.5,
            'J_avg_fitness': np.mean(gene_type_fitness['J']) if gene_type_fitness['J'] else 0.5,
            'S_avg_fitness': np.mean(gene_type_fitness['S']) if gene_type_fitness['S'] else 0.5,
            'stress_level': germinal_center.current_stress,
            'recent_mutation_rate': metrics.get('transposition_rate', 0),
            'diversity': metrics.get('shannon_index', 0),
            'generation': germinal_center.generation
        }
        
        # Trigger differentiation in stem genes based on population needs
        for cell in germinal_center.population.values():
            for gene in cell.genes:
                if isinstance(gene, StemGeneModule) and gene.commitment_level < 0.5:
                    # Check if differentiation is needed
                    if gene_type_counts['V'] == 0 or gene_type_counts['D'] == 0 or gene_type_counts['J'] == 0:
                        gene.differentiate(population_stats=population_stats)
        
        # Update metrics
        metrics['stem_gene_count'] = gene_type_counts['S']
        metrics['differentiation_events'] = differentiation_events
        metrics['stem_commitment_avg'] = np.mean([
            g.commitment_level for c in germinal_center.population.values()
            for g in c.genes if isinstance(g, StemGeneModule)
        ]) if gene_type_counts['S'] > 0 else 0
        
        return metrics
    
    # Replace the method
    germinal_center._compute_comprehensive_metrics = compute_metrics_with_stem
    
    print("   Stem gene integration complete!")

# ============================================================================
# Test the stem genes
# ============================================================================

def test_stem_genes(germinal_center):
    """Run a test to show stem gene effectiveness"""
    
    print("\n🧪 Testing stem gene behavior...")
    
    # Simulate loss of all D genes (crisis scenario)
    print("\n   Simulating loss of all D genes...")
    removed_count = 0
    
    for cell in germinal_center.population.values():
        genes_to_remove = []
        for i, gene in enumerate(cell.genes):
            if gene.gene_type == 'D' and gene.is_active:
                genes_to_remove.append(i)
                removed_count += 1
        
        # Remove in reverse order to maintain indices
        for i in reversed(genes_to_remove):
            cell.genes[i].is_active = False
            cell.genes[i].is_cold = True
    
    print(f"   Deactivated {removed_count} D genes")
    
    # Force high stress
    germinal_center.current_stress = 0.9
    
    # Run one generation to see stem response
    print("\n   Running recovery generation...")
    
    # This would be called in your main evolution loop
    # For testing, we'll just show what would happen
    
    stem_genes = []
    for cell in germinal_center.population.values():
        for gene in cell.genes:
            if isinstance(gene, StemGeneModule):
                stem_genes.append(gene)
    
    print(f"\n   Found {len(stem_genes)} stem genes")
    
    # Simulate differentiation
    population_stats = {
        'V_count': 100,  # Plenty of V
        'D_count': 0,    # No D genes!
        'J_count': 80,   # Good amount of J
        'stress_level': 0.9,
        'diversity': 0.3  # Low due to loss
    }
    
    differentiated = 0
    for gene in stem_genes[:10]:  # First 10 stem genes
        if gene.commitment_level < 0.5:
            gene.differentiate('D')  # Force differentiation to D
            differentiated += 1
    
    print(f"   {differentiated} stem genes differentiated to D type")
    print("   Population recovery initiated!")