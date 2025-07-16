# File: stem_gene_prototype.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from typing import List, Dict, Optional, Tuple





from scripts.depricated.transposable_immune_ai_production_complete import (
    ContinuousDepthGeneModule, cfg
)

class StemGeneModule(ContinuousDepthGeneModule):
    """
    Enhanced stem gene for the viral recognition system
    Features:
    - Asymmetric division (self-renewal + differentiation)
    - Stochastic/noise-infused differentiation  
    - RL-based decision making
    - Hybrid/multi-lineage support
    - Niche/morphogen modeling
    - Error correction/apoptosis
    """
    
    def __init__(self, gene_types: Optional[List[str]] = None):
        # Expandable gene types (defaults to V, D, J, S)
        self.gene_types = gene_types or ['V', 'D', 'J', 'S']
        
        # Initialize as stem type (last gene type in the list)
        stem_type = self.gene_types[-1]  # Last type is always stem (S or TS)
        super().__init__(stem_type, 0)
        
        # Differentiation state: dynamic size based on gene_types
        self.differentiation_state = nn.Parameter(torch.zeros(len(self.gene_types)))
        self.differentiation_state.data[-1] = 1.0  # Start as pure stem (last type is stem)
        
        # Track differentiation history
        self.differentiation_history = []
        self.commitment_level = 0.0
        self.plasticity = nn.Parameter(torch.tensor(1.0))
        self.is_active = True  # For apoptosis
        
        # Morphogen field for niche modeling
        self.morphogen_field = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.1)
        
        # Enhanced population sensing with morphogen input
        sensor_input_dim = 12 + cfg.hidden_dim  # Stats + morphogen field
        self.population_sensor = nn.Sequential(
            nn.Linear(sensor_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add regularization
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, len(self.gene_types)),
            nn.Softmax(dim=-1)
        )
        
        # RL components for adaptive decision making
        self.policy_net = nn.Linear(sensor_input_dim, len(self.gene_types))
        self.value_net = nn.Linear(sensor_input_dim, 1)
        self.rl_optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), 
            lr=1e-4
        )
        self.rl_memory = []  # Store experiences
        
        # Dynamic morphable components for each gene type
        self.gene_components = nn.ModuleDict({
            gene_type: nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            ) for gene_type in self.gene_types
        })
        
        # Position preference based on differentiation
        self.position_preferences = {
            'V': (0.15, 0.1),  # mean, std for position
            'D': (0.45, 0.1),
            'J': (0.8, 0.1),
            'S': (0.5, 0.2)    # Stem cells can go anywhere
        }
    
    def sense_population_needs(self, population_stats: Dict, use_rl: bool = False) -> torch.Tensor:
        """Enhanced population sensing with morphogen field integration"""
        # Extract population statistics dynamically
        gene_counts = {}
        gene_performance = {}
        total_genes = 0
        
        for gene_type in self.gene_types[:-1]:  # Exclude stem
            count = population_stats.get(f'{gene_type}_count', 0)
            gene_counts[gene_type] = count
            total_genes += count
            gene_performance[gene_type] = population_stats.get(f'{gene_type}_avg_fitness', 0.5)
        
        # Population stress indicators
        stress_level = population_stats.get('stress_level', 0.0)
        mutation_rate = population_stats.get('recent_mutation_rate', 0.0)
        
        # Create feature vector
        features = []
        for gene_type in self.gene_types[:-1]:
            features.extend([
                gene_counts.get(gene_type, 0) / max(total_genes, 1),
                gene_performance.get(gene_type, 0.5),
                float(gene_counts.get(gene_type, 0) == 0)  # Missing indicator
            ])
        
        features.extend([
            stress_level,
            mutation_rate,
            population_stats.get('diversity', 0.5)
        ])
        
        # Pad to expected size (12 features)
        while len(features) < 12:
            features.append(0.0)
        features = features[:12]  # Truncate if too many
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.population_sensor.parameters().__next__().device)
        
        # Concatenate with morphogen field
        features_with_morphogen = torch.cat([features_tensor, self.morphogen_field.detach()])
        
        if use_rl:
            # Use policy network for RL-based decision
            return F.softmax(self.policy_net(features_with_morphogen), dim=-1)
        else:
            # Use standard population sensor
            return self.population_sensor(features_with_morphogen)
    
    def differentiate(self, target_type: Optional[str] = None, 
                     population_stats: Optional[Dict] = None,
                     use_rl: bool = False):
        """Enhanced differentiation with noise and RL support"""
        
        if target_type:
            # Directed differentiation
            if target_type not in self.gene_types:
                return  # Invalid type
                
            type_idx = self.gene_types.index(target_type)
            
            # Gradual differentiation with noise
            with torch.no_grad():
                current_stem = self.differentiation_state[-1].item()
                commitment = min(0.9, self.commitment_level + 0.3)
                
                # Add biological noise for stochasticity (ensure same device)
                noise = torch.randn_like(self.differentiation_state).to(self.differentiation_state.device) * 0.1 * (1 - self.commitment_level)
                
                self.differentiation_state.data = torch.zeros(len(self.gene_types)).to(self.differentiation_state.device)
                self.differentiation_state.data[type_idx] = commitment
                self.differentiation_state.data[-1] = 1 - commitment  # Stem is last
                
                # Apply noise and renormalize
                self.differentiation_state.data += noise
                self.differentiation_state.data = F.softmax(self.differentiation_state.data, dim=0)
                
                self.commitment_level = commitment
                
                # Update position based on differentiation
                if target_type != self.gene_types[-1]:  # Not stem
                    mean, std = self.position_preferences.get(target_type, (0.5, 0.2))
                    self.position = np.clip(np.random.normal(mean, std), 0, 1)
                    self.gene_type = target_type
                
                print(f"   🎯 Stem gene differentiating: {target_type} (commitment: {commitment:.2f})")        
        elif population_stats:
            # Context-based differentiation
            needs = self.sense_population_needs(population_stats, use_rl=use_rl)
            
            # Store experience for RL
            if use_rl:
                features = self._extract_features(population_stats)
                self.rl_memory.append({
                    'features': features,
                    'needs': needs.detach()
                })
            
            # Stochastic differentiation based on needs
            if random.random() < self.plasticity.item():
                # Choose based on population needs
                chosen_idx = torch.multinomial(needs, 1).item()
                target_type = self.gene_types[chosen_idx]
                
                if target_type != self.gene_types[-1]:  # Don't differentiate to stem
                    self.differentiate(target_type)
                    
                    # Learn from outcome if using RL
                    if use_rl and len(self.rl_memory) > 1:
                        self._update_rl_policy(population_stats)
        
        # Record differentiation event
        self.differentiation_history.append({
            'generation': population_stats.get('generation', 0) if population_stats else 0,
            'target': target_type,
            'commitment': self.commitment_level,
            'trigger': 'directed' if target_type else 'population_need'
        })
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced forward pass with error correction"""
        
        # Error correction - check for invalid states
        if torch.isnan(self.differentiation_state).any() or self.commitment_level > 1.0:
            self.is_active = False
            print(f"   ⚠️ Gene {self.gene_id} undergoing apoptosis due to invalid state")
            return torch.zeros_like(x)
        
        # Base input projection
        h = self.input_projection(x)
        
        # Apply differentiation-weighted processing
        diff_probs = F.softmax(self.differentiation_state, dim=0)
        
        # Combine different gene type processings dynamically
        h_combined = torch.zeros_like(h)
        for i, gene_type in enumerate(self.gene_types):
            if gene_type in self.gene_components:
                h_combined += diff_probs[i] * self.gene_components[gene_type](h)
        
        # Apply epigenetic regulation (inherited from parent class)
        h_regulated = self._apply_epigenetic_regulation(h_combined)
        
        # Integrate morphogen field influence
        morphogen_gate = torch.sigmoid(self.morphogen_field[:h_regulated.size(-1)])
        h_regulated = h_regulated * morphogen_gate
        
        # Continue with parent processing
        return super().forward(x, edge_index, batch)
    
    def transpose(self, stress_level: float, population_diversity: float) -> Tuple[Optional['StemGeneModule'], Optional[str]]:
        """Enhanced stem-specific transposition with asymmetric division"""
        
        # Stem cells are more mobile when undifferentiated
        mobility_factor = 2.0 - self.commitment_level
        
        # Check if we should perform asymmetric division under high stress
        if stress_level > 0.8 and self.commitment_level < 0.3:
            if random.random() < stress_level * 0.5:
                # Try asymmetric division
                population_stats = {
                    'stress_level': stress_level,
                    'diversity': population_diversity,
                    'generation': getattr(self, 'current_generation', 0)
                }
                daughter = self.divide_asymmetrically(population_stats)
                if daughter:
                    return daughter, 'asymmetric_division'
        
        # Check if we should differentiate instead of transpose
        elif stress_level > 0.7 and self.commitment_level < 0.5:
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
                    self.differentiation_state.data = torch.zeros(len(self.gene_types)).to(self.differentiation_state.device)
                    self.differentiation_state.data[-1] = 1.0  # Stem is last
                    self.commitment_level = 0.0
                    self.plasticity.data = torch.tensor(1.0)
                    self.gene_type = self.gene_types[-1]
                    self.position = 0.5  # Return to neutral position
                    
                print(f"   🔄 Gene {self.gene_id} de-differentiated to stem state!")
                
                return True
        return False
    
    def divide_asymmetrically(self, population_stats: Dict) -> Optional['StemGeneModule']:
        """Create a differentiated daughter while maintaining self as stem"""
        if random.random() < self.plasticity.item() * 0.5:  # Probability based on plasticity
            daughter = copy.deepcopy(self)
            daughter.gene_id = f"{self.gene_id}_daughter_{len(self.differentiation_history)}"
            daughter.differentiate(population_stats=population_stats)  # Daughter differentiates
            daughter.commitment_level = min(1.0, daughter.commitment_level + 0.2)  # Accelerate commitment
            daughter.plasticity.data *= 0.8  # Reduce daughter's plasticity
            
            # Slightly modify daughter's morphogen field
            daughter.morphogen_field.data += torch.randn_like(daughter.morphogen_field) * 0.1
            
            print(f"   🔄 Asymmetric division: Stem created {daughter.gene_type} daughter")
            return daughter
        return None
    
    def update_morphogen(self, neighbor_fields: List[torch.Tensor]):
        """Update morphogen field based on neighbors (niche modeling)"""
        if neighbor_fields:
            avg_neighbor = torch.mean(torch.stack(neighbor_fields), dim=0)
            self.morphogen_field.data = 0.9 * self.morphogen_field.data + 0.1 * avg_neighbor
    
    def add_new_type(self, new_type: str):
        """Dynamically add support for new gene type"""
        if new_type not in self.gene_types:
            self.gene_types.append(new_type)
            # Expand differentiation state
            new_state = torch.zeros(len(self.gene_types))
            new_state[:-1] = self.differentiation_state.data
            new_state[-1] = 0  # New type starts at 0
            self.differentiation_state = nn.Parameter(new_state)
            # Add new component
            self.gene_components[new_type] = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            )
            print(f"   ➕ Added new gene type: {new_type}")
    
    def _extract_features(self, population_stats: Dict) -> torch.Tensor:
        """Extract features for RL from population stats"""
        features = []
        for gene_type in self.gene_types[:-1]:
            features.append(population_stats.get(f'{gene_type}_count', 0))
            features.append(population_stats.get(f'{gene_type}_avg_fitness', 0.5))
        features.extend([
            population_stats.get('stress_level', 0.0),
            population_stats.get('diversity', 0.5)
        ])
        # Pad to consistent size
        while len(features) < 12:
            features.append(0.0)
        return torch.tensor(features[:12], dtype=torch.float32)
    
    def _update_rl_policy(self, population_stats: Dict):
        """Update RL policy based on outcomes"""
        if len(self.rl_memory) < 2:
            return
            
        # Simple reward: improvement in population fitness
        current_fitness = population_stats.get('avg_fitness', 0.5)
        prev_memory = self.rl_memory[-2]
        reward = current_fitness - prev_memory.get('fitness', 0.5)
        
        # A2C update
        features = self._extract_features(population_stats)
        features_with_morphogen = torch.cat([features, self.morphogen_field.detach()])
        
        value = self.value_net(features_with_morphogen)
        prev_features = prev_memory['features']
        prev_features_with_morphogen = torch.cat([prev_features, self.morphogen_field.detach()])
        prev_value = self.value_net(prev_features_with_morphogen)
        
        advantage = reward + 0.99 * value - prev_value
        
        # Get action probabilities
        action_probs = F.softmax(self.policy_net(prev_features_with_morphogen), dim=-1)
        chosen_action = prev_memory.get('chosen_action', 0)
        
        # Compute losses
        policy_loss = -torch.log(action_probs[chosen_action] + 1e-8) * advantage.detach()
        value_loss = advantage.pow(2)
        
        # Update
        loss = policy_loss + 0.5 * value_loss
        self.rl_optimizer.zero_grad()
        loss.backward()
        self.rl_optimizer.step()
        
        # Store fitness for next update
        self.rl_memory[-1]['fitness'] = current_fitness
    
    def guided_differentiation(self, patient_state: dict, population_state: dict):
        """Guided differentiation based on patient and population state"""
        # Combine states for decision
        combined_stats = {**population_state, **patient_state}
        
        # Use RL-based differentiation if enabled
        if hasattr(self, 'use_rl_stems') and self.use_rl_stems:
            self.differentiate(population_stats=combined_stats, use_rl=True)
        else:
            self.differentiate(population_stats=combined_stats)






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
            self.differentiation_state.data = torch.zeros(len(self.gene_types)).to(self.differentiation_state.device)
            
            # Don't fully commit - maintain some plasticity
            differentiation_strength = min(0.8, self.confidence_scores.get(learned_type, 0.5))
            
            # Create type map dynamically based on current gene types
            type_map = {gene_type: i for i, gene_type in enumerate(self.gene_types)}
            
            if learned_type in type_map:
                self.differentiation_state.data[type_map[learned_type]] = differentiation_strength
                self.differentiation_state.data[-1] = 1 - differentiation_strength  # Remain partly stem
                
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
            self.differentiation_state.data = torch.zeros(len(self.gene_types)).to(self.differentiation_state.device)
            type_map = {gene_type: i for i, gene_type in enumerate(self.gene_types)}
            
            # 40% each from teachers, 20% stem
            if type1 in type_map:
                self.differentiation_state.data[type_map[type1]] = 0.4
            if type2 in type_map:
                self.differentiation_state.data[type_map[type2]] = 0.4
            self.differentiation_state.data[-1] = 0.2  # Stem is last
            
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
                    # Create stem gene with default gene types
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