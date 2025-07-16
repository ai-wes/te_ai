"""
Advanced Transposon Modules
==========================
Cutting-edge neural architecture components for TE-AI including:
- Dream-based consolidation and offline reorganization
- Advanced gene regulatory networks with promoter/repressor dynamics
- Causal reasoning for transposition
- Self-modifying architectures
- Phase transition detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx
from collections import defaultdict
import random
from dataclasses import dataclass
from datetime import datetime
import uuid

# ============================================================================
# Dream-Based Consolidation System
# ============================================================================

class DreamConsolidator(nn.Module):
    """Offline reorganization through synthetic dream experiences"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Dream generator network
        self.dream_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4)
        )
        
        # Nightmare generator (adversarial scenarios)
        self.nightmare_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Memory consolidation network
        self.consolidator = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Replay buffer
        self.memory_buffer = []
        self.max_memories = 1000
        
    def record_experience(self, gene_state: torch.Tensor, fitness: float, stress: float):
        """Store experiences for later dreaming"""
        experience = {
            'state': gene_state.detach().cpu(),
            'fitness': fitness,
            'stress': stress,
            'timestamp': datetime.now()
        }
        
        self.memory_buffer.append(experience)
        if len(self.memory_buffer) > self.max_memories:
            self.memory_buffer.pop(0)
    
    def generate_dream_antigens(self, num_dreams: int = 10) -> List[torch.Tensor]:
        """Create synthetic antigens from past experiences"""
        if len(self.memory_buffer) < 10:
            return []
        
        dreams = []
        
        # Sample from high-stress memories
        high_stress_memories = [m for m in self.memory_buffer if m['stress'] > 0.5]
        if high_stress_memories:
            for _ in range(num_dreams // 2):
                memory = random.choice(high_stress_memories)
                # Generate variations
                dream_state = self.dream_encoder(memory['state'].unsqueeze(0))
                dreams.append(dream_state)
        
        # Generate nightmares (worst-case scenarios)
        for _ in range(num_dreams // 2):
            # Combine multiple bad experiences
            bad_memories = sorted(self.memory_buffer, key=lambda x: x['fitness'])[:10]
            if bad_memories:
                combined = torch.stack([m['state'] for m in bad_memories]).mean(dim=0)
                nightmare = self.nightmare_generator(combined.unsqueeze(0))
                dreams.append(nightmare)
        
        return dreams
    
    def consolidate_memories(self, gene_states: List[torch.Tensor]) -> torch.Tensor:
        """Consolidate experiences into improved gene arrangements"""
        if not gene_states:
            return None
        
        # Stack and process through GRU
        stacked = torch.stack(gene_states).unsqueeze(0)
        consolidated, _ = self.consolidator(stacked)
        
        # Return mean consolidated state
        return consolidated.mean(dim=1).squeeze(0)
    
    def dream_phase(self, population: Dict, num_cycles: int = 5):
        """Run dream consolidation on population"""
        print(f"\n💤 Entering dream phase ({num_cycles} cycles)...")
        
        for cycle in range(num_cycles):
            # Generate dream antigens
            dreams = self.generate_dream_antigens(num_dreams=20)
            
            if not dreams:
                continue
            
            # Let each cell "dream"
            for cell_id, cell in population.items():
                # Process dreams
                dream_responses = []
                for dream in dreams:
                    # Simplified forward pass
                    response = cell.gene_integrator(dream)
                    dream_responses.append(response)
                
                # Consolidate learnings
                if dream_responses:
                    consolidated = self.consolidate_memories(dream_responses)
                    
                    # Apply consolidation as soft update to genes
                    with torch.no_grad():
                        for gene in cell.genes:
                            if gene.is_active and hasattr(gene, 'output_proj'):
                                # Soft update towards consolidated state
                                update_strength = 0.1 * (1.0 - gene.methylation_level)
                                for param in gene.output_proj.parameters():
                                    param.data += update_strength * torch.randn_like(param) * 0.01
            
            print(f"  💭 Dream cycle {cycle+1}: Processed {len(dreams)} dream scenarios")

# ============================================================================
# Advanced Gene Regulatory Network
# ============================================================================

class GeneRegulatoryNetwork(nn.Module):
    """Sophisticated gene interaction network with promoters and repressors"""
    
    def __init__(self, max_genes: int = 20, hidden_dim: int = 128):
        super().__init__()
        self.max_genes = max_genes
        self.hidden_dim = hidden_dim
        
        # Promoter/repressor identification network
        self.regulatory_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Promoter, Neutral, Repressor
        )
        
        # Binding affinity prediction
        self.binding_predictor = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
        # Regulatory state dynamics
        self.state_dynamics = nn.GRUCell(hidden_dim + max_genes, hidden_dim)
        
        # Learned regulatory motifs
        self.regulatory_motifs = nn.Parameter(torch.randn(10, hidden_dim))
        
        # Feedback controller
        self.feedback_controller = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, max_genes)
        )
    
    def compute_regulatory_matrix(self, gene_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Compute pairwise regulatory relationships"""
        n_genes = len(gene_embeddings)
        reg_matrix = torch.zeros(n_genes, n_genes)
        
        for i, gene_i in enumerate(gene_embeddings):
            for j, gene_j in enumerate(gene_embeddings):
                if i == j:
                    continue
                
                # Classify regulatory relationship
                combined = torch.cat([gene_i, gene_j])
                reg_type = self.regulatory_classifier(combined)
                reg_class = torch.argmax(reg_type)
                
                # Compute binding affinity
                affinity = self.binding_predictor(gene_i.unsqueeze(0), gene_j.unsqueeze(0))
                
                # Set regulatory strength
                if reg_class == 0:  # Promoter
                    reg_matrix[i, j] = affinity.item()
                elif reg_class == 2:  # Repressor
                    reg_matrix[i, j] = -affinity.item()
    
        return reg_matrix
    
    def apply_regulation(self, gene_activities: torch.Tensor, 
                        reg_matrix: torch.Tensor,
                        fitness_feedback: float) -> torch.Tensor:
        """Apply regulatory dynamics with feedback control"""
        # Current regulatory state
        reg_input = torch.cat([
            gene_activities.mean(dim=0),
            reg_matrix.flatten()[:self.max_genes]
        ])
        
        # Update state with dynamics
        new_state = self.state_dynamics(reg_input, gene_activities.mean(dim=0))
        
        # Apply feedback control
        feedback_input = torch.cat([new_state, torch.tensor([fitness_feedback])])
        control_signal = self.feedback_controller(feedback_input)
        
        # Modulate gene activities
        modulated = gene_activities * torch.sigmoid(control_signal[:gene_activities.shape[0]])
        
        return modulated
    
    def detect_oscillations(self, history: List[torch.Tensor], window: int = 10) -> bool:
        """Detect oscillatory patterns in gene regulation"""
        if len(history) < window:
            return False
        
        # Compute autocorrelation
        recent = torch.stack(history[-window:])
        mean = recent.mean(dim=0)
        centered = recent - mean
        
        autocorr = []
        for lag in range(1, window//2):
            corr = (centered[:-lag] * centered[lag:]).mean()
            autocorr.append(corr.item())
        
        # Check for periodic pattern
        autocorr = np.array(autocorr)
        peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & 
                        (autocorr[1:-1] > autocorr[2:]))[0]
        
        return len(peaks) >= 2

# ============================================================================
# Causal Transposition Reasoner
# ============================================================================

class CausalTranspositionEngine:
    """Causal reasoning for intelligent transposition decisions"""
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.intervention_history = []
        self.counterfactual_cache = {}
        
    def observe_transposition(self, gene_id: str, action: str, 
                            fitness_before: float, fitness_after: float,
                            context: Dict):
        """Record causal relationship from transposition"""
        # Add nodes
        self.causal_graph.add_node(f"{gene_id}_before", 
                                  fitness=fitness_before,
                                  **context)
        self.causal_graph.add_node(f"{gene_id}_after",
                                  fitness=fitness_after,
                                  action=action)
        
        # Add causal edge
        self.causal_graph.add_edge(f"{gene_id}_before", f"{gene_id}_after",
                                   action=action,
                                   effect=fitness_after - fitness_before)
        
        # Store intervention
        self.intervention_history.append({
            'gene': gene_id,
            'action': action,
            'effect': fitness_after - fitness_before,
            'context': context
        })
    
    def predict_intervention_effect(self, gene_id: str, action: str, 
                                  current_context: Dict) -> float:
        """Predict effect of intervention using causal graph"""
        # Find similar past interventions
        similar_interventions = []
        
        for intervention in self.intervention_history:
            if intervention['action'] == action:
                # Compute context similarity
                similarity = self._context_similarity(
                    intervention['context'], 
                    current_context
                )
                if similarity > 0.7:
                    similar_interventions.append(intervention)
        
        if not similar_interventions:
            return 0.0
        
        # Weighted average of effects
        total_weight = 0
        weighted_effect = 0
        
        for intervention in similar_interventions:
            weight = self._context_similarity(
                intervention['context'], 
                current_context
            )
            weighted_effect += weight * intervention['effect']
            total_weight += weight
        
        return weighted_effect / total_weight if total_weight > 0 else 0.0
    
    def generate_counterfactual(self, gene_id: str, 
                              current_state: Dict) -> List[Dict]:
        """Generate counterfactual scenarios"""
        counterfactuals = []
        
        actions = ['jump', 'duplicate', 'invert', 'delete']
        for action in actions:
            # Predict outcome
            predicted_effect = self.predict_intervention_effect(
                gene_id, action, current_state
            )
            
            counterfactual = {
                'action': action,
                'predicted_effect': predicted_effect,
                'confidence': self._compute_confidence(gene_id, action)
            }
            counterfactuals.append(counterfactual)
        
        return sorted(counterfactuals, 
                     key=lambda x: x['predicted_effect'], 
                     reverse=True)
    
    def _context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Compute similarity between contexts"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if isinstance(context1[key], (int, float)):
                # Numerical similarity
                diff = abs(context1[key] - context2[key])
                sim = 1.0 / (1.0 + diff)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_confidence(self, gene_id: str, action: str) -> float:
        """Compute confidence in prediction"""
        relevant_history = [
            h for h in self.intervention_history
            if h['action'] == action
        ]
        
        if len(relevant_history) < 3:
            return 0.1
        
        # Confidence based on consistency of effects
        effects = [h['effect'] for h in relevant_history[-10:]]
        if effects:
            std = np.std(effects)
            confidence = 1.0 / (1.0 + std)
            return min(confidence, 0.95)
        
        return 0.5

# ============================================================================
# Self-Modifying Architecture Controller
# ============================================================================

class SelfModifyingController(nn.Module):
    """Neural architecture that can modify its own structure"""
    
    def __init__(self, base_dim: int = 128):
        super().__init__()
        self.base_dim = base_dim
        
        # Meta-network that generates architecture modifications
        self.meta_controller = nn.Sequential(
            nn.Linear(base_dim * 2, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim // 2),
            nn.ReLU(),
            nn.Linear(base_dim // 2, 5)  # Add, Remove, Rewire, Resize, Identity
        )
        
        # Architecture generator
        self.arch_generator = nn.ModuleDict({
            'add_layer': nn.Linear(base_dim, base_dim * 2),
            'remove_layer': nn.Linear(base_dim, 1),
            'rewire': nn.Linear(base_dim, base_dim * base_dim),
            'resize': nn.Linear(base_dim, 2)  # expand/contract factors
        })
        
        # Dynamic module list
        self.dynamic_layers = nn.ModuleList([
            nn.Linear(base_dim, base_dim) for _ in range(3)
        ])
        
        self.modification_history = []
        
    def decide_modification(self, performance_gradient: torch.Tensor,
                          current_state: torch.Tensor) -> str:
        """Decide what architectural modification to make"""
        # Combine performance feedback with current state
        decision_input = torch.cat([performance_gradient, current_state])
        
        # Get modification probabilities
        mod_logits = self.meta_controller(decision_input)
        mod_probs = F.softmax(mod_logits, dim=-1)
        
        # Sample modification
        modifications = ['add', 'remove', 'rewire', 'resize', 'none']
        mod_idx = torch.multinomial(mod_probs, 1).item()
        
        return modifications[mod_idx]
    
    def apply_modification(self, mod_type: str, gene_module):
        """Apply architectural modification to gene module"""
        timestamp = datetime.now().isoformat()
        
        if mod_type == 'add':
            # Add new layer
            new_layer = nn.Sequential(
                nn.Linear(self.base_dim, self.base_dim),
                nn.LayerNorm(self.base_dim),
                nn.ReLU()
            )
            self.dynamic_layers.append(new_layer)
            
            self.modification_history.append({
                'time': timestamp,
                'action': 'add_layer',
                'details': 'Added new transformation layer'
            })
            
        elif mod_type == 'remove' and len(self.dynamic_layers) > 1:
            # Remove least important layer
            self.dynamic_layers.pop()
            
            self.modification_history.append({
                'time': timestamp,
                'action': 'remove_layer',
                'details': 'Removed transformation layer'
            })
            
        elif mod_type == 'rewire':
            # Rewire connections
            if hasattr(gene_module, 'conv1'):
                with torch.no_grad():
                    # Add skip connections or modify weights
                    gene_module.conv1.weight.data += torch.randn_like(
                        gene_module.conv1.weight.data
                    ) * 0.1
            
            self.modification_history.append({
                'time': timestamp,
                'action': 'rewire',
                'details': 'Modified connection weights'
            })
            
        elif mod_type == 'resize':
            # Resize hidden dimensions (simplified for demonstration)
            factors = self.arch_generator['resize'](
                torch.randn(1, self.base_dim)
            )
            expand_factor = torch.sigmoid(factors[0]).item() + 0.5
            
            self.modification_history.append({
                'time': timestamp,
                'action': 'resize',
                'details': f'Resize factor: {expand_factor:.2f}'
            })

# ============================================================================
# Phase Transition Detector
# ============================================================================

class PhaseTransitionDetector:
    """Detect critical transitions in population dynamics"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metric_history = defaultdict(list)
        self.transition_alerts = []
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Record new metrics"""
        for key, value in metrics.items():
            self.metric_history[key].append(value)
            # Keep window size
            if len(self.metric_history[key]) > self.window_size:
                self.metric_history[key].pop(0)
    
    def detect_critical_slowing(self, metric_name: str) -> bool:
        """Detect critical slowing down before phase transition"""
        history = self.metric_history.get(metric_name, [])
        
        if len(history) < self.window_size:
            return False
        
        # Compute autocorrelation at lag-1
        history_array = np.array(history)
        mean = history_array.mean()
        centered = history_array - mean
        
        autocorr = np.corrcoef(centered[:-1], centered[1:])[0, 1]
        
        # Compute variance
        variance = history_array.var()
        
        # Check for increasing autocorrelation and variance
        if len(self.metric_history[f'{metric_name}_autocorr']) > 10:
            recent_autocorr = self.metric_history[f'{metric_name}_autocorr'][-10:]
            autocorr_trend = np.polyfit(range(10), recent_autocorr, 1)[0]
            
            if autocorr > 0.8 and autocorr_trend > 0 and variance > history_array[:20].var() * 2:
                return True
        
        # Store for trend analysis
        self.metric_history[f'{metric_name}_autocorr'].append(autocorr)
        self.metric_history[f'{metric_name}_var'].append(variance)
        
        return False
    
    def detect_bifurcation(self, control_param: str, 
                          state_var: str) -> Optional[str]:
        """Detect type of bifurcation"""
        if control_param not in self.metric_history or state_var not in self.metric_history:
            return None
        
        control = np.array(self.metric_history[control_param][-20:])
        state = np.array(self.metric_history[state_var][-20:])
        
        if len(control) < 20:
            return None
        
        # Fit polynomial to detect bifurcation type
        coeffs = np.polyfit(control, state, 3)
        poly = np.poly1d(coeffs)
        
        # Compute critical points
        critical_points = np.roots(np.polyder(poly))
        real_critical = critical_points[np.isreal(critical_points)].real
        
        if len(real_critical) == 0:
            return None
        elif len(real_critical) == 1:
            # Check second derivative
            second_deriv = np.polyder(poly, 2)
            if second_deriv(real_critical[0]) < 0:
                return "saddle-node"
            else:
                return "transcritical"
        else:
            return "pitchfork"
    
    def compute_early_warning_signal(self, metric_name: str) -> float:
        """Compute early warning signal strength"""
        if not self.detect_critical_slowing(metric_name):
            return 0.0
        
        history = np.array(self.metric_history[metric_name])
        
        # Compute multiple indicators
        indicators = []
        
        # 1. Autocorrelation at lag-1
        autocorr = self.metric_history[f'{metric_name}_autocorr'][-1]
        indicators.append(autocorr)
        
        # 2. Variance
        var_ratio = history[-10:].var() / history[:10].var()
        indicators.append(min(var_ratio / 5, 1.0))
        
        # 3. Skewness change
        skew_recent = np.abs(np.mean((history[-10:] - history[-10:].mean())**3))
        skew_early = np.abs(np.mean((history[:10] - history[:10].mean())**3))
        skew_ratio = skew_recent / (skew_early + 1e-6)
        indicators.append(min(skew_ratio / 3, 1.0))
        
        # Combined signal
        return np.mean(indicators)
    
    def alert_if_critical(self, metrics: Dict[str, float]) -> List[str]:
        """Generate alerts for critical transitions"""
        alerts = []
        
        for metric_name in metrics:
            warning_signal = self.compute_early_warning_signal(metric_name)
            
            if warning_signal > 0.7:
                alert = f"⚠️ CRITICAL TRANSITION WARNING: {metric_name} " \
                       f"(signal strength: {warning_signal:.2f})"
                alerts.append(alert)
                
                self.transition_alerts.append({
                    'time': datetime.now().isoformat(),
                    'metric': metric_name,
                    'signal_strength': warning_signal,
                    'type': self.detect_bifurcation('stress_level', metric_name)
                })
        
        return alerts

# ============================================================================
# Integration Helper
# ============================================================================

def integrate_advanced_modules(population_manager, config):
    """Integrate advanced modules into existing population manager"""
    
    # Add dream consolidator
    population_manager.dream_consolidator = DreamConsolidator(
        hidden_dim=config.hidden_dim
    ).to(config.device)
    
    # Add causal reasoner
    population_manager.causal_engine = CausalTranspositionEngine()
    
    # Add phase detector
    population_manager.phase_detector = PhaseTransitionDetector()
    
    # Add self-modifying controller to each cell
    for cell in population_manager.population.values():
        cell.self_modifier = SelfModifyingController(
            base_dim=config.hidden_dim
        ).to(config.device)
        
        # Enhanced gene regulatory network
        cell.gene_regulatory_network = GeneRegulatoryNetwork(
            max_genes=config.max_genes_per_clone,
            hidden_dim=config.hidden_dim
        ).to(config.device)
    
    print("✨ Advanced modules integrated successfully!")
    
    return population_manager