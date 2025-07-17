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
from scripts.config import cfg
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.utils.telemetry import TermColors

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ============================================================================# ============================================================================
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
        
        self._cache = {}
        self._cache_generation = -1

    
    def _compute_early_warning_indicators(self, metrics: Dict, population_state: Dict):
        """Compute all early warning indicators"""
    
    
        numeric_metrics_for_autocorr = [
            'mean_fitness', 'fitness_variance', 'shannon_index', 'gene_richness'
        ]
        
        # 1. Autocorrelation at lag-1 (only on numeric metrics)
        for metric_name in numeric_metrics_for_autocorr:
            if metric_name in self.metric_history:
                values = self.metric_history[metric_name]
                if len(values) >= 10:
                    values_array = np.array(list(values), dtype=np.float64) # Ensure float type
                    if values_array.std() > 1e-9: # Use a small epsilon for stability
                        autocorr = np.corrcoef(values_array[:-1], values_array[1:])[0, 1]
                        # We can average the autocorrelation of all numeric signals
                        # For simplicity, we'll just use the fitness autocorrelation for now.
                        if metric_name == 'mean_fitness':
                            self.indicators['autocorrelation'].append(autocorr)

        # 2. Variance (this part was already safe as it explicitly checks for 'fitness')
        if 'mean_fitness' in self.metric_history: # Check mean_fitness for variance trend
            recent_fitness = list(self.metric_history['mean_fitness'])[-20:]
            if len(recent_fitness) >= 10:
                variance = np.var(recent_fitness)
                self.indicators['variance'].append(variance)
        
        # 3. Skewness (this is safe)
        if 'fitness_distribution' in population_state:
            fitness_dist = population_state['fitness_distribution']
            if len(fitness_dist) > 1:
                skewness = stats.skew(fitness_dist)
                self.indicators['skewness'].append(skewness)
        
        # 4. Spatial correlation (this is safe)
        if 'gene_positions' in population_state:
            positions = population_state['gene_positions']
            if len(positions) > 10:
                spatial_corr = self._compute_morans_i(positions)
                self.indicators['spatial_correlation'].append(spatial_corr)
        
        # 5. Recovery rate from perturbations (this is safe)
        if 'perturbation_response' in metrics:
            recovery_rate = metrics['perturbation_response']
            self.indicators['recovery_rate'].append(recovery_rate)            
            
            
    
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
        num_new = min(50, cfg.max_population - len(population_manager.population))
        population_manager._add_random_individuals(num_new)
        
        return True
    
    def _stabilization_intervention(self, population_manager) -> bool:
        """Intervention to stabilize population"""
        print("   🛡️ Intervention: Stabilizing population")
        
        # Reduce mutation rate temporarily
        original_mutation = cfg.mutation_rate
        cfg.mutation_rate *= 0.1
        
        # Increase selection pressure
        original_selection = cfg.selection_pressure
        cfg.selection_pressure *= 1.5
        
        # Schedule restoration
        population_manager.scheduled_tasks.append({
            'generation': population_manager.generation + 10,
            'action': lambda: setattr(cfg, 'mutation_rate', original_mutation)
        })
        population_manager.scheduled_tasks.append({
            'generation': population_manager.generation + 10,
            'action': lambda: setattr(cfg, 'selection_pressure', original_selection)
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
        
        # ============================================================================
        # ADDED PRINT STATEMENT
        # ============================================================================
        print(f"   {TermColors.BOLD}{TermColors.BRIGHT_YELLOW}[Gen {population_manager.generation}] Saving {elite_count} elite cells from collapse.{TermColors.RESET}")
        # ============================================================================
        
        # Create new diverse population
        new_population = {}
        
        # Keep elite
        for elite_id in elite_ids:
            if elite_id in population_manager.population:
                new_population[elite_id] = population_manager.population[elite_id]
        
        # Generate diverse new individuals
        while len(new_population) < cfg.initial_population:
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



    def update(self, metrics: Dict[str, float], population_state: Dict):
        """Cached update that reuses expensive computations"""
        generation = metrics.get('generation', self._cache_generation + 1)
        
        # Only recompute if generation changed
        if generation != self._cache_generation:
            self._cache = {}
            self._cache_generation = generation
        
        # Use cached indicators if available
        cache_key = f"indicators_{generation}"
        if cache_key not in self._cache:
            self._compute_early_warning_indicators(metrics, population_state)
            self._cache[cache_key] = dict(self.indicators)
        else:
            # Restore from cache
            for key, values in self._cache[cache_key].items():
                self.indicators[key] = deque(values, maxlen=self.window_size)
        
        # Detect phase and handle transitions
        new_phase = self._detect_phase_state()
        if new_phase != self.current_phase:
            self._record_transition(self.current_phase, new_phase, metrics)
            self.current_phase = new_phase
            # Optionally trigger intervention
            intervention = self.phase_states[new_phase]['intervention']
            if intervention and hasattr(self, 'population_manager'):
                self.intervention_strategies[intervention](self.population_manager)
        return new_phase
    
    def _compute_morans_i(self, positions: List[Tuple[float, float]]) -> float:
        """
        Optimized Moran's I using matrix operations
        """
        if len(positions) < 3:
            return 0.0
        
        # Sample efficiently
        n = min(len(positions), 100)
        if len(positions) > n:
            indices = np.random.choice(len(positions), n, replace=False)
            positions = np.array(positions)[indices]
        else:
            positions = np.array(positions)
        
        # Vectorized distance computation
        # Use broadcasting for efficient pairwise distances
        pos_expanded = positions[:, np.newaxis, :]  # Shape: (n, 1, 2)
        distances = np.linalg.norm(pos_expanded - positions, axis=2)  # Shape: (n, n)
        
        # Vectorized weight matrix
        W = 1.0 / (1.0 + distances)
        np.fill_diagonal(W, 0)  # No self-connections
        
        # Normalize
        W_sum = W.sum()
        if W_sum == 0:
            return 0.0
        W = W / W_sum
        
        # Vectorized Moran's I computation
        values = positions[:, 0]
        mean_val = values.mean()
        deviations = values - mean_val
        
        # Matrix computation
        numerator = np.sum(W * np.outer(deviations, deviations))
        denominator = np.sum(deviations ** 2)
        
        if denominator == 0:
            return 0.0
        
        return (n / W_sum) * (numerator / denominator)


