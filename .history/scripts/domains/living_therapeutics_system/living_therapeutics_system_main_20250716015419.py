#!/usr/bin/env python3
"""
Living Therapeutic System - Main Orchestrator
==============================================
Main system class that coordinates all therapeutic components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque, defaultdict
import uuid
import random
import time
import json
import os
from torch_geometric.data import Data, Batch
from datetime import datetime

# Import configuration - use main config for core system compatibility
from config import cfg as main_cfg
from .living_therapeutics_system_config import THERAPY_CFG as cfg

# Import optimized simulation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fast_optimized_te_ai import run_optimized_simulation

# Import genes
from .living_therapeutics_system_genes import (
    TherapeuticSeedGene,
    TherapeuticStemGene,
    BiosensorGene,
    TherapeuticEffectorGene,
    AdaptiveControllerGene
)

# Import parent modules from scripts directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from transposable_immune_ai_production_complete import (
    QuantumGeneModule, 
    ProductionGerminalCenter,
    ProductionBCell,
    ContinuousDepthGeneModule,
    generate_realistic_antigen,
    run_production_simulation
)
from stem_gene_module import StemGeneModule
from fast_optimized_te_ai import OptimizedBatchEvaluator, OptimizedProductionGerminalCenter

# ============================================================================
# THERAPEUTIC DOMAIN ADAPTER
# ============================================================================

class TherapeuticDomainAdapter:
    """Adapts patient data to work with core TE-AI antigen system"""
    
    @staticmethod
    def patient_to_antigen(patient_state: Dict) -> Data:
        """Convert patient biomarkers to antigen format for TE-AI processing"""
        # Use the core system's antigen generation with therapeutic features
        biomarkers = patient_state.get('biomarkers', torch.zeros(cfg.num_biomarkers))
        
        # Create graph structure representing patient state
        num_nodes = 20
        x = torch.zeros(num_nodes, cfg.feature_dim)
        
        # Encode biomarkers into node features
        if torch.is_tensor(biomarkers):
            # Distribute biomarker values across nodes
            for i in range(min(num_nodes, len(biomarkers))):
                x[i, :10] = biomarkers[i].expand(10) if biomarkers[i].dim() == 0 else biomarkers[i][:10]
        
        # Add disease-specific patterns
        disease_severity = patient_state.get('disease_severity', 0.5)
        x[:, 10] = disease_severity
        
        # Create edges based on biomarker correlations
        edge_index = []
        for i in range(num_nodes):
            for j in range(i+1, min(i+3, num_nodes)):
                edge_index.extend([[i, j], [j, i]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Add global features
        u = torch.tensor([disease_severity, len(patient_state.get('critical_conditions', {}))])
        
        return Data(x=x, edge_index=edge_index, u=u)
    
    @staticmethod
    def therapeutic_response_to_fitness(response: Dict) -> float:
        """Convert therapeutic response metrics to fitness score"""
        efficacy = response.get('efficacy_score', 0.5)
        safety = response.get('safety_score', 0.5)
        speed = response.get('response_speed', 0.5)
        
        # Weighted combination
        fitness = 0.4 * efficacy + 0.3 * safety + 0.3 * speed
        return fitness

# ============================================================================
# PATIENT DATA GENERATOR
# ============================================================================

class PatientDataGenerator:
    """Generates realistic patient biomarker data"""
    
    def __init__(self, patient_profile: Dict):
        self.profile = patient_profile
        self.time = 0
        self.disease_state = patient_profile.get('severity', 0.5)
        
    def generate_biomarkers(self) -> torch.Tensor:
        """Generate realistic biomarker values based on disease state"""
        # Use consistent device
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        biomarkers = torch.randn(cfg.num_biomarkers, device=device) * 0.1
        
        # Disease-specific patterns
        if self.profile['disease'] == 'autoimmune_inflammatory':
            # Inflammatory markers
            biomarkers[0] = 2.0 + self.disease_state * 3.0  # IL-6
            biomarkers[1] = 1.5 + self.disease_state * 2.0  # TNF-α
            biomarkers[2] = 5.0 + self.disease_state * 10.0  # CRP
            
            # Add circadian variation
            circadian = np.sin(self.time * np.pi / 12)  # 24-hour cycle
            biomarkers[:3] += circadian * 0.5
            
        elif self.profile['disease'] == 'cancer':
            # Tumor markers
            biomarkers[5] = 10.0 + self.disease_state * 50.0  # CA-125
            biomarkers[6] = 5.0 + self.disease_state * 20.0   # CEA
            
        elif self.profile['disease'] == 'metabolic_syndrome':
            # Metabolic markers
            biomarkers[3] = 90 + self.disease_state * 100  # Glucose
            biomarkers[7] = 150 + self.disease_state * 100  # Triglycerides
            
        # Add noise and ensure positive values
        biomarkers = torch.abs(biomarkers + torch.randn_like(biomarkers) * 0.1)
        
        self.time += 1
        return biomarkers
    
    def update_disease_state(self, treatment_efficacy: float):
        """Update disease state based on treatment"""
        # Disease improves with effective treatment
        improvement = treatment_efficacy * 0.05
        self.disease_state = max(0.1, self.disease_state - improvement)
        
        # Add some variability
        self.disease_state += np.random.normal(0, 0.02)
        self.disease_state = np.clip(self.disease_state, 0.0, 1.0)

# ============================================================================
# THERAPEUTIC ANTIGEN ADAPTER
# ============================================================================

class TherapeuticAntigenAdapter:
    """Converts patient biomarkers to antigen format for TE-AI"""
    
    @staticmethod
    def biomarkers_to_antigen(biomarkers: torch.Tensor, patient_state: Dict) -> Data:
        """Convert biomarker data to graph format"""
        # Create nodes representing different biomarker categories
        num_nodes = 20  # Biomarker groups
        
        # Node features: biomarker values grouped by category
        node_features = []
        
        # Group 1: Inflammatory markers (nodes 0-4)
        for i in range(5):
            idx = i % len(biomarkers)
            node_features.append([
                biomarkers[idx].item(),
                patient_state.get('disease_severity', 0.5),
                patient_state.get('inflammatory_score', 0.5),
                float(i == 0)  # Mark primary inflammatory node
            ])
        
        # Group 2: Metabolic markers (nodes 5-9)
        for i in range(5, 10):
            idx = i % len(biomarkers)
            node_features.append([
                biomarkers[idx].item(),
                patient_state.get('metabolic_score', 0.5),
                0.0,
                float(i == 5)  # Mark primary metabolic node
            ])
        
        # Group 3: Disease-specific markers (nodes 10-14)
        for i in range(10, 15):
            idx = i % len(biomarkers)
            severity = patient_state.get('disease_probabilities', torch.zeros(10))
            node_features.append([
                biomarkers[idx].item(),
                severity[i-10].item() if i-10 < len(severity) else 0.0,
                0.0,
                0.0
            ])
        
        # Group 4: Critical condition indicators (nodes 15-19)
        critical = patient_state.get('critical_conditions', {})
        for i in range(15, 20):
            condition_names = list(critical.keys())
            is_critical = float(i-15 < len(condition_names) and critical.get(condition_names[i-15], False))
            node_features.append([
                is_critical * 10.0,  # High value if critical
                is_critical,
                0.0,
                is_critical
            ])
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edges: fully connected within groups, sparse between groups
        edge_list = []
        
        # Within-group connections
        groups = [(0, 5), (5, 10), (10, 15), (15, 20)]
        for start, end in groups:
            for i in range(start, end):
                for j in range(i+1, end):
                    edge_list.extend([[i, j], [j, i]])
        
        # Between-group connections (sparse)
        for i in range(0, 20, 5):
            if i + 5 < 20:
                edge_list.extend([[i, i+5], [i+5, i]])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Global features
        global_features = torch.tensor([
            patient_state.get('disease_severity', 0.5),
            len(critical),
            biomarkers.mean().item(),
            biomarkers.std().item()
        ])
        
        return Data(x=x, edge_index=edge_index, global_features=global_features)

# ============================================================================
# THERAPEUTIC GERMINAL CENTER - PROPERLY EXTENDS CORE SYSTEM
# ============================================================================

class TherapeuticGerminalCenter(ProductionGerminalCenter):
    """Extends core TE-AI germinal center with therapeutic capabilities"""
    
    def __init__(self, patient_profile: Dict):
        # Initialize the full production system
        super().__init__()
        
        self.patient_profile = patient_profile
        self.patient_generator = PatientDataGenerator(patient_profile)
        self.domain_adapter = TherapeuticDomainAdapter()
        
        # Therapeutic-specific metrics (additional to core metrics)
        self.treatment_history = deque(maxlen=1000)
        self.patient_outcomes = deque(maxlen=1000)
        
        # Initialize gene pool if not already done by parent
        if not hasattr(self, 'initial_genes'):
            self.initial_genes = []
        
        # Add therapeutic genes to the core gene pool
        self._inject_therapeutic_genes()
        
    def _inject_therapeutic_genes(self):
        """Add therapeutic genes to core system's gene pool"""
        # Create therapeutic gene instances that work with core system
        therapeutic_genes = [
            BiosensorGene(),
            TherapeuticEffectorGene('anti-inflammatory'),
            TherapeuticEffectorGene('immunomodulation'),
            TherapeuticEffectorGene('metabolic_regulation'),
            AdaptiveControllerGene(),
            TherapeuticStemGene()
        ]
        
        # Move to device
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        for gene in therapeutic_genes:
            gene = gene.to(device)
            # Add to initial gene pool (core system will handle distribution)
            self.initial_genes.append(gene)
    
    def initialize_population(self, size: int = None):
        """Use core system's initialization with therapeutic genes added"""
        # Core system uses _initialize_population (private method)
        # We need to manually initialize since it's called in parent __init__
        if not self.population:
            # Create initial population with therapeutic genes
            self._create_therapeutic_population(size or cfg.initial_population)
        
        print(f"   ✅ Therapeutic population initialized using core TE-AI system")
        print(f"   Population: {len(self.population)} cells with transposable elements")
    
    def _create_therapeutic_population(self, size: int):
        """Create population with therapeutic genes using core system components"""
        from transposable_immune_ai_production_complete import ProductionBCell
        
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        
        # Mix of therapeutic and core genes
        for i in range(size):
            genes = []
            
            # Add therapeutic genes
            if i % 5 == 0:
                genes.append(BiosensorGene().to(device))
            if i % 5 == 1:
                genes.append(TherapeuticEffectorGene(random.choice(cfg.therapeutic_modes)).to(device))
            if i % 5 == 2:
                genes.append(AdaptiveControllerGene().to(device))
            
            # Always add some stem genes
            genes.append(TherapeuticStemGene().to(device))
            
            # Add core system genes from the pool
            if hasattr(self, 'initial_genes') and self.initial_genes:
                # Add random core genes
                num_core = random.randint(1, 3)
                for _ in range(num_core):
                    if self.initial_genes:
                        core_gene = random.choice(self.initial_genes)
                        genes.append(copy.deepcopy(core_gene))
            
            # Create cell using core system's ProductionBCell
            cell = ProductionBCell(genes)
            self.population[cell.cell_id] = cell
        
    def evolve_generation(self, antigens: List[Data]):
        """Use core system's evolution with therapeutic adaptations"""
        # Generate patient data
        biomarkers = self.patient_generator.generate_biomarkers()
        patient_state = self._assess_patient(biomarkers)
        
        # Convert patient state to antigens
        therapeutic_antigens = [self.domain_adapter.patient_to_antigen(patient_state) 
                               for _ in range(len(antigens))]
        
        # Call parent's FULL evolution cycle - this handles EVERYTHING
        super().evolve_generation(therapeutic_antigens)
        
        # After evolution, add therapeutic-specific processing
        print("\n💊 Therapeutic Response Phase")
        treatment = self._generate_treatment_from_population(patient_state)
        patient_response = self._simulate_patient_response(treatment)
        self._therapeutic_adaptation(patient_response)
        
        # Update patient state based on treatment
        self.patient_generator.update_disease_state(patient_response['efficacy_score'])
        
        # Record therapeutic outcomes
        self.patient_outcomes.append({
            'generation': self.generation,
            'disease_state': self.patient_generator.disease_state,
            'efficacy': patient_response['efficacy_score'],
            'safety': patient_response['safety_score']
        })
    
    def _generate_treatment_from_population(self, patient_state: Dict) -> Dict:
        """Generate treatment using evolved population"""
        therapeutic_outputs = []
        
        # Collect responses from cells with therapeutic genes
        for cell in list(self.population.values())[:100]:  # Sample
            for gene in cell.genes:
                if hasattr(gene, 'generate_therapeutic'):
                    try:
                        output = gene.generate_therapeutic(patient_state)
                        therapeutic_outputs.append(output)
                    except:
                        pass
        
        # Combine using core system's principles
        if therapeutic_outputs:
            return self._combine_therapeutics(therapeutic_outputs)
        else:
            device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
            return {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0, 'safety_score': 1.0}
    
    def _therapeutic_adaptation(self, patient_response: Dict):
        """Adapt population based on therapeutic success"""
        # Trigger stress if poor response (uses core stress response)
        if patient_response['efficacy_score'] < 0.5:
            self.current_stress = 1.0 - patient_response['efficacy_score']
            print(f"   🔥 Poor therapeutic response - stress level: {self.current_stress:.3f}")
            
            # Core system handles transposition under stress
            self._execute_stress_response()
    
    def get_population_fitness(self) -> float:
        """Get average fitness of population"""
        fitnesses = []
        for cell in self.population.values():
            if hasattr(cell, 'fitness_history') and cell.fitness_history:
                fitnesses.append(cell.fitness_history[-1])
            else:
                fitnesses.append(0.5)
        return np.mean(fitnesses) if fitnesses else 0.5
        
    def _assess_patient(self, biomarkers: torch.Tensor) -> Dict:
        """Comprehensive patient assessment using biosensor genes"""
        patient_state = {
            'biomarkers': biomarkers,
            'timestamp': time.time()
        }
        
        # Use biosensor genes
        sensor_outputs = []
        for cell in list(self.population.values())[:50]:  # Sample cells
            for gene in cell.genes:
                if isinstance(gene, BiosensorGene):
                    output = gene.sense_patient_state(biomarkers)
                    sensor_outputs.append(output)
        
        # Aggregate sensing
        if sensor_outputs:
            # Ensure all tensors are on same device
            device = biomarkers.device
            encoded_states = [s['encoded_state'].to(device) for s in sensor_outputs]
            disease_probs = [s['disease_probabilities'].to(device) for s in sensor_outputs]
            
            patient_state.update({
                'encoded_state': torch.stack(encoded_states).mean(0),
                'disease_probabilities': torch.stack(disease_probs).mean(0),
                'critical_conditions': sensor_outputs[0].get('critical_conditions', {}),
                'disease_severity': max(0.1, min(1.0, self.patient_generator.disease_state))
            })
        
        return patient_state
    
    def _generate_treatment(self, patient_state: Dict) -> Dict:
        """Generate therapeutic intervention"""
        therapeutic_outputs = []
        controller_plans = []
        
        # Collect therapeutic responses
        for cell in list(self.population.values())[:100]:  # Process subset
            for gene in cell.genes:
                if isinstance(gene, TherapeuticEffectorGene):
                    output = gene.generate_therapeutic(patient_state)
                    therapeutic_outputs.append(output)
                elif isinstance(gene, AdaptiveControllerGene):
                    plan = gene.plan_treatment([], patient_state, horizon=12)
                    controller_plans.append(plan)
        
        # Combine treatments
        if therapeutic_outputs:
            combined = self._combine_therapeutics(therapeutic_outputs)
        else:
            device = patient_state.get('biomarkers', torch.zeros(1)).device
            combined = {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0, 'safety_score': 0.5}
        
        return {
            'therapeutic': combined,
            'num_effectors': len(therapeutic_outputs),
            'num_controllers': len(controller_plans)
        }
    
    def _combine_therapeutics(self, outputs: List[Dict]) -> Dict:
        """Safely combine multiple therapeutic outputs"""
        if not outputs:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0, 'safety_score': 1.0}
        
        # Average therapeutics - ensure all on same device
        device = outputs[0]['therapeutic'].device
        therapeutics = torch.stack([o['therapeutic'].to(device) for o in outputs])
        doses = torch.stack([o['dose'].to(device) if torch.is_tensor(o['dose']) else torch.tensor(o['dose'], device=device) for o in outputs])
        safety_scores = torch.stack([o['safety_score'].to(device) if torch.is_tensor(o['safety_score']) else torch.tensor(o['safety_score'], device=device) for o in outputs])
        
        # Combine with safety constraints
        combined_therapeutic = therapeutics.mean(0)
        combined_dose = doses.mean() * safety_scores.min()  # Conservative dosing
        combined_safety = safety_scores.min()
        
        return {
            'therapeutic': combined_therapeutic,
            'dose': combined_dose,
            'safety_score': combined_safety
        }
    
    def _evaluate_therapeutic_fitness(self, antigens: List[Data], treatment: Dict) -> Dict[str, float]:
        """Evaluate cells based on therapeutic contribution"""
        fitness_scores = {}
        
        # Base fitness on treatment quality
        base_fitness = treatment['therapeutic']['dose'].item() * treatment['therapeutic']['safety_score'].item()
        
        for cell_id, cell in self.population.items():
            # Count therapeutic genes
            therapeutic_genes = sum(1 for g in cell.genes if isinstance(g, 
                (BiosensorGene, TherapeuticEffectorGene, AdaptiveControllerGene)))
            
            # Fitness based on specialization
            specialization_bonus = therapeutic_genes * 0.1
            
            # Diversity bonus
            gene_types = set(g.gene_type for g in cell.genes if g.is_active)
            diversity_bonus = len(gene_types) * 0.05
            
            fitness = base_fitness + specialization_bonus + diversity_bonus
            fitness_scores[cell_id] = min(1.0, fitness)
            
            # Update cell records
            cell.fitness_history.append(fitness)
        
        return fitness_scores
    
    def _simulate_patient_response(self, treatment: Dict) -> Dict:
        """Simulate patient response to treatment"""
        # Handle treatment structure
        if 'therapeutic' in treatment and isinstance(treatment['therapeutic'], dict):
            therapeutic = treatment['therapeutic'].get('therapeutic', torch.zeros(20))
            dose = treatment['therapeutic'].get('dose', 0.0)
            safety_score = treatment['therapeutic'].get('safety_score', 1.0)
        else:
            # Fallback
            device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
            therapeutic = torch.zeros(20, device=device)
            dose = 0.0
            safety_score = 1.0
        
        # Ensure tensors are on correct device
        if not torch.is_tensor(therapeutic):
            device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
            therapeutic = torch.tensor(therapeutic, device=device)
        
        # Convert to scalars if tensors
        if torch.is_tensor(dose):
            dose = dose.item()
        if torch.is_tensor(safety_score):
            safety_score = safety_score.item()
        
        # Calculate treatment effect
        effect_magnitude = (therapeutic.sum().item() * dose) / 20.0  # Normalize
        
        # Patient-specific response
        response_variability = np.random.normal(1.0, 0.15)
        actual_effect = effect_magnitude * response_variability
        
        # Generate response metrics
        response = {
            'efficacy_score': min(1.0, actual_effect),
            'safety_score': safety_score,
            'response_speed': np.random.uniform(0.4, 0.9),
            'biomarker_change': torch.randn(cfg.num_biomarkers, device=therapeutic.device) * actual_effect
        }
        
        # Check for adverse events
        if np.random.random() < (1 - safety_score) * 0.05:
            response['adverse_event'] = True
            response['efficacy_score'] *= 0.7
            print("   ⚠️  Adverse event detected!")
        
        return response
    
    def _adapt_to_patient_response(self, response: Dict):
        """Adapt population based on patient response"""
        # Trigger stress if poor response
        if response['efficacy_score'] < 0.5:
            self.current_stress = 1.0 - response['efficacy_score']
            print(f"   🔥 Poor response - stress level: {self.current_stress:.3f}")
            
            # Increase transposition in therapeutic genes
            for cell in list(self.population.values())[:50]:
                for gene in cell.genes:
                    if isinstance(gene, (TherapeuticEffectorGene, BiosensorGene)):
                        if np.random.random() < self.current_stress * 0.3:
                            child = gene.transpose(self.current_stress, 0.5)
                            if child:
                                print(f"   🧬 Therapeutic gene transposed: {gene.gene_type}")
        
        # Guide stem cell differentiation
        if response.get('adverse_event', False):
            # Need more safety-focused genes
            self._guide_stem_differentiation('AC')  # Adaptive controllers
        elif response['efficacy_score'] < 0.3:
            # Need better effectors
            self._guide_stem_differentiation('TE')
    
    def _guide_stem_differentiation(self, target_type: str):
        """Guide stem cells to differentiate"""
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, TherapeuticStemGene) and gene.commitment_level < 0.5:
                    # Force differentiation
                    population_state = {
                        'cells': list(self.population.values()),
                        'avg_fitness': np.mean([c.fitness_history[-1] if c.fitness_history else 0 
                                              for c in self.population.values()])
                    }
                    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
                    gene.guided_differentiation(
                        {'encoded_state': torch.randn(cfg.hidden_dim, device=device), 
                         'disease_severity': self.patient_generator.disease_state},
                        population_state
                    )





# ============================================================================
# INTEGRATION WITH TE-AI SYSTEM
# ============================================================================

# REMOVED - No longer needed! Core system handles population creation
# The therapeutic genes are injected into the core gene pool

# ============================================================================
# OPTIMIZED THERAPEUTIC GERMINAL CENTER
# ============================================================================

class TherapeuticOptimizedGerminalCenter(OptimizedProductionGerminalCenter):
    """Extends OPTIMIZED germinal center with therapeutic capabilities"""
    
    def __init__(self, patient_profile: Dict):
        # Initialize parent (no arguments needed)
        super().__init__()
        
        self.patient_profile = patient_profile
        self.patient_generator = PatientDataGenerator(patient_profile)
        self.domain_adapter = TherapeuticDomainAdapter()
        
        # Use optimized batch evaluator
        self.batch_evaluator = OptimizedBatchEvaluator(device=cfg.device)
        
        # Therapeutic-specific tracking
        self.treatment_history = deque(maxlen=1000)
        self.patient_outcomes = deque(maxlen=1000)
        
        # Set global reference for visualization
        import transposable_immune_ai_production_complete as prod
        prod._current_germinal_center = self
        
    def evolve_generation(self, antigens: List[Data]):
        """Use parent's optimized evolution with therapeutic adaptations"""
        # Convert patient state antigens if needed
        therapeutic_antigens = antigens
        
        # Call parent's OPTIMIZED evolution cycle
        super().evolve_generation(therapeutic_antigens)
        
        # Additional therapeutic tracking
        avg_fitness = np.mean([cell.fitness_history[-1] if cell.fitness_history else 0.5 
                              for cell in self.population.values()])
        self.treatment_history.append({
            'generation': self.generation,
            'fitness': avg_fitness,
            'timestamp': time.time()
        })

# ============================================================================
# THERAPEUTIC EVOLUTION CYCLE
# ============================================================================

class LivingTherapeuticSystem:
    """Therapeutic system that uses OPTIMIZED TE-AI components"""
    
    def __init__(self, patient_profile: Dict):
        self.patient_profile = patient_profile
        
        # Use OPTIMIZED germinal center with therapeutic adaptations
        self.germinal_center = TherapeuticOptimizedGerminalCenter(patient_profile)
        
        # Population is initialized automatically by parent class
        self.population = self.germinal_center.population
        
        # Patient state tracking
        self.patient_history = deque(maxlen=1000)
        self.treatment_history = deque(maxlen=1000)
        self.patient_generator = PatientDataGenerator(patient_profile)
        
        print(f"\n✅ Living Therapeutic System initialized with OPTIMIZED components:")
        print(f"   - Patient: {patient_profile.get('patient_id', 'Unknown')}")
        print(f"   - Disease: {patient_profile.get('disease_type', 'Unknown')}")
        print(f"   - Population: {len(self.population)} cells")
        print(f"   - Using OptimizedBatchEvaluator for parallel processing")
        
    def write_therapeutic_visualization_state(self):
        """Write therapeutic-specific visualization state to JSON."""
        # Use a lock to prevent race conditions, similar to the core system
        global state_lock
        if 'state_lock' not in globals():
            import threading
            state_lock = threading.Lock()


        zones = {
            'darkZone': {'center': [-100, 0, 0], 'radius': 70},
            'lightZone': {'center': [100, 0, 0], 'radius': 70},
            'memoryZone': {'center': [0, 120, 0], 'radius': 60},
            'quantumLayer': {'center': [0, -120, 0], 'radius': 70},
            'mantleZone': {'center': [0, 0, 0], 'radius': 200}
        }

        def get_cell_zone(cell_type):
            if cell_type == 'stem': return 'memoryZone'
            if cell_type == 'effector': return 'lightZone'
            if cell_type == 'biosensor': return 'darkZone'
            if cell_type == 'controller': return 'quantumLayer'
            return 'mantleZone'

        def get_position_in_zone(zone_name, index, total_in_zone):
            zone_data = zones[zone_name]
            angle = (index / max(1, total_in_zone)) * 2 * np.pi
            radius = np.random.rand() * zone_data['radius'] * 0.8
            height = (np.random.rand() - 0.5) * zone_data['radius'] * 0.5
            
            return {
                'x': zone_data['center'][0] + np.cos(angle) * radius,
                'y': zone_data['center'][1] + height,
                'z': zone_data['center'][2] + np.sin(angle) * radius
            }
            
        # --- Assemble the state dictionary ---
        # --- Assemble the state dictionary ---
        state = {
            'type': 'therapeutic',
            'generation': self.germinal_center.generation,
            'timestamp': time.time(),
            'patient': {
                'id': self.patient_profile.get('id', 'Unknown'),
                'disease': self.patient_profile.get('disease', 'Unknown'),
                'severity': self.patient_generator.disease_state
            },
            'population_size': len(self.population),
            'cells': []
        }
        
        # --- Pre-process cells to determine types and count per zone ---
        all_cell_info = []
        zone_counts = defaultdict(int)
        for cell_id, cell in self.population.items():
            type_counts = defaultdict(int)
            for gene in cell.genes:
                if hasattr(gene, 'gene_type'):
                    type_counts[gene.gene_type] += 1
            
            dominant_type_abbrev = max(type_counts, key=type_counts.get) if type_counts else None
            type_mapping = {'BS': 'biosensor', 'TE': 'effector', 'AC': 'controller', 'TS': 'stem'}
            cell_type = type_mapping.get(dominant_type_abbrev, 'balanced')
            
            zone_name = get_cell_zone(cell_type)
            zone_counts[zone_name] += 1
            
            all_cell_info.append({'id': cell_id, 'cell': cell, 'type': cell_type, 'zone': zone_name})

        # --- Add cell data with pre-calculated positions ---
        current_zone_indices = defaultdict(int)
        for info in all_cell_info:
            cell_id, cell, cell_type, zone_name = info['id'], info['cell'], info['type'], info['zone']
            
            # Calculate position
            index_in_zone = current_zone_indices[zone_name]
            total_in_zone = zone_counts[zone_name]
            position = get_position_in_zone(zone_name, index_in_zone, total_in_zone)
            current_zone_indices[zone_name] += 1

            # Get gene list
            gene_list = []
            type_counts = defaultdict(int)
            for gene in cell.genes:
                if gene.is_active and hasattr(gene, 'gene_type'):
                    type_counts[gene.gene_type] += 1
                    gene_list.append({
                        'gene_id': str(id(gene)),
                        'gene_type': gene.gene_type,
                        'type': gene.gene_type,
                        'is_active': True,
                        'position': getattr(gene, 'position', 0.5),
                        'depth': gene.compute_depth().item() if hasattr(gene, 'compute_depth') else 1.0,
                        'is_quantum': 'Quantum' in gene.__class__.__name__
                    })
            
            cell_data = {
                'id': cell_id,
                'fitness': cell.fitness_history[-1] if cell.fitness_history else 0.5,
                'genes': gene_list,
                'type': cell_type,
                'therapeutic_genes': {
                    'biosensor': type_counts.get('BS', 0),
                    'effector': type_counts.get('TE', 0),
                    'controller': type_counts.get('AC', 0),
                    'stem': type_counts.get('TS', 0)
                },
                'position': position # <-- USE THE CALCULATED POSITION
            }
            state['cells'].append(cell_data)
            
            
                    
        # Add therapeutic-specific metrics
        if self.treatment_history:
            latest_treatment = self.treatment_history[-1]
            if 'therapeutic' in latest_treatment and isinstance(latest_treatment['therapeutic'], dict):
                efficacy = latest_treatment['therapeutic'].get('efficacy_score', 0)
                safety = latest_treatment['therapeutic'].get('safety_score', 1.0)
            else:
                efficacy = latest_treatment.get('efficacy_score', 0)
                safety = latest_treatment.get('safety_score', 1.0)
            state['therapeutic_metrics'] = {'efficacy': efficacy, 'safety': safety, 'resistance': self.patient_profile.get('treatment_resistance', 0)}
        
        # Add visualization-friendly data
        state['stress_level'] = self.germinal_center.current_stress
        state['stress'] = self.germinal_center.current_stress
        state['mean_fitness'] = np.mean([c.fitness_history[-1] if c.fitness_history else 0.5 for c in self.population.values()])
        state['fitness'] = state['mean_fitness']
        state['diversity'] = len(set(g.gene_type for c in self.population.values() for g in c.genes if g.is_active)) / 4.0
        state['phase'] = getattr(self.germinal_center, 'phase_detector', type('',(object,),{'current_phase':'stable'})).current_phase
        state['nodes'] = []
        state['links'] = []
        
        
        
        # --- File Writing Logic (merged from the generic function) ---
        # Define the output path for the therapeutic visualization
        # This assumes the script is run from the root of the project.
        # Adjust the path if necessary.
        # Define the output path
        viz_dir = os.path.join(os.getcwd(), 'scripts', 'visualization', 'therapeutic')
        os.makedirs(viz_dir, exist_ok=True)
        therapeutic_json_path = os.path.join(viz_dir, 'therapeutic_visualization_state.json')

        with state_lock:
            try:
                with open(therapeutic_json_path, 'w') as f:
                    json.dump(state, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Failed to write therapeutic visualization state: {e}")
                
                        
    def therapeutic_cycle(self, current_biomarkers: torch.Tensor) -> Dict:
        """One cycle using CORE TE-AI evolution with therapeutic adaptations"""
        
        # 1. Sense patient state
        patient_state = self._comprehensive_patient_assessment(current_biomarkers)
        
        # 2. Convert to antigens for core system
        adapter = TherapeuticDomainAdapter()
        therapeutic_antigens = [adapter.patient_to_antigen(patient_state) 
                               for _ in range(cfg.batch_size)]
        
        # 3. RUN CORE TE-AI EVOLUTION CYCLE
        # This includes transposition, dream consolidation, phase transitions!
        self.germinal_center.evolve_generation(therapeutic_antigens)
        
        # 4. Extract therapeutic response from evolved population
        therapeutic_response = self._extract_therapeutic_response(patient_state)
        
        # 5. Simulate patient response
        patient_response = self._simulate_patient_response(therapeutic_response)
        
        # 6. Update fitness based on therapeutic success
        self._update_therapeutic_fitness(patient_response)
        
        # 7. Record history
        self.patient_history.append(patient_state)
        self.treatment_history.append(therapeutic_response)
        
        # 8. Write visualization state if requested
        self.write_therapeutic_visualization_state()
        
        return {
            'patient_state': patient_state,
            'treatment': therapeutic_response,
            'response': patient_response,
            'population_size': len(self.population),
            'avg_fitness': np.mean([cell.fitness_history[-1] if cell.fitness_history else 0.5 
                                   for cell in self.population.values()]),
            'generation': self.germinal_center.generation,
            'transposition_events': len(self.germinal_center.transposition_events),
            'dream_consolidation': self.germinal_center.generation % cfg.dream_frequency == 0
        }
    
    def _comprehensive_patient_assessment(self, biomarkers: torch.Tensor) -> Dict:
        """Comprehensive patient state assessment"""
        patient_state = {
            'biomarkers': biomarkers,
            'timestamp': time.time()
        }
        
        # Run all biosensor genes
        sensor_outputs = []
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, BiosensorGene):
                    sensor_output = gene.sense_patient_state(biomarkers)
                    sensor_outputs.append(sensor_output)
        
        # Aggregate sensor data
        if sensor_outputs:
            # Get device from biomarkers or use default
            device = biomarkers.device if torch.is_tensor(biomarkers) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Stack encoded states on same device
            encoded_states = [s['encoded_state'].to(device) for s in sensor_outputs]
            patient_state['encoded_state'] = torch.stack(encoded_states).mean(0)
            
            # Stack disease probabilities on same device
            disease_probs = [s['disease_probabilities'].to(device) for s in sensor_outputs]
            patient_state['disease_probabilities'] = torch.stack(disease_probs).mean(0)
            
            # Check for critical conditions
            critical_conditions = {}
            for output in sensor_outputs:
                for condition, detected in output['critical_conditions'].items():
                    if detected:
                        critical_conditions[condition] = True
            
            patient_state['critical_conditions'] = critical_conditions
            
            # Calculate severity
            patient_state['disease_severity'] = patient_state['disease_probabilities'].max().item()
        else:
            # No sensors yet - create default values
            device = biomarkers.device if torch.is_tensor(biomarkers) else torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
            patient_state['encoded_state'] = torch.zeros(cfg.hidden_dim, device=device)
            patient_state['disease_probabilities'] = torch.zeros(3, device=device)
            patient_state['critical_conditions'] = {}
            patient_state['disease_severity'] = 0.5  # Default moderate severity
        
        return patient_state
    
    def _extract_therapeutic_response(self, patient_state: Dict) -> Dict:
        """Extract therapeutic response from core-evolved population"""
        therapeutic_outputs = []
        controller_plans = []
        
        # Sample evolved population (core system has already done selection)
        for cell in list(self.population.values())[:100]:
            for gene in cell.genes:
                # Therapeutic genes generate responses
                if isinstance(gene, TherapeuticEffectorGene):
                    try:
                        therapeutic = gene.generate_therapeutic(patient_state)
                        therapeutic_outputs.append(therapeutic)
                    except:
                        pass
                elif isinstance(gene, AdaptiveControllerGene):
                    try:
                        plan = gene.plan_treatment(
                            list(self.patient_history)[-24:],
                            patient_state
                        )
                        controller_plans.append(plan)
                    except:
                        pass
        
        # Combine outputs
        if therapeutic_outputs:
            combined_therapeutic = self._combine_therapeutics(therapeutic_outputs)
        else:
            device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
            combined_therapeutic = {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0, 'safety_score': 1.0}
        
        return {
            'therapeutic': combined_therapeutic,
            'plan': controller_plans[0] if controller_plans else {'plan': [], 'horizon': 24},
            'num_effectors': len(therapeutic_outputs),
            'num_controllers': len(controller_plans),
            'from_generation': self.germinal_center.generation
        }
    
    def _combine_therapeutics(self, therapeutic_outputs: List[Dict]) -> Dict:
        """Intelligently combine multiple therapeutic outputs"""
        if not therapeutic_outputs:
            # Return empty therapeutic on correct device
            device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
            return {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0, 'safety_score': 1.0}
        
        # Group by mode
        by_mode = defaultdict(list)
        for output in therapeutic_outputs:
            by_mode[output['mode']].append(output)
        
        # Find max therapeutic size across all outputs
        # Handle both 1D and 2D tensors
        max_size = 0
        for output in therapeutic_outputs:
            therapeutic = output['therapeutic']
            if therapeutic.dim() == 2:
                size = therapeutic.shape[1]  # For [1, X] tensors, use the second dimension
            else:
                size = therapeutic.shape[0]  # For [X] tensors, use the first dimension
            max_size = max(max_size, size)
        # Max therapeutic size determined
        
        # Combine within modes (avoid overdosing)
        # Get device from first therapeutic tensor
        device = therapeutic_outputs[0]['therapeutic'].device if therapeutic_outputs else torch.device('cpu')
        combined = {
            'therapeutic': torch.zeros(max_size, device=device),
            'dose': 0.0,
            'safety_score': 1.0
        }
        
        for mode, outputs in by_mode.items():
            # Process this mode
            
            # Average therapeutics of same type
            # Get device from first therapeutic
            device = outputs[0]['therapeutic'].device
            
            # Ensure all tensors are on same device
            therapeutic_tensors = [o['therapeutic'].to(device) for o in outputs]
            mode_therapeutic = torch.stack(therapeutic_tensors).mean(0)
            
            dose_tensors = [o['dose'].to(device) if torch.is_tensor(o['dose']) else torch.tensor(o['dose'], device=device) for o in outputs]
            mode_dose = torch.stack(dose_tensors).mean()
            
            safety_tensors = [o['safety_score'].to(device) if torch.is_tensor(o['safety_score']) else torch.tensor(o['safety_score'], device=device) for o in outputs]
            mode_safety = torch.stack(safety_tensors).min()
            
            # Average computed
            
            # Add to combined with safety limits
            safe_dose = min(mode_dose.item(), cfg.max_therapeutic_strength)
            # Safe dose determined
            
            # Handle different therapeutic sizes by padding or truncating
            if mode_therapeutic.dim() > 1:
                # Flatten if needed
                mode_therapeutic = mode_therapeutic.squeeze()
            
            # Scale by safe dose
            scaled_therapeutic = mode_therapeutic * safe_dose
            
            if mode_therapeutic.shape[0] <= max_size:
                # Pad with zeros if needed
                if mode_therapeutic.shape[0] < max_size:
                    padded = torch.zeros(max_size, device=mode_therapeutic.device)
                    # Pad to max size
                    padded[:mode_therapeutic.shape[0]] = scaled_therapeutic
                    combined['therapeutic'] += padded
                else:
                    # Exact size match
                    # Size matches
                    combined['therapeutic'] += scaled_therapeutic
            else:
                # Therapeutic is larger than max_size - we need to truncate
                # Truncate to max size
                combined['therapeutic'] += scaled_therapeutic[:max_size]
            
            combined['dose'] += safe_dose / len(by_mode)  # Normalize by number of modes
            combined['safety_score'] *= mode_safety
        
        return combined
    
    def _simulate_patient_response(self, treatment: Dict) -> Dict:
        """Simulate how patient responds to treatment"""
        # Handle both old and new treatment structures
        if 'therapeutic' in treatment and isinstance(treatment['therapeutic'], dict):
            # New structure from _generate_population_response
            if 'therapeutic' in treatment['therapeutic']:
                therapeutic = treatment['therapeutic']['therapeutic']
                dose = treatment['therapeutic']['dose']
                safety_score = treatment['therapeutic'].get('safety_score', 0.5)
            else:
                # Fallback for when therapeutic response is empty
                device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
                therapeutic = torch.zeros(20, device=device)
                dose = 0.0
                safety_score = 1.0
        else:
            # Fallback structure
            device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
            therapeutic = torch.zeros(20, device=device)
            dose = 0.0
            safety_score = 1.0
        
        # Calculate therapeutic effect
        if torch.is_tensor(dose):
            dose_value = dose.item()
        else:
            dose_value = float(dose)
            
        if torch.is_tensor(safety_score):
            safety_value = safety_score.item()
        else:
            safety_value = float(safety_score)
        
        base_effect = (therapeutic.sum() * dose_value).item() if torch.is_tensor(therapeutic.sum()) else therapeutic.sum() * dose_value
        
        # Add variability based on patient
        patient_variability = random.gauss(1.0, 0.2)
        actual_effect = base_effect * patient_variability
        
        # Determine response
        response = {
            'efficacy_score': min(1.0, max(0.0, actual_effect / 10.0)),
            'safety_score': safety_value,
            'response_speed': random.uniform(0.3, 0.9),
            'efficiency_score': dose_value,  # Lower dose = more efficient
            'biomarker_change': torch.randn(cfg.num_biomarkers, device=therapeutic.device) * actual_effect
        }
        
        # Check for adverse events
        if random.random() < (1 - response['safety_score']) * 0.1:
            response['adverse_event'] = True
            response['efficacy_score'] *= 0.5
        
        return response
    
    def _update_therapeutic_fitness(self, patient_response: Dict):
        """Update cell fitness based on therapeutic success"""
        # Convert therapeutic response to fitness modifier
        therapeutic_fitness = TherapeuticDomainAdapter.therapeutic_response_to_fitness(patient_response)
        
        # Apply fitness bonus to cells with therapeutic genes
        for cell in self.population.values():
            therapeutic_gene_count = sum(
                1 for gene in cell.genes 
                if isinstance(gene, (BiosensorGene, TherapeuticEffectorGene, AdaptiveControllerGene))
            )
            
            # Boost fitness for therapeutic contributors
            if therapeutic_gene_count > 0:
                fitness_boost = therapeutic_fitness * (1 + 0.1 * therapeutic_gene_count)
                if hasattr(cell, 'fitness_history') and cell.fitness_history:
                    cell.fitness_history[-1] *= fitness_boost
        
        # Update stress based on treatment outcome (core system handles response)
        if patient_response['efficacy_score'] < 0.5:
            self.germinal_center.current_stress = 1.0 - patient_response['efficacy_score']
            print(f"   🔥 Poor therapeutic outcome - stress: {self.germinal_center.current_stress:.3f}")




        
    def _inject_enhanced_stem_cells(self, count: int = 10):
        """Add enhanced stem cells to the population"""
        print(f"\n💉 Injecting {count} enhanced stem cells...")
        
        stem_cells_added = 0
        for cell_id, cell in list(self.population.items())[:count]:
            if len(cell.genes) < 10:  # Max genes per cell
                # Create enhanced stem with therapeutic gene types
                stem = StemGeneModule(gene_types=['BS', 'TE', 'AC', 'TS'])
                
                # Move to same device as other genes
                if cell.genes:
                    device = next(cell.genes[0].parameters()).device
                    stem = stem.to(device)
                
                # Initialize with therapeutic-specific preferences
                stem.position_preferences = {
                    'BS': (0.2, 0.1),   # Biosensors near input
                    'TE': (0.5, 0.15),  # Effectors in middle
                    'AC': (0.8, 0.1),   # Controllers near output
                    'TS': (0.5, 0.3)    # Stems anywhere
                }
                
                cell.genes.append(stem)
                stem_cells_added += 1
                
                # Set initial morphogen field based on patient state
                if hasattr(self, 'last_biomarkers'):
                    stem.morphogen_field.data[:len(self.last_biomarkers)] = self.last_biomarkers
        
        print(f"   ✓ Added {stem_cells_added} enhanced stem cells")
        
    def _update_morphogen_fields(self):
        """Update morphogen fields for niche modeling"""
        # Group cells by spatial proximity
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, StemGeneModule) and hasattr(gene, 'position'):
                    # Find neighbors within radius
                    neighbors = []
                    for other_cell in self.population.values():
                        for other_gene in other_cell.genes:
                            if (other_gene != gene and 
                                hasattr(other_gene, 'morphogen_field') and
                                hasattr(other_gene, 'position')):
                                distance = abs(gene.position - other_gene.position)
                                if distance < 0.2:  # Neighbor radius
                                    neighbors.append(other_gene.morphogen_field)
                    
                    # Update morphogen field
                    if neighbors:
                        gene.update_morphogen(neighbors)
    
    def _enhanced_population_response(self, patient_state: dict) -> dict:
        """Enhanced response generation with stem cell features"""
        # Store biomarkers for morphogen initialization
        self.last_biomarkers = patient_state['biomarkers']
        
        # Update morphogen fields first
        self._update_morphogen_fields()
        
        # Check for stem cell asymmetric division under stress
        if patient_state.get('disease_severity', 0) > 0.8:
            self._trigger_asymmetric_divisions(patient_state)
        
        # Generate therapeutic responses from population
        therapeutic_responses = {}
        
        for cell_id, cell in self.population.items():
            cell_response = {'therapeutic': None, 'dose': 0.0, 'safety_score': 1.0}
            
            # Generate response from each therapeutic gene
            for gene in cell.genes:
                if hasattr(gene, 'generate_therapeutic'):
                    try:
                        response = gene.generate_therapeutic(patient_state)
                        if response and 'therapeutic' in response:
                            cell_response = response
                            break
                    except Exception as e:
                        print(f"   Warning: Gene {gene.gene_type} failed: {e}")
                        continue
            
            therapeutic_responses[cell_id] = cell_response
        
        return therapeutic_responses
    
    def _trigger_asymmetric_divisions(self, patient_state: dict):
        """Trigger asymmetric division in stem cells during crisis"""
        print("   🚨 High severity detected - triggering stem cell divisions...")
        
        new_daughters = []
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, StemGeneModule) and gene.commitment_level < 0.3:
                    population_stats = {
                        'BS_count': sum(1 for c in self.population.values() 
                                      for g in c.genes if isinstance(g, BiosensorGene)),
                        'TE_count': sum(1 for c in self.population.values() 
                                      for g in c.genes if isinstance(g, TherapeuticEffectorGene)),
                        'AC_count': sum(1 for c in self.population.values() 
                                      for g in c.genes if isinstance(g, AdaptiveControllerGene)),
                        'stress_level': patient_state.get('disease_severity', 0.5),
                        'diversity': 0.5,
                        'generation': self.generation
                    }
                    
                    daughter = gene.divide_asymmetrically(population_stats)
                    if daughter:
                        new_daughters.append((cell, daughter))
                        
                    # Only do a few per cycle
                    if len(new_daughters) >= 3:
                        break
        
        # Add daughters to cells
        for cell, daughter in new_daughters:
            if len(cell.genes) < 10:
                cell.genes.append(daughter)
    
    def run_treatment_cycle(self) -> Dict:
        """Run treatment cycle using FULL CORE TE-AI SYSTEM"""
        # Generate biomarkers
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        biomarkers = torch.randn(cfg.num_biomarkers, device=device)
        
        # Add disease patterns
        if self.patient_profile.get('disease') == 'autoimmune_inflammatory':
            severity = self.patient_profile.get('severity', 0.5)
            biomarkers[0] = severity * 100 + torch.randn(1, device=device).squeeze() * 10
            biomarkers[1] = severity * 50 + torch.randn(1, device=device).squeeze() * 5
            biomarkers[2] = severity * 75 + torch.randn(1, device=device).squeeze() * 15
        
        # USE CORE SYSTEM'S EVOLUTION ENGINE
        return self.therapeutic_cycle(biomarkers)
    
    def get_system_status(self) -> Dict:
        """Get current system status for reporting"""
        return {
            'population_size': len(self.population),
            'therapeutic_genes': sum(1 for c in self.population.values() 
                                   for g in c.genes 
                                   if isinstance(g, (BiosensorGene, TherapeuticEffectorGene, AdaptiveControllerGene))),
            'stem_cells': sum(1 for c in self.population.values() 
                            for g in c.genes 
                            if isinstance(g, TherapeuticStemGene)),
            'avg_fitness': np.mean([c.fitness_history[-1] if c.fitness_history else 0 
                                  for c in self.population.values()]),
            'patient_profile': self.patient_profile
        }
    
    def emergency_intervention(self, condition: str):
        """Emergency intervention for critical conditions"""
        print(f"   🚨 EMERGENCY INTERVENTION: {condition}")
        
        # Trigger rapid stem cell differentiation
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, TherapeuticStemGene):
                    gene.emergency_response(condition)

