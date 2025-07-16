#     │ - LivingTherapeuticSystem: Main orchestrator class                                                                                         │#!/usr/bin/env python3
"""
Living Therapeutic System with Enhanced Stem Cells
=================================================
Integrates the new biologically-inspired stem cells into the therapeutic system
"""

import torch
import numpy as np
from datetime import datetime

# Import therapeutic system
from living_therapeutic_system import (
    LivingTherapeuticSystem, TherapeuticConfig, THERAPY_CFG,
    BiosensorGene, TherapeuticEffectorGene, AdaptiveControllerGene
)

# Import enhanced stem cells
from stem_gene_module import StemGeneModule

# File: living_therapeutic_system.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque, defaultdict
import uuid
import random
import time
from torch_geometric.data import Data, Batch

# Import from production system
from transposable_immune_ai_production_complete import (
     QuantumGeneModule, 
)
from living_therapeutics_system_config import THERAPY_CFG as cfg
from living_therapeutics_system_genes import (
    TherapeuticSeedGene,
    TherapeuticStemGene,
    BiosensorGene,
    TherapeuticEffectorGene,
    AdaptiveControllerGene
)

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Optional
import time
import argparse

# Import therapeutic system
from living_therapeutic_system import (
    LivingTherapeuticSystem, TherapeuticConfig, THERAPY_CFG,
    BiosensorGene, TherapeuticEffectorGene, AdaptiveControllerGene,
    TherapeuticStemGene, create_living_therapeutic_population
)

# Import main TE-AI system
from transposable_immune_ai_production_complete import (
    ProductionGerminalCenter, generate_realistic_antigen,
    cfg, TermColors
)

# Import optimized version for performance
from fast_optimized_te_ai import OptimizedProductionGerminalCenter

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
        # Use CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        biomarkers = torch.randn(THERAPY_CFG.num_biomarkers, device=device) * 0.1
        
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
# INTEGRATED THERAPEUTIC GERMINAL CENTER
# ============================================================================

class TherapeuticGerminalCenter(OptimizedProductionGerminalCenter):
    """Specialized germinal center for therapeutic evolution"""
    
    def __init__(self, patient_profile: Dict):
        super().__init__()
        self.patient_profile = patient_profile
        self.patient_generator = PatientDataGenerator(patient_profile)
        self.therapeutic_system = None  # Will be set after population init
        
        # Therapeutic-specific metrics
        self.treatment_history = deque(maxlen=1000)
        self.patient_outcomes = deque(maxlen=1000)
        
    def initialize_therapeutic_population(self, size: int = 100):
        """Initialize with therapeutic genes"""
        self.population = create_living_therapeutic_population(self.patient_profile)
        
        # Add diversity
        while len(self.population) < size:
            # Clone and mutate successful cells
            if self.population:
                parent = random.choice(list(self.population.values()))
                child = self.clone_pool.fast_clone(parent)
                self.population[child.cell_id] = child
        
        print(f"   Initialized therapeutic population with {len(self.population)} cells")
        
    def evolve_therapeutic_generation(self):
        """Evolution cycle for therapeutic system"""
        generation_start = time.time()
        self.generation += 1
        
        print(f"\n{'='*80}")
        print(f"THERAPEUTIC GENERATION {self.generation}")
        print(f"{'='*80}")
        
        # Generate patient data
        biomarkers = self.patient_generator.generate_biomarkers()
        
        # Phase 1: Patient assessment
        print("\n🏥 Phase 1: Patient Assessment")
        patient_state = self._assess_patient(biomarkers)
        
        # Convert to antigens for evolution
        antigens = [TherapeuticAntigenAdapter.biomarkers_to_antigen(
            biomarkers, patient_state
        ) for _ in range(cfg.batch_size)]
        
        # Phase 2: Therapeutic response generation
        print("\n💊 Phase 2: Therapeutic Response")
        treatment = self._generate_treatment(patient_state)
        
        # Phase 3: Evaluate population fitness
        print("\n📊 Phase 3: Fitness Evaluation")
        fitness_scores = self._evaluate_therapeutic_fitness(antigens, treatment)
        
        # Phase 4: Evolution based on therapeutic success
        print("\n🧬 Phase 4: Selection and Evolution")
        self._selection_and_reproduction_fast(fitness_scores)
        
        # Phase 5: Patient response simulation
        print("\n🔬 Phase 5: Patient Response")
        patient_response = self._simulate_patient_response(treatment)
        self.patient_generator.update_disease_state(patient_response['efficacy_score'])
        
        # Phase 6: Adaptation and learning
        print("\n🧠 Phase 6: Adaptation")
        self._adapt_to_patient_response(patient_response)
        
        # Record outcomes
        self.treatment_history.append(treatment)
        self.patient_outcomes.append({
            'generation': self.generation,
            'disease_state': self.patient_generator.disease_state,
            'efficacy': patient_response['efficacy_score'],
            'safety': patient_response['safety_score']
        })
        
        gen_time = time.time() - generation_start
        print(f"\n⏱️  Generation {self.generation} completed in {gen_time:.2f}s")
        print(f"   Disease severity: {self.patient_generator.disease_state:.3f}")
        print(f"   Treatment efficacy: {patient_response['efficacy_score']:.3f}")
        
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
        therapeutic = treatment['therapeutic']['therapeutic']
        dose = treatment['therapeutic']['dose']
        safety = treatment['therapeutic']['safety_score']
        
        # Calculate treatment effect
        effect_magnitude = (therapeutic.sum() * dose).item() / 20.0  # Normalize
        
        # Patient-specific response
        response_variability = np.random.normal(1.0, 0.15)
        actual_effect = effect_magnitude * response_variability
        
        # Generate response metrics
        response = {
            'efficacy_score': min(1.0, actual_effect),
            'safety_score': safety.item(),
            'response_speed': np.random.uniform(0.4, 0.9),
            'biomarker_change': torch.randn(THERAPY_CFG.num_biomarkers, device=therapeutic.device) * actual_effect
        }
        
        # Check for adverse events
        if np.random.random() < (1 - safety.item()) * 0.05:
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
                    gene.guided_differentiation(
                        {'encoded_state': torch.randn(cfg.hidden_dim, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), 
                         'disease_severity': self.patient_generator.disease_state},
                        population_state
                    )





# ============================================================================
# INTEGRATION WITH TE-AI SYSTEM
# ============================================================================

def create_living_therapeutic_population(patient_profile: Dict) -> Dict:
    """Initialize population for specific patient"""
    population = {}
    
    # Analyze patient to determine initial composition
    disease_type = patient_profile.get('disease', 'unknown')
    severity = patient_profile.get('severity', 0.5)
    
    # Create founder population
    for i in range(cfg.initial_population):
        genes = []
        
        # Seed genes based on disease
        if i < 10:  # Biosensors
            genes.append(BiosensorGene())
        elif i < 25:  # Therapeutic effectors
            mode = random.choice(THERAPY_CFG.therapeutic_modes)
            genes.append(TherapeuticEffectorGene(mode))
        elif i < 30:  # Controllers
            genes.append(AdaptiveControllerGene())
        
        # Add stem genes
        num_stem = random.randint(2, 5)
        for _ in range(num_stem):
            genes.append(TherapeuticStemGene())
        
        # Add some quantum genes for exploration
        if random.random() < 0.1:
            genes.append(QuantumGeneModule('QT', 0))
        
        # Create therapeutic cell
        from transposable_immune_ai_production_complete import ProductionBCell
        cell = ProductionBCell(genes)
        cell.therapeutic_profile = {
            'specialization': 'therapeutic',
            'patient_id': patient_profile.get('id'),
            'disease_target': disease_type
        }
        
        population[cell.cell_id] = cell
    
    return population

# ============================================================================
# THERAPEUTIC EVOLUTION CYCLE
# ============================================================================

class LivingTherapeuticSystem:
    """Complete living therapeutic system that evolves with patient"""
    
    def __init__(self, patient_profile: Dict):
        self.patient_profile = patient_profile
        self.population = create_living_therapeutic_population(patient_profile)
        
        # Patient state tracking
        self.patient_history = deque(maxlen=1000)
        self.treatment_history = deque(maxlen=1000)

        
        # Add morphogen field tracking
        self.morphogen_gradients = {}
        
        # Enable RL for stem cells
        self.use_rl_stems = True


        # Initialize germinal center
        from transposable_immune_ai_production_complete import ProductionGerminalCenter
        self.germinal_center = ProductionGerminalCenter()
        self.germinal_center.population = self.population
        
    def therapeutic_cycle(self, current_biomarkers: torch.Tensor) -> Dict:
        """One cycle of sense → plan → treat → evolve"""
        
        # 1. Sense patient state
        patient_state = self._comprehensive_patient_assessment(current_biomarkers)
        
        # 2. Population generates therapeutic response
        therapeutic_response = self._generate_population_response(patient_state)
        
        # 3. Apply treatment and simulate patient response
        patient_response = self._simulate_patient_response(therapeutic_response)
        
        # 4. Evolve based on treatment success
        self._evolve_therapeutic_population(patient_response)
        
        # 5. Record history
        self.patient_history.append(patient_state)
        self.treatment_history.append(therapeutic_response)
        
        return {
            'patient_state': patient_state,
            'treatment': therapeutic_response,
            'response': patient_response,
            'population_size': len(self.population),
            'avg_fitness': np.mean([c.fitness_history[-1] if c.fitness_history else 0 
                                   for c in self.population.values()])
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
            device = biomarkers.device
            patient_state['encoded_state'] = torch.zeros(cfg.hidden_dim, device=device)
            patient_state['disease_probabilities'] = torch.zeros(3, device=device)
            patient_state['critical_conditions'] = {}
            patient_state['disease_severity'] = 0.5  # Default moderate severity
        
        return patient_state
    
    def _generate_population_response(self, patient_state: Dict) -> Dict:
        """Generate coordinated therapeutic response from population"""
        therapeutic_outputs = []
        controller_plans = []
        
        for cell in self.population.values():
            for gene in cell.genes:
                # Therapeutic effectors generate treatments
                if isinstance(gene, TherapeuticEffectorGene):
                    therapeutic = gene.generate_therapeutic(patient_state)
                    therapeutic_outputs.append(therapeutic)
                
                # Controllers create treatment plans
                elif isinstance(gene, AdaptiveControllerGene):
                    plan = gene.plan_treatment(
                        list(self.patient_history)[-24:],
                        patient_state
                    )
                    controller_plans.append(plan)
                
                # Stem genes might differentiate
                elif isinstance(gene, TherapeuticStemGene):
                    population_state = {
                        'cells': list(self.population.values()),
                        'avg_fitness': np.mean([c.fitness_history[-1] if c.fitness_history else 0 
                                              for c in self.population.values()]),
                        'diversity': len(set(g.gene_type for c in self.population.values() 
                                           for g in c.genes))
                    }
                    gene.guided_differentiation(patient_state, population_state)
        
        # Combine therapeutic outputs
        if therapeutic_outputs:
            combined_therapeutic = self._combine_therapeutics(therapeutic_outputs)
        else:
            # Create empty therapeutic on correct device
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            combined_therapeutic = {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0}
        
        # Use best treatment plan
        if controller_plans:
            best_plan = max(controller_plans, 
                          key=lambda p: p.get('objectives', [0])[0])  # Sort by efficacy
        else:
            best_plan = {'plan': [], 'horizon': 24}
        
        return {
            'therapeutic': combined_therapeutic,
            'plan': best_plan,
            'num_effectors': len(therapeutic_outputs),
            'num_controllers': len(controller_plans)
        }
    
    def _combine_therapeutics(self, therapeutic_outputs: List[Dict]) -> Dict:
        """Intelligently combine multiple therapeutic outputs"""
        if not therapeutic_outputs:
            # Return empty therapeutic on correct device
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
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
            safe_dose = min(mode_dose.item(), THERAPY_CFG.max_therapeutic_strength)
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
        # Simplified patient response model
        therapeutic = treatment['therapeutic']['therapeutic']
        dose = treatment['therapeutic']['dose']
        
        # Calculate therapeutic effect
        base_effect = (therapeutic.sum() * dose).item()
        
        # Add variability based on patient
        patient_variability = random.gauss(1.0, 0.2)
        actual_effect = base_effect * patient_variability
        
        # Determine response
        response = {
            'efficacy_score': min(1.0, actual_effect / 10.0),
            'safety_score': treatment['therapeutic']['safety_score'].item(),
            'response_speed': random.uniform(0.3, 0.9),
            'efficiency_score': dose.item(),  # Lower dose = more efficient
            'biomarker_change': torch.randn(THERAPY_CFG.num_biomarkers, device=treatment['therapeutic']['therapeutic'].device) * actual_effect
        }
        
        # Check for adverse events
        if random.random() < (1 - response['safety_score']) * 0.1:
            response['adverse_event'] = True
            response['efficacy_score'] *= 0.5
        
        return response
    
    def _evolve_therapeutic_population(self, patient_response: Dict):
        """Evolve population based on treatment success"""
        # Calculate fitness for each cell based on contribution
        for cell in self.population.values():
            # Base fitness on treatment outcome
            cell_fitness = patient_response['efficacy_score'] * patient_response['safety_score']
            
            # Bonus for cells with therapeutic genes
            therapeutic_bonus = sum(
                0.1 for gene in cell.genes 
                if isinstance(gene, (BiosensorGene, TherapeuticEffectorGene, AdaptiveControllerGene))
            )
            
            cell_fitness += therapeutic_bonus
            cell.fitness_history.append(cell_fitness)
        
        # Trigger evolution if response is poor
        if patient_response['efficacy_score'] < 0.5:
            print("   Poor treatment response - triggering evolution!")
            stress_level = 1.0 - patient_response['efficacy_score']
            
            # Increase transposition
            for cell in list(self.population.values())[:50]:
                for gene in cell.genes:
                    if gene.is_active:
                        child = gene.transpose(stress_level, 0.5)
                        if child:
                            # Add to population if successful
                            pass
        
        # Learn from response
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, AdaptiveControllerGene):
                    gene.learn_from_response(
                        treatment['therapeutic'],
                        patient_response
                    )




        
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
    
    def _generate_population_response(self, patient_state: dict) -> dict:
        """Enhanced response generation with stem cell features"""
        # Store biomarkers for morphogen initialization
        self.last_biomarkers = patient_state['biomarkers']
        
        # Update morphogen fields first
        self._update_morphogen_fields()
        
        # Check for stem cell asymmetric division under stress
        if patient_state.get('disease_severity', 0) > 0.8:
            self._trigger_asymmetric_divisions(patient_state)
        
        # Continue with normal response
        return super()._generate_population_response(patient_state)
    
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

