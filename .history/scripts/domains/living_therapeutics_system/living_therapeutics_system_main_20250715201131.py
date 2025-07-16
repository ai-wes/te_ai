# File: living_therapeutics_system_main.py
"""
Living Therapeutic System - Main Orchestration
==============================================
Main system class that coordinates all therapeutic functionality
"""

import torch
import numpy as np
import time
import random
from typing import Dict, List
from collections import deque, defaultdict

# Import base classes
from transposable_immune_ai_production_complete import (
    ProductionGerminalCenter, cfg
)

# Import therapeutic components
from .living_therapeutics_system_genes import (
    BiosensorGene, TherapeuticEffectorGene, AdaptiveControllerGene, 
    TherapeuticStemGene
)
from .living_therapeutics_system_cells import (
    create_living_therapeutic_population, TherapeuticCellManager
)
from .living_therapeutics_system_config import THERAPY_CFG, get_device

# ============================================================================
# MAIN THERAPEUTIC SYSTEM
# ============================================================================

class LivingTherapeuticSystem:
    """Complete living therapeutic system that evolves with patient"""
    
    def __init__(self, patient_profile: Dict):
        self.patient_profile = patient_profile
        self.population = create_living_therapeutic_population(patient_profile)
        
        # Patient state tracking
        self.patient_history = deque(maxlen=1000)
        self.treatment_history = deque(maxlen=1000)
        
        # Cell management
        self.cell_manager = TherapeuticCellManager(self.population)
        
        # Initialize germinal center
        self.germinal_center = ProductionGerminalCenter()
        self.germinal_center.population = self.population
        
    def therapeutic_cycle(self, current_biomarkers: torch.Tensor) -> Dict:
        """One cycle of sense ’ plan ’ treat ’ evolve"""
        
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
    
    def run_treatment_cycle(self) -> Dict:
        """Run a single treatment cycle - simplified interface"""
        # Generate synthetic biomarkers for testing
        device = get_device()
        biomarkers = torch.randn(THERAPY_CFG.num_biomarkers, device=device)
        
        # For autoimmune patients, elevate inflammatory markers
        if self.patient_profile.get('disease') == 'autoimmune_inflammatory':
            biomarkers[0] = 5.0  # IL-6
            biomarkers[1] = 3.0  # TNF-±
            biomarkers[2] = 8.0  # CRP
        
        return self.therapeutic_cycle(biomarkers)
    
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
            device = biomarkers.device if torch.is_tensor(biomarkers) else get_device()
            
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
            device = patient_state.get('biomarkers', torch.zeros(1)).device
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
            device = get_device()
            return {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0, 'safety_score': 1.0}
        
        # Group by mode
        by_mode = defaultdict(list)
        for output in therapeutic_outputs:
            by_mode[output['mode']].append(output)
        
        # Find max therapeutic size across all outputs
        max_size = 0
        for output in therapeutic_outputs:
            therapeutic = output['therapeutic']
            if therapeutic.dim() == 2:
                size = therapeutic.shape[1]  # For [1, X] tensors, use the second dimension
            else:
                size = therapeutic.shape[0]  # For [X] tensors, use the first dimension
            max_size = max(max_size, size)
        
        # Combine within modes (avoid overdosing)
        # Get device from first therapeutic tensor
        device = therapeutic_outputs[0]['therapeutic'].device if therapeutic_outputs else get_device()
        combined = {
            'therapeutic': torch.zeros(max_size, device=device),
            'dose': 0.0,
            'safety_score': 1.0
        }
        
        for mode, outputs in by_mode.items():
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
            
            # Add to combined with safety limits
            safe_dose = min(mode_dose.item(), THERAPY_CFG.max_therapeutic_strength)
            
            # Handle different therapeutic sizes by padding or truncating
            if mode_therapeutic.dim() > 1:
                mode_therapeutic = mode_therapeutic.squeeze()
            
            # Scale by safe dose
            scaled_therapeutic = mode_therapeutic * safe_dose
            
            if mode_therapeutic.shape[0] <= max_size:
                # Pad with zeros if needed
                if mode_therapeutic.shape[0] < max_size:
                    padded = torch.zeros(max_size, device=mode_therapeutic.device)
                    padded[:mode_therapeutic.shape[0]] = scaled_therapeutic
                    combined['therapeutic'] += padded
                else:
                    # Exact size match
                    combined['therapeutic'] += scaled_therapeutic
            else:
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
            'biomarker_change': torch.randn(THERAPY_CFG.num_biomarkers, device=therapeutic.device) * actual_effect
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
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'patient_profile': self.patient_profile,
            'population_stats': self.cell_manager.get_population_stats(),
            'recent_efficacy': np.mean([
                outcome['response']['efficacy_score'] 
                for outcome in list(self.treatment_history)[-10:]
            ]) if self.treatment_history else 0.0,
            'treatment_count': len(self.treatment_history),
            'gene_distribution': self.cell_manager.count_gene_types()
        }
    
    def emergency_intervention(self, critical_condition: str):
        """Trigger emergency response throughout the system"""
        print(f"=¨ EMERGENCY INTERVENTION: {critical_condition}")
        
        # Activate emergency response in all stem genes
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, TherapeuticStemGene):
                    gene.emergency_response(critical_condition)
        
        # Balance population for emergency
        emergency_distribution = {
            'TE': 0.6,  # More effectors for treatment
            'AC': 0.2,  # Controllers for coordination
            'BS': 0.15, # Sensors for monitoring
            'TS': 0.05  # Few stem cells for adaptation
        }
        
        self.cell_manager.balance_population(emergency_distribution)