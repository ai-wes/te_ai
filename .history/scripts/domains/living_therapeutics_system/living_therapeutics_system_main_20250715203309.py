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

def run_enhanced_therapeutic_demo():
    """Demo the enhanced therapeutic system"""
    
    print("=" * 80)
    print("ENHANCED LIVING THERAPEUTIC SYSTEM WITH ADVANCED STEM CELLS")
    print("=" * 80)
    
    # Severe autoimmune patient
    patient = {
        'id': 'ENHANCED_001',
        'disease': 'severe_autoimmune',
        'severity': 0.9,  # Very severe
        'age': 35,
        'comorbidities': ['diabetes']
    }
    
    print(f"\nPatient Profile:")
    print(f"  Disease: Severe Autoimmune Disorder")
    print(f"  Initial Severity: {patient['severity']*100:.0f}%")
    print(f"  Challenge: Rapid progression, treatment resistance")
    
    # Initialize system
    print("\nInitializing enhanced therapeutic system...")
    system = LivingTherapeuticSystem(patient)
    
    # Inject enhanced stem cells
    system._inject_enhanced_stem_cells(count=15)
    
    # Run treatment simulation
    print("\n" + "="*60)
    print("24-HOUR TREATMENT SIMULATION")
    print("="*60)
    print("Hour | IL-6 | TNF-α | CRP | Severity | Stem Activity")
    print("-"*60)
    
    severity = patient['severity']
    
    for hour in range(24):
        # Generate biomarkers
        biomarkers = torch.zeros(THERAPY_CFG.num_biomarkers)
        
        # Inflammatory markers with circadian rhythm
        circadian = np.sin(hour * np.pi / 12)
        biomarkers[0] = 20 + severity * 100 + 15 * circadian    # IL-6
        biomarkers[1] = 10 + severity * 50 + 8 * circadian      # TNF-α 
        biomarkers[2] = 5 + severity * 50                       # CRP
        biomarkers[3] = 100 + severity * 50                     # Glucose
        biomarkers[4] = 7.4 - severity * 0.1                    # pH
        
        # Get therapeutic response
        patient_state = system._comprehensive_patient_assessment(biomarkers)
        response = system._generate_population_response(patient_state)
        
        # Count active stem cells and their states
        stem_stats = {'total': 0, 'differentiating': 0, 'active_rl': 0}
        for cell in system.population.values():
            for gene in cell.genes:
                if isinstance(gene, StemGeneModule):
                    stem_stats['total'] += 1
                    if gene.commitment_level > 0.2:
                        stem_stats['differentiating'] += 1
                    if len(gene.rl_memory) > 0:
                        stem_stats['active_rl'] += 1
        
        # Apply treatment
        efficacy = response['therapeutic']['dose'].item() if hasattr(response['therapeutic']['dose'], 'item') else response['therapeutic']['dose']
        safety = response['therapeutic']['safety_score'].item() if hasattr(response['therapeutic']['safety_score'], 'item') else response['therapeutic']['safety_score']
        
        # Enhanced treatment with stem cell boost
        stem_boost = stem_stats['differentiating'] / max(stem_stats['total'], 1) * 0.2
        treatment_effect = (efficacy * safety + stem_boost) * 0.15
        severity = max(0.1, severity - treatment_effect)
        
        # Natural progression
        if hour % 6 == 5:  # Disease flare every 6 hours
            severity = min(1.0, severity + 0.05)
        
        # Report
        if hour % 2 == 0:  # Every 2 hours
            stem_activity = f"{stem_stats['differentiating']}/{stem_stats['total']} diff"
            print(f"{hour:4d} | {biomarkers[0]:4.0f} | {biomarkers[1]:5.0f} | {biomarkers[2]:3.0f} | "
                  f"{severity:8.3f} | {stem_activity}")
        
        # Check for critical events
        if hour == 12:
            print("\n💊 Mid-treatment Analysis:")
            print(f"   - Active stem cells: {stem_stats['total']}")
            print(f"   - Differentiating: {stem_stats['differentiating']}")
            print(f"   - Using RL decisions: {stem_stats['active_rl']}")
            print(f"   - Current severity: {severity:.1%}")
            print()
    
    # Final report
    print("\n" + "="*60)
    print("TREATMENT COMPLETE")
    print("="*60)
    
    improvement = (patient['severity'] - severity) / patient['severity'] * 100
    print(f"Initial Severity: {patient['severity']:.1%}")
    print(f"Final Severity: {severity:.1%}")
    print(f"Improvement: {improvement:.1f}%")
    
    # Analyze stem cell contributions
    print("\nStem Cell Analysis:")
    differentiation_events = 0
    asymmetric_divisions = 0
    
    for cell in system.population.values():
        for gene in cell.genes:
            if isinstance(gene, StemGeneModule):
                differentiation_events += len(gene.differentiation_history)
                if 'daughter' in gene.gene_id:
                    asymmetric_divisions += 1
    
    print(f"  Total differentiation events: {differentiation_events}")
    print(f"  Asymmetric divisions: {asymmetric_divisions}")
    print(f"  Final stem population: {stem_stats['total']}")
    
    # Show final therapeutic gene distribution
    gene_types = {'BS': 0, 'TE': 0, 'AC': 0, 'TS': 0}
    for cell in system.population.values():
        for gene in cell.genes:
            if isinstance(gene, BiosensorGene):
                gene_types['BS'] += 1
            elif isinstance(gene, TherapeuticEffectorGene):
                gene_types['TE'] += 1
            elif isinstance(gene, AdaptiveControllerGene):
                gene_types['AC'] += 1
            elif isinstance(gene, StemGeneModule):
                gene_types['TS'] += 1
    
    print("\nFinal Therapeutic Population:")
    for gtype, count in gene_types.items():
        print(f"  {gtype}: {count}")
    
    print(f"\nResult: {'SUCCESS' if improvement > 40 else 'PARTIAL SUCCESS'}")
    print("\nEnhanced features demonstrated:")
    print("  ✓ Asymmetric stem cell division")
    print("  ✓ Morphogen field niche modeling")
    print("  ✓ RL-based differentiation decisions")
    print("  ✓ Crisis response with stem activation")

if __name__ == "__main__":
    print(f"\nStarting enhanced therapeutic simulation at {datetime.now()}")
    run_enhanced_therapeutic_demo()
    print(f"\nCompleted at {datetime.now()}")