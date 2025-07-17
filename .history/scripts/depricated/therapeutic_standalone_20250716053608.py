#!/usr/bin/env python3
"""
Standalone Living Therapeutic System Runner
==========================================
Real-world medical application of TE-AI without visualization dependencies
"""

import torch
import numpy as np
import time
import json
from collections import defaultdict
import argparse

# Import therapeutic components
from living_therapeutic_system import (
    TherapeuticConfig, THERAPY_CFG,
    BiosensorGene, TherapeuticEffectorGene, 
    AdaptiveControllerGene, TherapeuticStemGene,
    LivingTherapeuticSystem
)

# Import base TE-AI
from scripts.depricated.transposable_immune_ai_production_complete import ProductionBCell
from scripts.config import cfg

# ============================================================================
# DISEASE SIMULATION MODELS
# ============================================================================

class DiseaseModel:
    """Simulates realistic disease progression and treatment response"""
    
    def __init__(self, disease_type: str, severity: float):
        self.disease_type = disease_type
        self.severity = severity
        self.time = 0
        self.treatment_resistance = 0.0
        self.disease_parameters = self._initialize_disease_parameters()
        
    def _initialize_disease_parameters(self):
        """Set disease-specific parameters"""
        params = {
            'autoimmune_inflammatory': {
                'progression_rate': 0.01,
                'treatment_sensitivity': 0.8,
                'relapse_probability': 0.1,
                'biomarker_weights': {
                    'IL-6': 3.0,
                    'TNF-α': 2.5,
                    'CRP': 2.0,
                    'ESR': 1.5
                }
            },
            'cancer': {
                'progression_rate': 0.02,
                'treatment_sensitivity': 0.6,
                'mutation_rate': 0.05,
                'biomarker_weights': {
                    'CA-125': 5.0,
                    'CEA': 4.0,
                    'PSA': 3.0,
                    'ctDNA': 6.0
                }
            },
            'metabolic_syndrome': {
                'progression_rate': 0.005,
                'treatment_sensitivity': 0.9,
                'lifestyle_factor': 0.3,
                'biomarker_weights': {
                    'glucose': 2.0,
                    'insulin': 1.8,
                    'triglycerides': 1.5,
                    'HDL': -1.2  # Negative correlation
                }
            }
        }
        return params.get(self.disease_type, params['autoimmune_inflammatory'])
    
    def generate_biomarkers(self) -> torch.Tensor:
        """Generate realistic biomarker values"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        biomarkers = torch.zeros(THERAPY_CFG.num_biomarkers, device=device)
        
        # Base noise
        biomarkers += torch.randn(THERAPY_CFG.num_biomarkers, device=device) * 0.1
        
        # Disease-specific patterns
        if self.disease_type == 'autoimmune_inflammatory':
            # Inflammatory cascade
            biomarkers[0] = 20 + self.severity * 80      # IL-6
            biomarkers[1] = 10 + self.severity * 40      # TNF-α
            biomarkers[2] = 5 + self.severity * 45       # CRP
            biomarkers[3] = 95 + torch.randn(1, device=device).item() * 10     # Glucose (stress response)
            biomarkers[4] = 7.4 - self.severity * 0.05   # pH
            
            # Circadian variation
            circadian = np.sin(self.time * np.pi / 12)
            biomarkers[:3] *= (1 + circadian * 0.2)
            
        elif self.disease_type == 'cancer':
            # Tumor markers
            biomarkers[5] = 35 + self.severity * 465     # CA-125
            biomarkers[6] = 5 + self.severity * 95       # CEA
            biomarkers[7] = self.severity * 100          # PSA
            biomarkers[8] = self.severity * 50           # ctDNA
            
            # Metabolic stress
            biomarkers[3] = 110 + self.severity * 40     # Glucose
            biomarkers[9] = 2 + self.severity * 3        # Lactate
            
        elif self.disease_type == 'metabolic_syndrome':
            # Metabolic markers
            biomarkers[3] = 100 + self.severity * 100    # Glucose
            biomarkers[10] = 5 + self.severity * 25      # Insulin
            biomarkers[11] = 150 + self.severity * 200   # Triglycerides
            biomarkers[12] = 60 - self.severity * 20     # HDL
            biomarkers[13] = 130 + self.severity * 50    # LDL
            
        # Add treatment resistance effect
        biomarkers *= (1 + self.treatment_resistance * 0.2)
        
        self.time += 1
        return torch.abs(biomarkers)  # Ensure positive values
    
    def apply_treatment(self, treatment_effect: float, treatment_type: str):
        """Update disease state based on treatment"""
        # Calculate effective treatment considering resistance
        effective_treatment = treatment_effect * (1 - self.treatment_resistance)
        effective_treatment *= self.disease_parameters['treatment_sensitivity']
        
        # Update severity
        improvement = effective_treatment * 0.05
        self.severity = max(0.01, self.severity - improvement)
        
        # Develop resistance over time
        if treatment_effect > 0.7:  # Strong treatment
            self.treatment_resistance = min(0.8, self.treatment_resistance + 0.01)
        
        # Disease-specific responses
        if self.disease_type == 'cancer' and np.random.random() < self.disease_parameters['mutation_rate']:
            # Cancer can mutate and become more resistant
            self.treatment_resistance += 0.05
            print("   ⚠️  Disease mutation detected - increased resistance!")
        
        # Natural progression without treatment
        if treatment_effect < 0.3:
            self.severity += self.disease_parameters['progression_rate']
            self.severity = min(1.0, self.severity)

# ============================================================================
# STREAMLINED THERAPEUTIC SYSTEM
# ============================================================================

class StreamlinedTherapeuticSystem(LivingTherapeuticSystem):
    """Optimized therapeutic system for standalone operation"""
    
    def __init__(self, patient_profile: dict):
        super().__init__(patient_profile)
        self.disease_model = DiseaseModel(
            patient_profile['disease'], 
            patient_profile['severity']
        )
        
        # Performance metrics
        self.treatment_metrics = {
            'total_cycles': 0,
            'successful_treatments': 0,
            'adverse_events': 0,
            'average_efficacy': 0.0
        }
    
    def run_treatment_cycle(self) -> dict:
        """Single treatment cycle with realistic disease simulation"""
        self.treatment_metrics['total_cycles'] += 1
        
        # 1. Generate current patient state
        biomarkers = self.disease_model.generate_biomarkers()
        patient_state = self._comprehensive_patient_assessment(biomarkers)
        
        # 2. Generate therapeutic response
        therapeutic_response = self._generate_population_response(patient_state)
        
        # 3. Calculate treatment effectiveness
        treatment_efficacy = self._calculate_treatment_efficacy(
            therapeutic_response, patient_state
        )
        
        # 4. Apply treatment to disease model
        self.disease_model.apply_treatment(
            treatment_efficacy, 
            therapeutic_response.get('dominant_mode', 'mixed')
        )
        
        # 5. Evolve population based on outcome
        self._adaptive_evolution(treatment_efficacy)
        
        # 6. Update metrics
        self.treatment_metrics['average_efficacy'] = (
            self.treatment_metrics['average_efficacy'] * 0.9 + 
            treatment_efficacy * 0.1
        )
        
        if treatment_efficacy > 0.6:
            self.treatment_metrics['successful_treatments'] += 1
        
        return {
            'disease_severity': self.disease_model.severity,
            'treatment_efficacy': treatment_efficacy,
            'resistance': self.disease_model.treatment_resistance,
            'population_fitness': np.mean([
                c.fitness_history[-1] if c.fitness_history else 0 
                for c in self.population.values()
            ]),
            'therapeutic_diversity': self._calculate_therapeutic_diversity()
        }
    
    def _calculate_treatment_efficacy(self, therapeutic_response: dict, 
                                    patient_state: dict) -> float:
        """Calculate how effective the treatment is"""
        # Base efficacy from therapeutic output
        dose = therapeutic_response['therapeutic']['dose']
        base_efficacy = dose.item() if hasattr(dose, 'item') else dose
        
        # Modify based on disease state
        disease_severity = patient_state.get('disease_severity', 0.5)
        severity_factor = 1.0 - (disease_severity - 0.5) * 0.5
        
        # Safety modifier
        safety_score = therapeutic_response['therapeutic']['safety_score']
        safety = safety_score.item() if hasattr(safety_score, 'item') else safety_score
        
        # Critical condition bonus
        if patient_state.get('critical_conditions'):
            if therapeutic_response['num_controllers'] > 0:
                base_efficacy *= 1.2  # Bonus for having controllers during crisis
        
        # Final efficacy
        efficacy = base_efficacy * severity_factor * safety
        
        # Add noise for realism
        efficacy += np.random.normal(0, 0.05)
        
        return np.clip(efficacy, 0.0, 1.0)
    
    def _adaptive_evolution(self, treatment_efficacy: float):
        """Evolve population based on treatment success"""
        # Calculate fitness for all cells
        for cell in self.population.values():
            # Base fitness on treatment outcome
            base_fitness = treatment_efficacy
            
            # Bonus for therapeutic genes
            therapeutic_bonus = 0
            for gene in cell.genes:
                if isinstance(gene, BiosensorGene):
                    therapeutic_bonus += 0.1
                elif isinstance(gene, TherapeuticEffectorGene):
                    therapeutic_bonus += 0.15
                elif isinstance(gene, AdaptiveControllerGene):
                    therapeutic_bonus += 0.2
                elif isinstance(gene, TherapeuticStemGene):
                    if gene.commitment_level > 0.5:
                        therapeutic_bonus += 0.1
            
            cell.fitness_history.append(base_fitness + therapeutic_bonus)
        
        # Selection pressure based on performance
        if treatment_efficacy < 0.4:
            # Poor performance - increase evolution
            self._apply_selection_pressure(pressure=0.5)
            self._trigger_stem_differentiation()
        elif treatment_efficacy > 0.8:
            # Good performance - mild selection
            self._apply_selection_pressure(pressure=0.2)
    
    def _apply_selection_pressure(self, pressure: float):
        """Apply selection and reproduction"""
        population_list = list(self.population.items())
        fitnesses = [cell.fitness_history[-1] if cell.fitness_history else 0 
                    for _, cell in population_list]
        
        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        # Keep top performers
        num_survivors = int(len(population_list) * (1 - pressure))
        survivors = [population_list[i][0] for i in sorted_indices[:num_survivors]]
        
        # Remove poor performers
        for i in sorted_indices[num_survivors:]:
            cell_id = population_list[i][0]
            del self.population[cell_id]
        
        # Reproduce from survivors
        while len(self.population) < cfg.initial_population:
            parent_id = np.random.choice(survivors)
            parent = self.population[parent_id]
            
            # Clone with mutation
            child = parent.clone()
            # Apply mutation if the method exists
            mutation_rate = cfg.mutation_rate * (2 - pressure)
            if hasattr(child, '_mutate'):
                # _mutate takes no arguments, uses internal mutation rate
                child._mutate()
            elif hasattr(child, 'mutate'):
                child.mutate(mutation_rate)
            self.population[child.cell_id] = child
    
    def _trigger_stem_differentiation(self):
        """Guide stem cells to differentiate based on needs"""
        # Count current therapeutic genes
        gene_counts = defaultdict(int)
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, BiosensorGene):
                    gene_counts['sensor'] += 1
                elif isinstance(gene, TherapeuticEffectorGene):
                    gene_counts['effector'] += 1
                elif isinstance(gene, AdaptiveControllerGene):
                    gene_counts['controller'] += 1
        
        # Determine what's needed
        total_therapeutic = sum(gene_counts.values())
        if total_therapeutic == 0:
            needed = 'sensor'  # Start with sensing
        elif gene_counts['sensor'] < 10:
            needed = 'sensor'
        elif gene_counts['effector'] < 20:
            needed = 'effector'
        elif gene_counts['controller'] < 5:
            needed = 'controller'
        else:
            needed = None
        
        # Differentiate stem cells
        if needed:
            for cell in list(self.population.values())[:50]:
                for gene in cell.genes:
                    if isinstance(gene, TherapeuticStemGene) and gene.commitment_level < 0.5:
                        # Force differentiation
                        if needed == 'sensor':
                            target_type = 'BS'
                        elif needed == 'effector':
                            target_type = 'TE'
                        else:
                            target_type = 'AC'
                        
                        gene.emergency_response(f"need_{needed}")
                        break
    
    def _calculate_therapeutic_diversity(self) -> float:
        """Calculate diversity of therapeutic approaches"""
        gene_types = defaultdict(int)
        therapeutic_modes = set()
        
        for cell in self.population.values():
            for gene in cell.genes:
                gene_types[type(gene).__name__] += 1
                if isinstance(gene, TherapeuticEffectorGene):
                    therapeutic_modes.add(gene.therapeutic_mode)
        
        # Diversity score
        type_diversity = len(gene_types) / 5.0  # Normalize
        mode_diversity = len(therapeutic_modes) / len(THERAPY_CFG.therapeutic_modes)
        
        return (type_diversity + mode_diversity) / 2

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_therapeutic_treatment(args):
    """Run complete therapeutic treatment simulation"""
    
    # Patient profiles
    patient_profiles = {
        'autoimmune': {
            'id': 'PT001',
            'disease': 'autoimmune_inflammatory',
            'severity': 0.8,
            'age': 45,
            'comorbidities': ['diabetes']
        },
        'cancer': {
            'id': 'PT002', 
            'disease': 'cancer',
            'severity': 0.7,
            'age': 60,
            'comorbidities': ['hypertension']
        },
        'metabolic': {
            'id': 'PT003',
            'disease': 'metabolic_syndrome', 
            'severity': 0.6,
            'age': 50,
            'comorbidities': ['obesity', 'sleep_apnea']
        }
    }
    
    patient = patient_profiles[args.patient_type]
    
    print(f"\n{'='*80}")
    print(f"LIVING THERAPEUTIC TE-AI SYSTEM")
    print(f"{'='*80}")
    print(f"Patient: {patient['id']}")
    print(f"Disease: {patient['disease'].replace('_', ' ').title()}")
    print(f"Initial Severity: {patient['severity']:.1%}")
    print(f"Treatment Duration: {args.hours} hours")
    print(f"{'='*80}\n")
    
    # Initialize therapeutic system
    print("Initializing therapeutic population...")
    system = StreamlinedTherapeuticSystem(patient)
    
    # Treatment tracking
    treatment_log = []
    hourly_summaries = []
    
    # Run treatment
    print(f"\nStarting treatment simulation...\n")
    
    for hour in range(args.hours):
        # Run treatment cycle
        outcome = system.run_treatment_cycle()
        treatment_log.append(outcome)
        
        # Hourly report
        if hour % args.report_interval == 0:
            print(f"Hour {hour:3d}: Severity={outcome['disease_severity']:.3f} "
                  f"Efficacy={outcome['treatment_efficacy']:.3f} "
                  f"Resistance={outcome['resistance']:.3f}")
            
            # Check critical conditions
            if outcome['disease_severity'] > 0.9:
                print("   ⚠️  CRITICAL: Disease severity very high!")
            elif outcome['disease_severity'] < 0.2:
                print("   ✅ IMPROVING: Approaching remission")
        
        # Daily summary
        if (hour + 1) % 24 == 0:
            day = (hour + 1) // 24
            daily_outcomes = treatment_log[-24:]
            
            avg_severity = np.mean([o['disease_severity'] for o in daily_outcomes])
            avg_efficacy = np.mean([o['treatment_efficacy'] for o in daily_outcomes])
            improvement = (patient['severity'] - avg_severity) / patient['severity'] * 100
            
            print(f"\n{'='*60}")
            print(f"DAY {day} SUMMARY")
            print(f"{'='*60}")
            print(f"Average Severity: {avg_severity:.3f}")
            print(f"Average Efficacy: {avg_efficacy:.3f}")
            print(f"Total Improvement: {improvement:.1f}%")
            print(f"Population Fitness: {outcome['population_fitness']:.3f}")
            print(f"Therapeutic Diversity: {outcome['therapeutic_diversity']:.3f}")
            
            hourly_summaries.append({
                'day': day,
                'severity': avg_severity,
                'efficacy': avg_efficacy,
                'improvement': improvement
            })
            
            # Check for remission
            if avg_severity < 0.15:
                print(f"\n🎉 REMISSION ACHIEVED after {day} days!")
                break
    
    # Final report
    print(f"\n{'='*80}")
    print(f"TREATMENT COMPLETE")
    print(f"{'='*80}")
    
    initial_severity = patient['severity']
    final_severity = system.disease_model.severity
    total_improvement = (initial_severity - final_severity) / initial_severity * 100
    
    print(f"\nFinal Results:")
    print(f"  Initial Severity: {initial_severity:.3f}")
    print(f"  Final Severity: {final_severity:.3f}")
    print(f"  Total Improvement: {total_improvement:.1f}%")
    print(f"  Treatment Cycles: {system.treatment_metrics['total_cycles']}")
    print(f"  Successful Treatments: {system.treatment_metrics['successful_treatments']}")
    print(f"  Success Rate: {system.treatment_metrics['successful_treatments'] / system.treatment_metrics['total_cycles']:.1%}")
    print(f"  Final Resistance: {system.disease_model.treatment_resistance:.3f}")
    
    # Save results
    if args.save_results:
        results = {
            'patient': patient,
            'treatment_duration': len(treatment_log),
            'final_state': {
                'severity': final_severity,
                'improvement': total_improvement,
                'resistance': system.disease_model.treatment_resistance
            },
            'metrics': system.treatment_metrics,
            'daily_summaries': hourly_summaries,
            'treatment_log': treatment_log[-100:] if args.save_full_log else []
        }
        
        filename = f"therapeutic_results_{patient['disease']}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, torch.Tensor) else x)
        print(f"\nResults saved to: {filename}")
    
    return final_severity < 0.2  # Success if severity reduced below 20%

def main():
    parser = argparse.ArgumentParser(
        description='Living Therapeutic TE-AI System - Standalone Runner'
    )
    parser.add_argument(
        '--patient-type', 
        choices=['autoimmune', 'cancer', 'metabolic'],
        default='autoimmune',
        help='Type of disease to treat'
    )
    parser.add_argument(
        '--hours',
        type=int,
        default=168,  # One week
        help='Treatment duration in hours'
    )
    parser.add_argument(
        '--report-interval',
        type=int,
        default=6,
        help='Hours between progress reports'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save treatment results to JSON file'
    )
    parser.add_argument(
        '--save-full-log',
        action='store_true',
        help='Save complete treatment log (large file)'
    )
    
    args = parser.parse_args()
    
    # Run simulation
    success = run_therapeutic_treatment(args)
    
    # Exit code based on success
    exit(0 if success else 1)

if __name__ == "__main__":
    main()