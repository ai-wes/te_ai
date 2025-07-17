#!/usr/bin/env python3
"""
Run Living Therapeutic System with TE-AI
========================================
Real-world application of TE-AI for adaptive therapeutic intervention
"""

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
from scripts.depricated.transposable_immune_ai_production_complete import (
    ProductionGerminalCenter, generate_realistic_antigen,
    cfg, TermColors
)

# Import optimized version for performance
from scripts.run_optimized_simulation import OptimizedProductionGerminalCenter

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
# MAIN RUNNER
# ============================================================================

def run_therapeutic_simulation(args):
    """Run the living therapeutic system"""
    
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
            'severity': 0.6,
            'age': 60,
            'comorbidities': []
        },
        'metabolic': {
            'id': 'PT003',
            'disease': 'metabolic_syndrome',
            'severity': 0.7,
            'age': 50,
            'comorbidities': ['hypertension', 'obesity']
        }
    }
    
    # Select patient
    patient = patient_profiles.get(args.patient_type, patient_profiles['autoimmune'])
    
    print(f"\n{TermColors.BOLD}🧬 LIVING THERAPEUTIC TE-AI SYSTEM{TermColors.RESET}")
    print("=" * 80)
    print(f"Patient ID: {patient['id']}")
    print(f"Disease: {patient['disease']}")
    print(f"Initial Severity: {patient['severity']}")
    print(f"Comorbidities: {patient.get('comorbidities', [])}")
    print("=" * 80)
    
    # Initialize therapeutic germinal center
    therapeutic_gc = TherapeuticGerminalCenter(patient)
    therapeutic_gc.initialize_therapeutic_population(args.population_size)
    
    # Run evolution
    print(f"\nStarting {args.days}-day treatment simulation...")
    print(f"Population size: {args.population_size}")
    print(f"Evolution cycles per day: {args.cycles_per_day}")
    
    # Track outcomes
    daily_outcomes = []
    
    for day in range(args.days):
        print(f"\n{TermColors.BRIGHT_BLUE}{'='*80}{TermColors.RESET}")
        print(f"{TermColors.BRIGHT_BLUE}DAY {day + 1}{TermColors.RESET}")
        print(f"{TermColors.BRIGHT_BLUE}{'='*80}{TermColors.RESET}")
        
        day_start_severity = therapeutic_gc.patient_generator.disease_state
        
        # Multiple evolution cycles per day
        for cycle in range(args.cycles_per_day):
            therapeutic_gc.evolve_therapeutic_generation()
        
        # Daily summary
        day_end_severity = therapeutic_gc.patient_generator.disease_state
        improvement = (day_start_severity - day_end_severity) / day_start_severity * 100
        
        recent_outcomes = list(therapeutic_gc.patient_outcomes)[-args.cycles_per_day:]
        avg_efficacy = np.mean([o['efficacy'] for o in recent_outcomes])
        avg_safety = np.mean([o['safety'] for o in recent_outcomes])
        
        print(f"\n{TermColors.GREEN}Daily Summary:{TermColors.RESET}")
        print(f"  Disease severity: {day_start_severity:.3f} → {day_end_severity:.3f} "
              f"({improvement:+.1f}% improvement)")
        print(f"  Average efficacy: {avg_efficacy:.3f}")
        print(f"  Average safety: {avg_safety:.3f}")
        print(f"  Population size: {len(therapeutic_gc.population)}")
        
        # Count therapeutic genes
        gene_counts = {'BS': 0, 'TE': 0, 'AC': 0, 'TS': 0}
        for cell in therapeutic_gc.population.values():
            for gene in cell.genes:
                if isinstance(gene, BiosensorGene):
                    gene_counts['BS'] += 1
                elif isinstance(gene, TherapeuticEffectorGene):
                    gene_counts['TE'] += 1
                elif isinstance(gene, AdaptiveControllerGene):
                    gene_counts['AC'] += 1
                elif isinstance(gene, TherapeuticStemGene):
                    gene_counts['TS'] += 1
        
        print(f"  Gene distribution: BS:{gene_counts['BS']} TE:{gene_counts['TE']} "
              f"AC:{gene_counts['AC']} TS:{gene_counts['TS']}")
        
        daily_outcomes.append({
            'day': day + 1,
            'severity': day_end_severity,
            'improvement': improvement,
            'efficacy': avg_efficacy,
            'safety': avg_safety
        })
        
        # Check for remission
        if day_end_severity < 0.2:
            print(f"\n{TermColors.BRIGHT_GREEN}✅ REMISSION ACHIEVED!{TermColors.RESET}")
            break
    
    # Final report
    print(f"\n{TermColors.BOLD}{'='*80}{TermColors.RESET}")
    print(f"{TermColors.BOLD}TREATMENT COMPLETE{TermColors.RESET}")
    print(f"{TermColors.BOLD}{'='*80}{TermColors.RESET}")
    
    initial_severity = patient['severity']
    final_severity = therapeutic_gc.patient_generator.disease_state
    total_improvement = (initial_severity - final_severity) / initial_severity * 100
    
    print(f"\nOverall Results:")
    print(f"  Initial severity: {initial_severity:.3f}")
    print(f"  Final severity: {final_severity:.3f}")
    print(f"  Total improvement: {total_improvement:.1f}%")
    print(f"  Treatment days: {len(daily_outcomes)}")
    print(f"  Total generations: {therapeutic_gc.generation}")
    
    # Save results if requested
    if args.save_results:
        import json
        results = {
            'patient': patient,
            'daily_outcomes': daily_outcomes,
            'final_state': {
                'severity': final_severity,
                'improvement': total_improvement,
                'generations': therapeutic_gc.generation
            }
        }
        
        filename = f"therapeutic_results_{patient['id']}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Living Therapeutic TE-AI System')
    parser.add_argument('--patient-type', choices=['autoimmune', 'cancer', 'metabolic'],
                      default='autoimmune', help='Patient disease type')
    parser.add_argument('--days', type=int, default=7, help='Treatment duration in days')
    parser.add_argument('--cycles-per-day', type=int, default=4, 
                      help='Evolution cycles per day')
    parser.add_argument('--population-size', type=int, default=100,
                      help='Initial population size')
    parser.add_argument('--save-results', action='store_true',
                      help='Save treatment results to file')
    
    args = parser.parse_args()
    
    # Run simulation
    run_therapeutic_simulation(args)

if __name__ == "__main__":
    main()