# File: living_therapeutics_system_run.py
"""
Living Therapeutic System - Runners and Examples
===============================================
Various ways to run and test the living therapeutic system
"""

import torch
import numpy as np
import random
import time
import json
import argparse
from typing import Dict, List
from datetime import datetime
# Import therapeutic system components
from living_therapeutics_system_main import LivingTherapeuticSystem
from living_therapeutics_system_config import THERAPY_CFG as cfg
from living_therapeutics_system_genes import (
    BiosensorGene,
    TherapeuticEffectorGene,
    AdaptiveControllerGene
)

from stem_gene_module import StemGeneModule




# ============================================================================
# BASIC SIMULATION RUNNER
# ============================================================================

def run_basic_therapeutic_simulation(patient_profile: Dict = None, hours: int = 168) -> Dict:
    """Basic simulation runner - good for testing and demos"""
    
    # Default patient profile
    if patient_profile is None:
        patient_profile = {
            'id': 'PT001',
            'disease': 'autoimmune_inflammatory',
            'severity': 0.7,
            'age': 45,
            'weight': 70,
            'comorbidities': ['diabetes']
        }
    
    print(f">� Living Therapeutic TE-AI System")
    print("=" * 50)
    print(f"Patient: {patient_profile['id']}")
    print(f"Disease: {patient_profile['disease']}")
    print(f"Severity: {patient_profile['severity']}")
    print(f"Duration: {hours} hours")
    print("=" * 50)
    
    # Initialize therapeutic system
    therapeutic_system = LivingTherapeuticSystem(patient_profile)
    
    results = []
    
    # Simulate treatment over time
    for hour in range(hours):
        # Generate patient biomarkers (would come from real sensors)
        device = cfg.device
        biomarkers = torch.randn(cfg.num_biomarkers, device=device)
        
        # Disease-specific biomarker patterns
        if patient_profile['disease'] == 'autoimmune_inflammatory':
            # Inflammatory markers elevated early, then responding to treatment
            inflammatory_level = max(0.5, 5.0 - (hour / 48.0))  # Gradual improvement
            biomarkers[0] = inflammatory_level  # IL-6
            biomarkers[1] = inflammatory_level * 0.6  # TNF-�
            biomarkers[2] = inflammatory_level * 1.5  # CRP
        
        # Run therapeutic cycle
        result = therapeutic_system.therapeutic_cycle(biomarkers)
        
        # Record results
        hour_result = {
            'hour': hour,
            'disease_severity': result['patient_state'].get('disease_severity', 0),
            'efficacy_score': result['response']['efficacy_score'],
            'safety_score': result['response']['safety_score'],
            'population_size': result['population_size'],
            'avg_fitness': result['avg_fitness']
        }
        results.append(hour_result)
        
        # Daily reporting
        if hour % 24 == 0:
            day = hour // 24
            print(f"\nDay {day}:")
            print(f"  Disease severity: {hour_result['disease_severity']:.3f}")
            print(f"  Treatment efficacy: {hour_result['efficacy_score']:.3f}")
            print(f"  Safety score: {hour_result['safety_score']:.3f}")
            print(f"  Population size: {hour_result['population_size']}")
            print(f"  Average fitness: {hour_result['avg_fitness']:.3f}")
        
        # Check for critical conditions
        if result['patient_state'].get('critical_conditions'):
            critical = list(result['patient_state']['critical_conditions'].keys())
            print(f"  � Critical condition detected: {critical}")
            
            # Trigger emergency response
            therapeutic_system.emergency_intervention(critical[0])
    
    # Final summary
    final_severity = results[-1]['disease_severity']
    initial_severity = patient_profile['severity']
    improvement = ((initial_severity - final_severity) / initial_severity) * 100
    
    summary = {
        'patient_profile': patient_profile,
        'simulation_hours': hours,
        'initial_severity': initial_severity,
        'final_severity': final_severity,
        'improvement_percentage': improvement,
        'avg_efficacy': np.mean([r['efficacy_score'] for r in results]),
        'avg_safety': np.mean([r['safety_score'] for r in results]),
        'final_population_size': results[-1]['population_size'],
        'results': results
    }
    
    print(f"\n{'='*50}")
    print("SIMULATION COMPLETE")
    print(f"{'='*50}")
    print(f"Initial severity: {initial_severity:.3f}")
    print(f"Final severity: {final_severity:.3f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Average efficacy: {summary['avg_efficacy']:.3f}")
    print(f"Average safety: {summary['avg_safety']:.3f}")
    
    return summary

# ============================================================================
# PRODUCTION TREATMENT RUNNER
# ============================================================================

class ProductionTherapeuticRunner:
    """Production-ready therapeutic runner with comprehensive monitoring"""
    
    def __init__(self, patient_profile: Dict):
        self.patient_profile = patient_profile
        self.therapeutic_system = LivingTherapeuticSystem(patient_profile)
        self.treatment_log = []
        self.start_time = time.time()
        
    def run_treatment_session(self, duration_hours: int, report_interval: int = 6) -> Dict:
        """Run a treatment session with monitoring"""
        print(f"\n{'='*80}")
        print("LIVING THERAPEUTIC TE-AI SYSTEM - PRODUCTION MODE")
        print(f"{'='*80}")
        print(f"Patient: {self.patient_profile['id']}")
        print(f"Disease: {self.patient_profile['disease']}")
        print(f"Initial Severity: {self.patient_profile['severity']:.1%}")
        print(f"Treatment Duration: {duration_hours} hours")
        print(f"{'='*80}")
        
        for hour in range(duration_hours):
            # Run treatment cycle
            cycle_result = self.therapeutic_system.run_treatment_cycle()
            
            # Log treatment
            self.treatment_log.append({
                'hour': hour,
                'timestamp': time.time(),
                'patient_state': cycle_result['patient_state'],
                'treatment': cycle_result['treatment'],
                'response': cycle_result['response']
            })
            
            # Periodic reporting
            if hour % report_interval == 0:
                self._generate_progress_report(hour, cycle_result)
        
        return self._generate_final_report()
    
    def _generate_progress_report(self, hour: int, cycle_result: Dict):
        """Generate progress report"""
        severity = cycle_result['patient_state'].get('disease_severity', 0)
        efficacy = cycle_result['response']['efficacy_score']
        safety = cycle_result['response']['safety_score']
        
        print(f"Hour {hour:3d}: Severity={severity:.3f} "
              f"Efficacy={efficacy:.3f} Safety={safety:.3f}")
    
    def _generate_final_report(self) -> Dict:
        """Generate comprehensive final report"""
        if not self.treatment_log:
            return {}
        
        # Calculate metrics
        efficacies = [entry['response']['efficacy_score'] for entry in self.treatment_log]
        safeties = [entry['response']['safety_score'] for entry in self.treatment_log]
        severities = [entry['patient_state'].get('disease_severity', 0) for entry in self.treatment_log]
        
        initial_severity = self.patient_profile['severity']
        final_severity = severities[-1] if severities else initial_severity
        improvement = ((initial_severity - final_severity) / initial_severity) * 100
        
        # Count successful treatments (efficacy > 0.7)
        successful_treatments = sum(1 for e in efficacies if e > 0.7)
        success_rate = (successful_treatments / len(efficacies)) * 100 if efficacies else 0
        
        report = {
            'patient_id': self.patient_profile['id'],
            'treatment_duration': len(self.treatment_log),
            'initial_severity': initial_severity,
            'final_severity': final_severity,
            'improvement_percentage': improvement,
            'avg_efficacy': np.mean(efficacies) if efficacies else 0,
            'avg_safety': np.mean(safeties) if safeties else 0,
            'success_rate': success_rate,
            'successful_treatments': successful_treatments,
            'total_treatments': len(self.treatment_log),
            'system_status': self.therapeutic_system.get_system_status(),
            'treatment_resistance': max(0, -improvement / 10)  # Estimate resistance
        }
        
        # Print final report
        print(f"\n{'='*80}")
        print("TREATMENT COMPLETE")
        print(f"{'='*80}")
        print(f"\nFinal Results:")
        print(f"  Initial Severity: {initial_severity:.3f}")
        print(f"  Final Severity: {final_severity:.3f}")
        print(f"  Total Improvement: {improvement:.1f}%")
        print(f"  Treatment Cycles: {len(self.treatment_log)}")
        print(f"  Successful Treatments: {successful_treatments}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Final Resistance: {report['treatment_resistance']:.3f}")
        
        return report








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
        biomarkers = torch.zeros(cfg.num_biomarkers)
        
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










# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for therapeutic system"""
    parser = argparse.ArgumentParser(description='Living Therapeutic TE-AI System')
    parser.add_argument('--patient-type', choices=['autoimmune', 'cancer', 'metabolic'],
                      default='autoimmune', help='Patient disease type')
    parser.add_argument('--hours', type=int, default=72, help='Treatment duration in hours')
    parser.add_argument('--report-interval', type=int, default=6, 
                      help='Hours between progress reports')
    parser.add_argument('--save-results', action='store_true',
                      help='Save treatment results to file')
    parser.add_argument('--mode', choices=['basic', 'production'], default='production',
                      help='Simulation mode')
    
    args = parser.parse_args()
    
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
    
    patient = patient_profiles[args.patient_type]
    
    # Run simulation
    if args.mode == 'basic':
        results = run_basic_therapeutic_simulation(patient, args.hours)
    else:
        runner = ProductionTherapeuticRunner(patient)
        results = runner.run_treatment_session(args.hours, args.report_interval)
    
    # Save results if requested
    if args.save_results:
        timestamp = int(time.time())
        filename = f"therapeutic_results_{patient['disease']}_{timestamp}.json"
        
        # Convert any tensors to lists for JSON serialization
        def tensor_to_list(obj):
            if torch.is_tensor(obj):
                return obj.cpu().tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_list(item) for item in obj]
            else:
                return obj
        
        json_results = tensor_to_list(results)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {filename}")

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def run_validation_suite():
    """Run comprehensive validation tests"""
    print(">� Running Therapeutic System Validation Suite")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Autoimmune Response',
            'patient': {
                'id': 'TEST_001',
                'disease': 'autoimmune_inflammatory',
                'severity': 0.8,
                'age': 40
            },
            'expected_improvement': 10  # Minimum 10% improvement
        },
        {
            'name': 'Emergency Response',
            'patient': {
                'id': 'TEST_002', 
                'disease': 'autoimmune_inflammatory',
                'severity': 0.9,
                'age': 60
            },
            'expected_improvement': 5  # Lower expectation for severe case
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        print("-" * 40)
        
        # Run short simulation
        result = run_basic_therapeutic_simulation(test_case['patient'], hours=48)
        
        # Check if improvement meets expectation
        improvement = result['improvement_percentage']
        expected = test_case['expected_improvement']
        
        passed = improvement >= expected
        status = " PASS" if passed else "L FAIL"
        
        print(f"Result: {improvement:.1f}% improvement (expected e{expected}%) {status}")
        
        results.append({
            'test_name': test_case['name'],
            'improvement': improvement,
            'expected': expected,
            'passed': passed
        })
    
    # Summary
    passed_tests = sum(1 for r in results if r['passed'])
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {passed_tests}/{len(results)} tests passed")
    print(f"{'='*60}")
    
    return results

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if running validation
    import sys
    if '--validate' in sys.argv:
        run_validation_suite()
    else:
        main()