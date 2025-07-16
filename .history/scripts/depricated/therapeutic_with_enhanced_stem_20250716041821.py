#!/usr/bin/env python3
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
from scripts.core.stem_gene_module import StemGeneModule

# Override the default TherapeuticStemGene with our enhanced version
import living_therapeutic_system
living_therapeutic_system.TherapeuticStemGene = StemGeneModule

class EnhancedTherapeuticSystem(LivingTherapeuticSystem):
    """Therapeutic system with enhanced stem cell capabilities"""
    
    def __init__(self, patient_profile: dict):
        super().__init__(patient_profile)
        
        # Add morphogen field tracking
        self.morphogen_gradients = {}
        
        # Enable RL for stem cells
        self.use_rl_stems = True
        
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
    system = EnhancedTherapeuticSystem(patient)
    
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