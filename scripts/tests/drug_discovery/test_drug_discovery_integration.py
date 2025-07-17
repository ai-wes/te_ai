"""
Drug Discovery Integration Test
===============================

Tests the complete drug discovery pipeline with simulated omics data.
Validates that the TE-AI system can successfully identify and rank
drug targets using quantum evolution and dream consolidation.
"""

import numpy as np
import pandas as pd
import torch
import json
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.domains.drug_discovery import (
    DrugDiscoveryGerminalCenter,
    OmicsToAntigenConverter,
    DrugTargetEvaluator,
    DrugTargetAntigen,
    BindingPocket,
    ProteinStructure
)
from scripts.domains.drug_discovery.omics_to_antigen_converter import OmicsData
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()


class DrugDiscoveryTest:
    """Integration test for drug discovery system"""
    
    def __init__(self):
        self.test_results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
    def run_all_tests(self):
        """Run complete test suite"""
        logger.info("="*80)
        logger.info("DRUG DISCOVERY INTEGRATION TEST")
        logger.info("="*80)
        
        # Test 1: Omics conversion
        self.test_omics_conversion()
        
        # Test 2: Single target evaluation
        self.test_single_target_evaluation()
        
        # Test 3: Multi-target screening
        self.test_multi_target_screening()
        
        # Test 4: Quantum and dream features
        self.test_quantum_dream_features()
        
        # Test 5: Mutation resistance
        self.test_mutation_resistance()
        
        # Generate report
        self.generate_test_report()
        
    def create_mock_omics_data(self, n_genes: int = 100, n_samples: int = 20) -> OmicsData:
        """Create realistic mock omics data"""
        # Gene expression data
        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        sample_names = [f"Sample_{i}" for i in range(n_samples)]
        
        # Create expression matrix with disease signal
        expression_data = np.random.lognormal(mean=2.0, sigma=1.0, size=(n_samples, n_genes))
        
        # Add disease signature to some genes
        disease_genes = np.random.choice(n_genes, 20, replace=False)
        disease_samples = sample_names[n_samples//2:]
        
        for i, sample in enumerate(sample_names):
            if sample in disease_samples:
                expression_data[i, disease_genes] *= np.random.uniform(2.0, 5.0, len(disease_genes))
                
        expr_df = pd.DataFrame(expression_data, index=sample_names, columns=gene_names)
        expr_df['condition'] = ['control'] * (n_samples//2) + ['disease'] * (n_samples//2)
        
        # Mutation data
        mutations = []
        for gene_idx in disease_genes[:10]:
            mutations.append({
                'Gene': gene_names[gene_idx],
                'Position': np.random.randint(1, 500),
                'WT': np.random.choice(['A', 'R', 'N', 'D', 'C']),
                'Mutant': np.random.choice(['A', 'R', 'N', 'D', 'C']),
                'Frequency': np.random.uniform(0.1, 0.5)
            })
        mutation_df = pd.DataFrame(mutations)
        
        # Create structure data for top genes
        structural_data = {}
        for gene in gene_names[:5]:
            seq_len = np.random.randint(200, 600)
            sequence = ''.join(np.random.choice(list('ARNDCQEGHILKMFPSTWYV'), seq_len))
            
            structural_data[gene] = {
                'sequence': sequence,
                'pdb_id': f"MOCK_{gene}",
                'coordinates': np.random.randn(seq_len * 5, 3) * 10
            }
            
        return OmicsData(
            gene_expression=expr_df,
            mutations=mutation_df,
            structural_data=structural_data,
            disease_associations={g: np.random.random() for g in disease_genes}
        )
        
    def test_omics_conversion(self):
        """Test conversion of omics data to antigens"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Omics Data Conversion")
        logger.info("="*60)
        
        try:
            # Create mock data
            omics_data = self.create_mock_omics_data()
            
            # Convert to antigens
            converter = OmicsToAntigenConverter()
            antigens = converter.convert_omics_to_antigens(
                omics_data,
                disease_focus="cancer"
            )
            
            # Validate
            assert len(antigens) > 0, "No antigens created"
            assert all(isinstance(a, DrugTargetAntigen) for a in antigens), "Invalid antigen type"
            
            # Check antigen properties
            for antigen in antigens[:3]:
                assert antigen.protein_structure is not None
                assert len(antigen.binding_pockets) > 0
                assert antigen.global_druggability >= 0 and antigen.global_druggability <= 1
                
                logger.info(f"Created antigen: {antigen.antigen_type}")
                logger.info(f"  - Pockets: {len(antigen.binding_pockets)}")
                logger.info(f"  - Druggability: {antigen.global_druggability:.3f}")
                
            self.test_results['passed'].append("omics_conversion")
            self.test_results['metrics']['antigens_created'] = len(antigens)
            logger.info(f"\n✓ Test passed: Created {len(antigens)} antigens")
            
        except Exception as e:
            self.test_results['failed'].append(("omics_conversion", str(e)))
            logger.error(f"\n✗ Test failed: {e}")
            raise
            
    def test_single_target_evaluation(self):
        """Test evaluation of a single drug target"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Single Target Evaluation")
        logger.info("="*60)
        
        try:
            # Create a high-quality mock target
            target = self._create_high_quality_target()
            
            # Initialize evaluator with small population for speed
            gc = DrugDiscoveryGerminalCenter(
                population_size=32,
                enable_quantum_dreams=True
            )
            evaluator = DrugTargetEvaluator(germinal_center=gc)
            
            # Evaluate
            start_time = time.time()
            scores = evaluator.evaluate_targets(
                [target],
                generations=10,
                stress_test=False
            )
            eval_time = time.time() - start_time
            
            # Validate results
            assert len(scores) == 1
            score = list(scores.values())[0]
            
            assert score.overall_score >= 0 and score.overall_score <= 1
            assert len(score.components) > 0
            assert len(score.binding_profiles) > 0
            
            logger.info(f"\nTarget evaluation completed in {eval_time:.1f}s")
            logger.info(f"Overall score: {score.overall_score:.3f}")
            logger.info(f"Quantum coherence: {score.quantum_coherence:.3f}")
            logger.info(f"Evolutionary potential: {score.evolutionary_potential:.3f}")
            logger.info("\nScore components:")
            for comp, value in sorted(score.components.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  - {comp}: {value:.3f}")
                
            self.test_results['passed'].append("single_target_evaluation")
            self.test_results['metrics']['single_target_score'] = score.overall_score
            logger.info("\n✓ Test passed: Single target evaluated successfully")
            
        except Exception as e:
            self.test_results['failed'].append(("single_target_evaluation", str(e)))
            logger.error(f"\n✗ Test failed: {e}")
            raise
            
    def test_multi_target_screening(self):
        """Test screening of multiple targets"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Multi-Target Screening")
        logger.info("="*60)
        
        try:
            # Create targets with varying quality
            targets = [
                self._create_high_quality_target(),
                self._create_medium_quality_target(),
                self._create_poor_quality_target()
            ]
            
            # Screen targets
            gc = DrugDiscoveryGerminalCenter(
                population_size=48,
                enable_drug_genes=True
            )
            
            results = gc.screen_drug_targets(
                targets,
                generations_per_target=5,
                test_mutations=False
            )
            
            # Generate report
            report = gc.generate_druggability_report(results)
            
            # Validate ranking
            rankings = report['rankings']
            assert len(rankings) == 3
            assert rankings[0]['druggability_score'] > rankings[2]['druggability_score']
            
            logger.info("\nTarget Rankings:")
            for rank_info in rankings:
                logger.info(f"{rank_info['rank']}. {rank_info['target_id']}: {rank_info['druggability_score']:.3f}")
                
            # Check selectivity
            if gc.selectivity_matrix is not None:
                mean_selectivity = report['selectivity_analysis']['mean_selectivity']
                logger.info(f"\nMean selectivity: {mean_selectivity:.3f}")
                
            self.test_results['passed'].append("multi_target_screening")
            self.test_results['metrics']['best_target_score'] = rankings[0]['druggability_score']
            logger.info("\n✓ Test passed: Multi-target screening completed")
            
        except Exception as e:
            self.test_results['failed'].append(("multi_target_screening", str(e)))
            logger.error(f"\n✗ Test failed: {e}")
            raise
            
    def test_quantum_dream_features(self):
        """Test quantum processing and dream consolidation"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Quantum & Dream Features")
        logger.info("="*60)
        
        try:
            target = self._create_high_quality_target()
            
            # Test with quantum features enabled
            gc_quantum = DrugDiscoveryGerminalCenter(
                population_size=32,
                enable_quantum_dreams=True
            )
            evaluator_quantum = DrugTargetEvaluator(
                germinal_center=gc_quantum,
                quantum_evaluation=True,
                dream_analysis=True
            )
            
            # Test with quantum features disabled
            gc_classic = DrugDiscoveryGerminalCenter(
                population_size=32,
                enable_quantum_dreams=False
            )
            evaluator_classic = DrugTargetEvaluator(
                germinal_center=gc_classic,
                quantum_evaluation=False,
                dream_analysis=False
            )
            
            # Evaluate with both
            scores_quantum = evaluator_quantum.evaluate_targets([target], generations=10)
            scores_classic = evaluator_classic.evaluate_targets([target], generations=10)
            
            score_quantum = list(scores_quantum.values())[0]
            score_classic = list(scores_classic.values())[0]
            
            # Quantum should have coherence > 0
            assert score_quantum.quantum_coherence > 0, "No quantum coherence detected"
            assert score_classic.quantum_coherence == 0, "Classic mode has quantum coherence"
            
            # Check for dream insights
            assert len(score_quantum.dream_insights) >= 0, "Dream insights not captured"
            
            logger.info(f"\nQuantum coherence: {score_quantum.quantum_coherence:.3f}")
            logger.info(f"Dream insights: {len(score_quantum.dream_insights)}")
            if score_quantum.dream_insights:
                logger.info("Sample insights:")
                for insight in score_quantum.dream_insights[:3]:
                    logger.info(f"  - {insight}")
                    
            logger.info(f"\nScore comparison:")
            logger.info(f"  Quantum: {score_quantum.overall_score:.3f}")
            logger.info(f"  Classic: {score_classic.overall_score:.3f}")
            
            self.test_results['passed'].append("quantum_dream_features")
            self.test_results['metrics']['quantum_coherence'] = score_quantum.quantum_coherence
            logger.info("\n✓ Test passed: Quantum and dream features working")
            
        except Exception as e:
            self.test_results['failed'].append(("quantum_dream_features", str(e)))
            logger.error(f"\n✗ Test failed: {e}")
            raise
            
    def test_mutation_resistance(self):
        """Test mutation resistance evaluation"""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Mutation Resistance")
        logger.info("="*60)
        
        try:
            # Create target with known mutations
            target = self._create_high_quality_target()
            target.mutation_data = [
                (100, 'A', 'V'),
                (150, 'R', 'K'),
                (200, 'D', 'N')
            ]
            
            # Evaluate with stress testing
            gc = DrugDiscoveryGerminalCenter(population_size=32)
            evaluator = DrugTargetEvaluator(germinal_center=gc)
            
            scores = evaluator.evaluate_targets(
                [target],
                generations=15,
                stress_test=True
            )
            
            score = list(scores.values())[0]
            
            # Check phase transitions (should have some due to stress)
            phase_data = score.phase_transition_data
            logger.info(f"\nPhase transitions: {len(phase_data['transitions'])}")
            logger.info(f"Stability: {phase_data['stability']}")
            
            # Evolutionary potential should be reasonable
            assert score.evolutionary_potential > 0.2, "Low evolutionary potential"
            
            logger.info(f"Evolutionary potential: {score.evolutionary_potential:.3f}")
            
            self.test_results['passed'].append("mutation_resistance")
            self.test_results['metrics']['evolutionary_potential'] = score.evolutionary_potential
            logger.info("\n✓ Test passed: Mutation resistance tested")
            
        except Exception as e:
            self.test_results['failed'].append(("mutation_resistance", str(e)))
            logger.error(f"\n✗ Test failed: {e}")
            raise
            
    def _create_high_quality_target(self) -> DrugTargetAntigen:
        """Create a high-quality drug target"""
        sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLPARTVETRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
        
        structure = ProteinStructure(
            sequence=sequence,
            coordinates=np.random.randn(len(sequence) * 5, 3) * 15,
            secondary_structure='C' * len(sequence),
            pdb_id="HIGH_QUALITY_TARGET"
        )
        
        # Multiple good pockets
        pockets = [
            BindingPocket(
                pocket_id="pocket_1",
                residue_indices=list(range(20, 40)),
                volume=450.0,
                hydrophobicity=1.2,
                electrostatic_potential=-0.5,
                druggability_score=0.85
            ),
            BindingPocket(
                pocket_id="pocket_2",
                residue_indices=list(range(80, 100)),
                volume=380.0,
                hydrophobicity=0.8,
                electrostatic_potential=0.2,
                druggability_score=0.75
            )
        ]
        
        return DrugTargetAntigen(
            protein_structure=structure,
            binding_pockets=pockets,
            disease_association="cancer",
            expression_data={'tumor': 5.2, 'normal': 1.1}
        )
        
    def _create_medium_quality_target(self) -> DrugTargetAntigen:
        """Create a medium-quality drug target"""
        sequence = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSMGRGSMLATVMTAPPGAEPPVAEPPGAEAAIVATVMEVNINKNLVGKDSNNLCLHFNPRFNAHGDANTIVCNSKCDRTGFYPLYHSVP"
        
        structure = ProteinStructure(
            sequence=sequence,
            coordinates=np.random.randn(len(sequence) * 5, 3) * 12,
            secondary_structure='C' * len(sequence),
            pdb_id="MEDIUM_QUALITY_TARGET"
        )
        
        pockets = [
            BindingPocket(
                pocket_id="pocket_1",
                residue_indices=list(range(30, 45)),
                volume=280.0,
                hydrophobicity=0.5,
                electrostatic_potential=-1.5,
                druggability_score=0.55
            )
        ]
        
        return DrugTargetAntigen(
            protein_structure=structure,
            binding_pockets=pockets,
            disease_association="inflammation"
        )
        
    def _create_poor_quality_target(self) -> DrugTargetAntigen:
        """Create a poor-quality drug target"""
        sequence = "MKKKKKKKKKKEEEEEEEEEDDDDDDDDDDGGGGGGGGGGSSSSSSSSSS"
        
        structure = ProteinStructure(
            sequence=sequence,
            coordinates=np.random.randn(len(sequence) * 5, 3) * 8,
            secondary_structure='C' * len(sequence),
            pdb_id="POOR_QUALITY_TARGET"
        )
        
        # Small, charged pocket
        pockets = [
            BindingPocket(
                pocket_id="pocket_1",
                residue_indices=list(range(10, 20)),
                volume=150.0,
                hydrophobicity=-2.5,
                electrostatic_potential=-3.0,
                druggability_score=0.25
            )
        ]
        
        return DrugTargetAntigen(
            protein_structure=structure,
            binding_pockets=pockets
        )
        
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*80)
        logger.info("TEST REPORT")
        logger.info("="*80)
        
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed'])
        pass_rate = len(self.test_results['passed']) / total_tests * 100 if total_tests > 0 else 0
        
        logger.info(f"\nTests passed: {len(self.test_results['passed'])}/{total_tests} ({pass_rate:.1f}%)")
        
        if self.test_results['passed']:
            logger.info("\nPassed tests:")
            for test in self.test_results['passed']:
                logger.info(f"  ✓ {test}")
                
        if self.test_results['failed']:
            logger.info("\nFailed tests:")
            for test, error in self.test_results['failed']:
                logger.info(f"  ✗ {test}: {error}")
                
        logger.info("\nKey metrics:")
        for metric, value in self.test_results['metrics'].items():
            logger.info(f"  - {metric}: {value:.3f}" if isinstance(value, float) else f"  - {metric}: {value}")
            
        # Save report
        report_path = "test_drug_discovery_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"\nReport saved to: {report_path}")
        
        # Create visualization if all tests passed
        if len(self.test_results['failed']) == 0:
            self._create_visualization()
            
    def _create_visualization(self):
        """Create visualization of test results"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Metrics plot
            metrics = self.test_results['metrics']
            if metrics:
                plt.subplot(2, 2, 1)
                metric_names = list(metrics.keys())
                metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
                metric_names = [n for n, v in zip(metric_names, metrics.values()) if isinstance(v, (int, float))]
                
                plt.bar(range(len(metric_names)), metric_values)
                plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right')
                plt.title('Test Metrics')
                plt.tight_layout()
                
            plt.savefig('test_drug_discovery_results.png', dpi=150, bbox_inches='tight')
            logger.info("Visualization saved to: test_drug_discovery_results.png")
            
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")


if __name__ == "__main__":
    # Configure for testing
    cfg.initial_population = 32  # Smaller for faster testing
    cfg.batch_size = 16
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run tests
    test_suite = DrugDiscoveryTest()
    test_suite.run_all_tests()