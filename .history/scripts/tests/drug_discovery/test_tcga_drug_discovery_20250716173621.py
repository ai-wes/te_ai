"""
TCGA Drug Discovery Test
========================

Tests the complete drug discovery pipeline using real TCGA data.
This demonstrates the power of TE-AI for identifying and ranking
drug targets from large-scale cancer genomics data.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.domains.drug_discovery.tcga_data_converter import TCGADataConverter, run_tcga_drug_discovery_pipeline
from scripts.domains.drug_discovery import DrugDiscoveryGerminalCenter, DrugTargetEvaluator
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()


def test_tcga_drug_discovery():
    """Test drug discovery using real TCGA data"""
    
    # Configuration
    TCGA_DATA_DIR = "/mnt/c/Users/wes/desktop/te_ai/maia_tcga_pancan"
    MAX_SAMPLES = 500  # Start with subset for testing
    TOP_TARGETS = 10
    GENERATIONS = 15
    
    logger.info("="*80)
    logger.info("REAL TCGA DRUG DISCOVERY TEST")
    logger.info("="*80)
    logger.info(f"TCGA Data Directory: {TCGA_DATA_DIR}")
    logger.info(f"Max Samples: {MAX_SAMPLES}")
    logger.info(f"Top Targets: {TOP_TARGETS}")
    
    # Check if data directory exists
    if not os.path.exists(TCGA_DATA_DIR):
        logger.error(f"TCGA data directory not found: {TCGA_DATA_DIR}")
        return False
        
    try:
        # Step 1: Run TCGA pipeline to get targets
        logger.info("\n🧬 Step 1: Loading and analyzing TCGA data...")
        start_time = time.time()
        
        pipeline_results = run_tcga_drug_discovery_pipeline(
            tcga_data_dir=TCGA_DATA_DIR,
            max_samples=MAX_SAMPLES,
            top_targets=TOP_TARGETS,
            cancer_focus=None  # Include all cancer types
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ TCGA pipeline completed in {load_time:.1f}s")
        
        # Extract results
        samples_data = pipeline_results['samples_data']
        target_ids = pipeline_results['target_ids']
        antigens = pipeline_results['antigens']
        tcga_report = pipeline_results['report']
        
        # Log TCGA analysis results
        logger.info(f"\n📊 TCGA Dataset Summary:")
        logger.info(f"  - Total samples: {len(samples_data['sample_ids'])}")
        logger.info(f"  - Cancer types: {tcga_report['dataset_summary']['cancer_types']}")
        logger.info(f"  - Transcriptomic features: {tcga_report['dataset_summary']['transcriptomics_genes']:,}")
        logger.info(f"  - Mutation features: {tcga_report['dataset_summary']['mutation_features']:,}")
        
        # Show cancer type distribution
        logger.info(f"\n🎯 Cancer Type Distribution:")
        for cancer_type, count in tcga_report['dataset_summary']['cancer_type_distribution'].items():
            logger.info(f"  - {cancer_type}: {count} samples")
            
        # Step 2: Initialize TE-AI for drug discovery
        logger.info(f"\n🧠 Step 2: Initializing TE-AI Drug Discovery System...")
        
        # Configure for real data processing
        cfg.initial_population = 64  # Reasonable size for real evaluation
        cfg.batch_size = 32
        
        germinal_center = DrugDiscoveryGerminalCenter(
            population_size=64,
            enable_drug_genes=True,
            enable_quantum_dreams=True
        )
        
        evaluator = DrugTargetEvaluator(
            germinal_center=germinal_center,
            quantum_evaluation=True,
            dream_analysis=True
        )
        
        logger.info(f"✅ TE-AI system initialized with {len(germinal_center.population)} cells")
        
        # Step 3: Evaluate drug targets
        logger.info(f"\n🔬 Step 3: Evaluating {len(antigens)} drug targets with TE-AI...")
        
        eval_start = time.time()
        
        # Evaluate subset for demo (full evaluation would take longer)
        test_antigens = antigens[:5]  # Test top 5 targets
        
        scores = evaluator.evaluate_targets(
            test_antigens,
            generations=GENERATIONS,
            parallel_evaluation=True,
            stress_test=True
        )
        
        eval_time = time.time() - eval_start
        logger.info(f"✅ Drug target evaluation completed in {eval_time:.1f}s")
        
        # Step 4: Analyze results
        logger.info(f"\n📈 Step 4: Analyzing drug target rankings...")
        
        # Sort targets by score
        ranked_targets = sorted(
            scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        logger.info(f"\n🏆 Top Drug Targets (TE-AI Ranked):")
        logger.info("="*60)
        
        for rank, (target_id, score) in enumerate(ranked_targets, 1):
            recommendation = "🔥 PRIORITY" if score.overall_score > 0.7 else \
                           "⭐ PROMISING" if score.overall_score > 0.5 else \
                           "📋 EVALUATE"
                           
            logger.info(f"{rank}. {target_id}")
            logger.info(f"   Overall Score: {score.overall_score:.3f} | {recommendation}")
            logger.info(f"   Quantum Coherence: {score.quantum_coherence:.3f}")
            logger.info(f"   Evolutionary Potential: {score.evolutionary_potential:.3f}")
            
            # Show top score components
            top_components = sorted(
                score.components.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            logger.info(f"   Key Factors: {', '.join([f'{c}={v:.2f}' for c, v in top_components])}")
            
            # Show dream insights if any
            if score.dream_insights:
                logger.info(f"   Dream Insights: {score.dream_insights[0]}")
                
            logger.info("")
            
        # Step 5: Generate comprehensive report
        logger.info(f"🔍 Step 5: Generating comprehensive drug discovery report...")
        
        # Create detailed report
        discovery_report = {
            "tcga_analysis": tcga_report,
            "te_ai_evaluation": {
                "evaluation_time": eval_time,
                "generations_per_target": GENERATIONS,
                "population_size": len(germinal_center.population),
                "quantum_enabled": True,
                "dream_analysis": True
            },
            "target_rankings": [
                {
                    "rank": rank,
                    "target_id": target_id,
                    "overall_score": score.overall_score,
                    "quantum_coherence": score.quantum_coherence,
                    "evolutionary_potential": score.evolutionary_potential,
                    "key_components": dict(sorted(
                        score.components.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]),
                    "binding_profiles": len(score.binding_profiles),
                    "phase_transitions": len(score.phase_transition_data['transitions']),
                    "dream_insights": score.dream_insights[:3]  # Top 3 insights
                }
                for rank, (target_id, score) in enumerate(ranked_targets, 1)
            ],
            "system_performance": {
                "total_processing_time": load_time + eval_time,
                "tcga_loading_time": load_time,
                "te_ai_evaluation_time": eval_time,
                "samples_processed": len(samples_data['sample_ids']),
                "targets_identified": len(target_ids),
                "targets_evaluated": len(test_antigens)
            }
        }
        
        # Save report
        report_file = "tcga_te_ai_drug_discovery_report.json"
        with open(report_file, 'w') as f:
            json.dump(discovery_report, f, indent=2)
            
        logger.info(f"✅ Comprehensive report saved to: {report_file}")
        
        # Summary
        logger.info(f"\n🎉 DRUG DISCOVERY COMPLETED SUCCESSFULLY!")
        logger.info(f"="*60)
        logger.info(f"📊 Processed {len(samples_data['sample_ids'])} TCGA samples")
        logger.info(f"🎯 Identified {len(target_ids)} potential drug targets")
        logger.info(f"🧠 Evaluated {len(test_antigens)} targets with TE-AI")
        logger.info(f"⏱️  Total time: {(load_time + eval_time):.1f}s")
        logger.info(f"🏆 Best target: {ranked_targets[0][0]} (score: {ranked_targets[0][1].overall_score:.3f})")
        
        # Validate results
        assert len(ranked_targets) > 0, "No targets were evaluated"
        assert all(0 <= score.overall_score <= 1 for _, score in ranked_targets), "Invalid scores"
        assert ranked_targets[0][1].overall_score >= ranked_targets[-1][1].overall_score, "Ranking error"
        
        logger.info(f"\n✅ All validation checks passed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_tcga_validation():
    """Quick validation of TCGA data loading"""
    
    TCGA_DATA_DIR = "/mnt/c/Users/wes/desktop/te_ai/maia_tcga_pancan"
    
    logger.info("🔍 Quick TCGA Data Validation...")
    
    if not os.path.exists(TCGA_DATA_DIR):
        logger.error(f"TCGA directory not found: {TCGA_DATA_DIR}")
        return False
        
    # Count files
    npz_files = list(Path(TCGA_DATA_DIR).glob("sample_*.npz"))
    logger.info(f"Found {len(npz_files)} NPZ files")
    
    if len(npz_files) == 0:
        logger.error("No NPZ files found")
        return False
        
    # Test loading one file
    test_file = npz_files[0]
    try:
        data = np.load(test_file)
        logger.info(f"✅ Successfully loaded: {test_file.name}")
        logger.info(f"   Keys: {list(data.keys())}")
        
        if 'transcriptomics' in data:
            logger.info(f"   Transcriptomics shape: {data['transcriptomics'].shape}")
        if 'genomics_mutations' in data:
            logger.info(f"   Mutations shape: {data['genomics_mutations'].shape}")
        if 'cancer_type' in data:
            logger.info(f"   Cancer type: {data['cancer_type']}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load test file: {e}")
        return False


if __name__ == "__main__":
    # First run quick validation
    logger.info("Starting TCGA Drug Discovery Test...")
    
    if not quick_tcga_validation():
        logger.error("TCGA validation failed, exiting")
        sys.exit(1)
        
    # Run full test
    success = test_tcga_drug_discovery()
    
    if success:
        logger.info("🎉 TCGA Drug Discovery Test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ TCGA Drug Discovery Test FAILED!")
        sys.exit(1)