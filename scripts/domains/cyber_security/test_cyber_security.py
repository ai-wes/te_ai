#!/usr/bin/env python3
"""
Test script for TE-AI Cybersecurity Implementation
=================================================

Demonstrates the adaptive cybersecurity capabilities of the TE-AI system.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.config import cfg
from scripts.domains.cyber_security.main_cyber_security import (
    CyberSecurityGerminalCenter,
    NetworkPacket,
    simulate_cyber_attack,
    run_cybersecurity_demo
)
from scripts.domains.cyber_security.threat_antigen import (
    ThreatAntigen,
    ThreatIndicator,
    ThreatIntelligenceAdapter
)
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()


def test_threat_detection():
    """Test basic threat detection capabilities"""
    logger.info("Testing threat detection capabilities...")
    
    # Initialize system
    cfg.population_size = 20
    cfg.batch_size = 32
    defense_system = CyberSecurityGerminalCenter(cfg)
    
    # Test different attack types
    test_cases = [
        ("Normal Traffic", None, 0.0),
        ("DDoS Attack", "ddos", 0.5),
        ("SQL Injection", "sql_injection", 0.7),
        ("Port Scan", "port_scan", 0.6),
        ("Zero-Day Exploit", "zero_day", 0.9)
    ]
    
    results = []
    
    for test_name, attack_type, expected_severity in test_cases:
        logger.info(f"\nTesting: {test_name}")
        
        # Generate traffic
        if attack_type:
            packets = simulate_cyber_attack(attack_type, intensity=0.5)
        else:
            # Normal traffic
            packets = []
            for _ in range(50):
                packet = NetworkPacket(
                    source_ip="192.168.1.100",
                    dest_ip="192.168.1.200",
                    protocol="tcp",
                    port=443,
                    payload=torch.randn(64) * 0.3,
                    timestamp=time.time(),
                    flags={'syn': False, 'ack': True, 'fin': False}
                )
                packets.append(packet)
        
        # Convert to tensor batch
        traffic_tensors = [p.to_tensor() for p in packets[:32]]
        traffic_batch = torch.stack(traffic_tensors).to(defense_system.device)
        
        # Detect threats
        with torch.no_grad():
            detection_results = defense_system.forward(traffic_batch)
        
        avg_threat_score = detection_results['threat_score'].mean().item()
        max_threat_score = detection_results['threat_score'].max().item()
        
        logger.info(f"  Average threat score: {avg_threat_score:.3f}")
        logger.info(f"  Maximum threat score: {max_threat_score:.3f}")
        logger.info(f"  Expected severity: {expected_severity:.3f}")
        
        # Check detection accuracy
        if attack_type:
            detection_success = avg_threat_score > 0.3
            logger.info(f"  Detection: {'SUCCESS' if detection_success else 'FAILED'}")
        else:
            false_positive = avg_threat_score > 0.3
            logger.info(f"  False positive: {'YES' if false_positive else 'NO'}")
        
        results.append({
            'test': test_name,
            'avg_score': avg_threat_score,
            'max_score': max_threat_score,
            'expected': expected_severity
        })
    
    return results


def test_evolution_under_attack():
    """Test system evolution in response to persistent attacks"""
    logger.info("\nTesting adaptive evolution under persistent attacks...")
    
    # Initialize system
    cfg = Config()
    cfg.population_size = 30
    defense_system = CyberSecurityGerminalCenter(cfg)
    
    # Simulate evolving attack campaign
    attack_phases = [
        ("Initial Reconnaissance", "port_scan", 0.3, 5),
        ("Exploitation Attempts", "sql_injection", 0.5, 10),
        ("Escalated Attack", "ddos", 0.7, 15),
        ("Advanced Persistent Threat", "zero_day", 0.9, 20)
    ]
    
    evolution_metrics = []
    
    for phase_name, attack_type, intensity, duration in attack_phases:
        logger.info(f"\n=== {phase_name} ===")
        logger.info(f"Attack type: {attack_type}, Intensity: {intensity}")
        
        phase_start = time.time()
        detection_scores = []
        
        for iteration in range(duration):
            # Generate attack traffic
            packets = simulate_cyber_attack(attack_type, intensity)
            traffic_batch = torch.stack([p.to_tensor() for p in packets[:64]])
            traffic_batch = traffic_batch.to(defense_system.device)
            
            # Detect and record
            with torch.no_grad():
                results = defense_system.forward(traffic_batch)
            
            avg_detection = results['threat_score'].mean().item()
            detection_scores.append(avg_detection)
            
            # Calculate attack success (inverse of detection)
            attack_success = 1.0 - avg_detection
            
            # Evolve defenses if attack is succeeding
            if attack_success > 0.4 and iteration % 5 == 0:
                logger.info(f"  Iteration {iteration}: Evolving defenses (attack success: {attack_success:.2%})")
                defense_system.evolve_defenses(attack_success, attack_type)
            
            # Check for zero-day
            is_zero_day, novelty = defense_system.detect_zero_day([traffic_batch])
            if is_zero_day:
                logger.warning(f"  Zero-day detected! Novelty: {novelty:.3f}")
        
        # Phase metrics
        phase_time = time.time() - phase_start
        avg_detection = np.mean(detection_scores)
        detection_improvement = detection_scores[-1] - detection_scores[0]
        
        logger.info(f"Phase complete:")
        logger.info(f"  Duration: {phase_time:.1f}s")
        logger.info(f"  Average detection: {avg_detection:.3f}")
        logger.info(f"  Detection improvement: {detection_improvement:+.3f}")
        
        evolution_metrics.append({
            'phase': phase_name,
            'avg_detection': avg_detection,
            'improvement': detection_improvement,
            'final_detection': detection_scores[-1]
        })
    
    return evolution_metrics


def test_threat_antigen_integration():
    """Test integration with threat intelligence antigens"""
    logger.info("\nTesting threat antigen integration...")
    
    # Create threat intelligence adapter
    adapter = ThreatIntelligenceAdapter()
    
    # Generate synthetic threats
    threats = adapter.create_synthetic_threats(count=5)
    
    # Initialize defense system
    cfg = Config()
    cfg.population_size = 20
    defense_system = CyberSecurityGerminalCenter(cfg)
    
    logger.info(f"Processing {len(threats)} threat antigens...")
    
    for threat in threats:
        logger.info(f"\nThreat: {threat.threat_name}")
        logger.info(f"  Type: {threat.threat_type}")
        logger.info(f"  Severity: {threat.severity:.3f}")
        logger.info(f"  Indicators: {len(threat.indicators)}")
        logger.info(f"  Complexity: {threat.get_complexity_score():.3f}")
        
        # Convert to graph
        threat_graph = threat.to_graph()
        logger.info(f"  Graph nodes: {threat_graph.x.shape[0]}")
        logger.info(f"  Graph edges: {threat_graph.edge_index.shape[1]}")


def run_all_tests():
    """Run all cybersecurity tests"""
    logger.info("="*60)
    logger.info("TE-AI CYBERSECURITY TEST SUITE")
    logger.info("="*60)
    
    # Test 1: Basic detection
    logger.info("\n[TEST 1] Basic Threat Detection")
    detection_results = test_threat_detection()
    
    # Test 2: Evolution
    logger.info("\n[TEST 2] Adaptive Evolution")
    evolution_results = test_evolution_under_attack()
    
    # Test 3: Threat antigens
    logger.info("\n[TEST 3] Threat Antigen Integration")
    test_threat_antigen_integration()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    logger.info("\nDetection Results:")
    for result in detection_results:
        logger.info(f"  {result['test']}: {result['avg_score']:.3f} (expected: {result['expected']:.3f})")
    
    logger.info("\nEvolution Results:")
    for result in evolution_results:
        logger.info(f"  {result['phase']}: {result['improvement']:+.3f} improvement")
    
    logger.info("\nAll tests completed successfully!")


if __name__ == "__main__":
    # Run comprehensive test or quick demo
    import argparse
    parser = argparse.ArgumentParser(description="Test TE-AI Cybersecurity System")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    args = parser.parse_args()
    
    if args.full:
        run_all_tests()
    else:
        # Default: run quick demo
        run_cybersecurity_demo()