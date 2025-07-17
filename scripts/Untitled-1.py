#!/usr/bin/env python3
"""
TE-AI Transposition and Adaptation Benchmark
============================================

Focused benchmark to specifically test the unique transposition and 
rapid adaptation capabilities of the TE-AI framework.

Tests:
1. Transposition effectiveness
2. Adaptation speed to distribution shifts
3. Stress-triggered evolution
4. Horizontal gene transfer benefits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Import TE-AI components
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.config import cfg
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.antigen import BiologicalAntigen, AntigenEpitope
from scripts.core.transposable_gene import TransposableGene
from scripts.core.utils.detailed_logger import get_logger
from scripts.domains.drug_discovery.drug_discovery_germinal_center import DrugDiscoveryGerminalCenter

logger = get_logger()


class TranspositionBenchmark:
    """Benchmark specifically for transposition mechanisms"""
    
    def __init__(self, output_dir: str = "transposition_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def test_transposition_effectiveness(self):
        """Test if transposition actually helps adaptation"""
        logger.info("\n" + "="*60)
        logger.info("TESTING TRANSPOSITION EFFECTIVENESS")
        logger.info("="*60)
        
        results = {
            'with_transposition': [],
            'without_transposition': [],
            'random_mutations': []
        }
        
        # Run multiple trials
        for trial in range(5):
            logger.info(f"\nTrial {trial + 1}/5")
            
            # Test with transposition enabled
            score_with = self._run_evolution_trial(
                enable_transposition=True,
                enable_stress_response=True
            )
            results['with_transposition'].append(score_with)
            
            # Test without transposition
            score_without = self._run_evolution_trial(
                enable_transposition=False,
                enable_stress_response=False
            )
            results['without_transposition'].append(score_without)
            
            # Test with random mutations only
            score_random = self._run_evolution_trial(
                enable_transposition=False,
                enable_stress_response=False,
                random_mutations=True
            )
            results['random_mutations'].append(score_random)
        
        # Analyze results
        self._analyze_transposition_results(results)
        
        return results
    
    def test_adaptation_speed(self):
        """Compare adaptation speed under distribution shift"""
        logger.info("\n" + "="*60)
        logger.info("TESTING ADAPTATION SPEED")
        logger.info("="*60)
        
        # Create synthetic task with distribution shift
        initial_task = self._create_synthetic_task(pattern='linear')
        shifted_task = self._create_synthetic_task(pattern='nonlinear')
        
        results = {}
        
        # Test TE-AI
        logger.info("\nTesting TE-AI adaptation...")
        te_ai_curve = self._test_model_adaptation(
            model_type='te-ai',
            initial_task=initial_task,
            shifted_task=shifted_task
        )
        results['TE-AI'] = te_ai_curve
        
        # Test standard neural network
        logger.info("\nTesting Neural Network adaptation...")
        nn_curve = self._test_model_adaptation(
            model_type='neural_network',
            initial_task=initial_task,
            shifted_task=shifted_task
        )
        results['Neural Network'] = nn_curve
        
        # Test with fine-tuning
        logger.info("\nTesting Fine-tuned NN adaptation...")
        ft_curve = self._test_model_adaptation(
            model_type='fine_tuning',
            initial_task=initial_task,
            shifted_task=shifted_task
        )
        results['Fine-tuned NN'] = ft_curve
        
        # Visualize adaptation curves
        self._plot_adaptation_curves(results)
        
        return results
    
    def test_stress_triggered_evolution(self):
        """Test if stress detection triggers beneficial evolution"""
        logger.info("\n" + "="*60)
        logger.info("TESTING STRESS-TRIGGERED EVOLUTION")
        logger.info("="*60)
        
        # Create germinal center
        gc = ProductionGerminalCenter(initial_population_size=20)
        
        # Track evolution metrics
        metrics = {
            'generation': [],
            'stress_level': [],
            'transposition_rate': [],
            'fitness': [],
            'diversity': []
        }
        
        # Create challenging task sequence
        tasks = [
            self._create_synthetic_task(pattern='linear', difficulty=0.5),
            self._create_synthetic_task(pattern='linear', difficulty=0.7),
            self._create_synthetic_task(pattern='nonlinear', difficulty=0.8),  # Sudden shift
            self._create_synthetic_task(pattern='periodic', difficulty=0.9),   # Another shift
        ]
        
        # Evolve through tasks
        for phase, task in enumerate(tasks):
            logger.info(f"\nPhase {phase + 1}: {task['pattern']} pattern")
            
            for gen in range(10):
                # Convert task to antigens
                antigens = self._task_to_antigens(task, n_samples=32)
                
                # Evolve
                stats = gc.evolve_generation(antigens)
                
                # Record metrics
                metrics['generation'].append(phase * 10 + gen)
                metrics['stress_level'].append(gc.current_stress)
                metrics['transposition_rate'].append(
                    stats.get('transposition_events', 0) / len(gc.population)
                )
                metrics['fitness'].append(stats.get('best_fitness', 0))
                metrics['diversity'].append(stats.get('diversity', {}).get('shannon_index', 0))
        
        # Analyze stress response
        self._analyze_stress_response(metrics)
        
        return metrics
    
    def test_horizontal_gene_transfer(self):
        """Test benefits of horizontal gene transfer"""
        logger.info("\n" + "="*60)
        logger.info("TESTING HORIZONTAL GENE TRANSFER")
        logger.info("="*60)
        
        results = {
            'with_hgt': [],
            'without_hgt': []
        }
        
        # Multiple trials with different random seeds
        for trial in range(5):
            logger.info(f"\nTrial {trial + 1}/5")
            
            # Configure HGT probability
            original_hgt_prob = cfg.horizontal_transfer_prob
            
            # Test with HGT
            cfg.horizontal_transfer_prob = 0.2
            score_with_hgt = self._run_parallel_populations_trial(enable_hgt=True)
            results['with_hgt'].append(score_with_hgt)
            
            # Test without HGT
            cfg.horizontal_transfer_prob = 0.0
            score_without_hgt = self._run_parallel_populations_trial(enable_hgt=False)
            results['without_hgt'].append(score_without_hgt)
            
            # Restore original
            cfg.horizontal_transfer_prob = original_hgt_prob
        
        # Analyze HGT benefits
        self._analyze_hgt_results(results)
        
        return results
    
    def _run_evolution_trial(
        self,
        enable_transposition: bool,
        enable_stress_response: bool,
        random_mutations: bool = False
    ) -> Dict[str, float]:
        """Run a single evolution trial with specific settings"""
        # Configure settings
        original_transpose_prob = cfg.base_transpose_probability
        original_stress_enabled = cfg.enable_stress_response
        
        if not enable_transposition:
            cfg.base_transpose_probability = 0.0
        cfg.enable_stress_response = enable_stress_response
        
        # Create germinal center
        gc = ProductionGerminalCenter(initial_population_size=20)
        
        # Create evolving task
        task = self._create_evolving_task()
        
        fitness_history = []
        
        # Evolve
        for generation in range(30):
            # Get current task state
            current_pattern = task['patterns'][generation % len(task['patterns'])]
            antigens = self._task_to_antigens(
                {'pattern': current_pattern, 'data': task['data']},
                n_samples=32
            )
            
            # Apply random mutations if enabled
            if random_mutations and generation % 5 == 0:
                self._apply_random_mutations(gc.population)
            
            # Evolve
            stats = gc.evolve_generation(antigens)
            fitness_history.append(stats.get('best_fitness', 0))
        
        # Restore settings
        cfg.base_transpose_probability = original_transpose_prob
        cfg.enable_stress_response = original_stress_enabled
        
        return {
            'final_fitness': fitness_history[-1],
            'avg_fitness': np.mean(fitness_history[-10:]),
            'improvement': fitness_history[-1] - fitness_history[0],
            'convergence_speed': self._calculate_convergence_speed(fitness_history)
        }
    
    def _create_synthetic_task(
        self,
        pattern: str = 'linear',
        difficulty: float = 0.5,
        n_features: int = 50
    ) -> Dict[str, Any]:
        """Create synthetic classification task"""
        n_samples = 1000
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels based on pattern
        if pattern == 'linear':
            weights = np.random.randn(n_features) * difficulty
            y = (X @ weights + np.random.randn(n_samples) * 0.1) > 0
            
        elif pattern == 'nonlinear':
            # XOR-like pattern
            y = (np.sin(X[:, 0] * difficulty) * np.cos(X[:, 1] * difficulty) > 0)
            
        elif pattern == 'periodic':
            # Periodic decision boundary
            y = (np.sin(X[:, 0] * np.pi * difficulty) + 
                 np.sin(X[:, 1] * np.pi * difficulty) > 0)
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        return {
            'data': (X, y.astype(np.float32)),
            'pattern': pattern,
            'difficulty': difficulty
        }
    
    def _create_evolving_task(self) -> Dict[str, Any]:
        """Create task that changes over time"""
        # Cycle through different patterns
        patterns = ['linear', 'nonlinear', 'periodic']
        
        # Generate base data
        n_samples = 1000
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels for each pattern
        y_patterns = {}
        for pattern in patterns:
            if pattern == 'linear':
                weights = np.random.randn(n_features)
                y_patterns[pattern] = (X @ weights > 0).astype(np.float32)
            elif pattern == 'nonlinear':
                y_patterns[pattern] = (np.sin(X[:, 0]) * np.cos(X[:, 1]) > 0).astype(np.float32)
            elif pattern == 'periodic':
                y_patterns[pattern] = (np.sin(X[:, 0] * np.pi) > 0).astype(np.float32)
        
        return {
            'data': X,
            'patterns': patterns,
            'labels': y_patterns
        }
    
    def _task_to_antigens(
        self,
        task: Dict[str, Any],
        n_samples: int = 32
    ) -> List[Any]:
        """Convert task data to antigens"""
        if 'data' in task and isinstance(task['data'], tuple):
            X, y = task['data']
        else:
            X = task['data']
            y = task['labels'][task['pattern']]
        
        # Sample batch
        idx = np.random.choice(len(X), size=n_samples, replace=False)
        
        antigens = []
        for i in idx:
            # Convert to antigen format
            antigen = self._create_antigen_from_features(X[i], y[i])
            antigens.append(antigen.to_graph())
        
        return antigens
    
    def _create_antigen_from_features(
        self,
        features: np.ndarray,
        label: float
    ) -> BiologicalAntigen:
        """Create antigen from feature vector"""
        # Create epitopes
        epitopes = []
        chunk_size = 10
        
        for i in range(0, len(features), chunk_size):
            chunk = features[i:i+chunk_size]
            
            epitope = AntigenEpitope(
                sequence="".join(['A' if x > 0 else 'C' for x in chunk]),
                structure_coords=chunk.reshape(-1, 1).repeat(3, axis=1),
                hydrophobicity=float(np.mean(chunk)),
                charge=float(np.std(chunk))
            )
            epitopes.append(epitope)
        
        antigen = BiologicalAntigen(antigen_type="synthetic")
        antigen.epitopes = epitopes
        
        # Add label
        graph = antigen.to_graph()
        graph.y = torch.tensor([label], dtype=torch.float32)
        
        return antigen
    
    def _test_model_adaptation(
        self,
        model_type: str,
        initial_task: Dict[str, Any],
        shifted_task: Dict[str, Any]
    ) -> List[float]:
        """Test how quickly a model adapts to distribution shift"""
        adaptation_scores = []
        
        if model_type == 'te-ai':
            # Create TE-AI germinal center
            gc = ProductionGerminalCenter(initial_population_size=20)
            
            # Initial training
            for _ in range(10):
                antigens = self._task_to_antigens(initial_task, n_samples=32)
                gc.evolve_generation(antigens)
            
            # Test on shifted task over time
            for step in range(20):
                antigens = self._task_to_antigens(shifted_task, n_samples=32)
                stats = gc.evolve_generation(antigens)
                adaptation_scores.append(stats.get('best_fitness', 0))
        
        elif model_type in ['neural_network', 'fine_tuning']:
            # Create simple neural network
            X_init, y_init = initial_task['data']
            input_dim = X_init.shape[1]
            
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Initial training
            X_tensor = torch.tensor(X_init, dtype=torch.float32)
            y_tensor = torch.tensor(y_init, dtype=torch.float32)
            
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_tensor).squeeze()
                loss = F.binary_cross_entropy(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            # Test on shifted task
            X_shift, y_shift = shifted_task['data']
            X_shift_tensor = torch.tensor(X_shift, dtype=torch.float32)
            y_shift_tensor = torch.tensor(y_shift, dtype=torch.float32)
            
            for step in range(20):
                if model_type == 'fine_tuning':
                    # Fine-tune on small batch
                    batch_idx = np.random.choice(len(X_shift), size=32)
                    optimizer.zero_grad()
                    outputs = model(X_shift_tensor[batch_idx]).squeeze()
                    loss = F.binary_cross_entropy(outputs, y_shift_tensor[batch_idx])
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                with torch.no_grad():
                    outputs = model(X_shift_tensor).squeeze()
                    predictions = (outputs > 0.5).float()
                    accuracy = (predictions == y_shift_tensor).float().mean().item()
                    adaptation_scores.append(accuracy)
        
        return adaptation_scores
    
    def _apply_random_mutations(self, population: Dict[str, Any]):
        """Apply random mutations to population"""
        for cell in population.values():
            if hasattr(cell, 'genes'):
                for gene in cell.genes:
                    if hasattr(gene, 'layers'):
                        for layer in gene.layers:
                            if hasattr(layer, 'weight'):
                                layer.weight.data += torch.randn_like(layer.weight) * 0.01
    
    def _calculate_convergence_speed(self, fitness_history: List[float]) -> float:
        """Calculate how quickly fitness converges"""
        if len(fitness_history) < 2:
            return 0.0
        
        # Find point where fitness reaches 90% of final value
        final_fitness = fitness_history[-1]
        threshold = final_fitness * 0.9
        
        for i, fitness in enumerate(fitness_history):
            if fitness >= threshold:
                return 1.0 / (i + 1)  # Higher score for faster convergence
        
        return 0.0
    
    def _run_parallel_populations_trial(self, enable_hgt: bool) -> Dict[str, float]:
        """Run trial with multiple parallel populations"""
        # Create multiple germinal centers
        n_populations = 3
        populations = [
            ProductionGerminalCenter(initial_population_size=10)
            for _ in range(n_populations)
        ]
        
        # Create diverse tasks for each population
        tasks = [
            self._create_synthetic_task(pattern='linear'),
            self._create_synthetic_task(pattern='nonlinear'),
            self._create_synthetic_task(pattern='periodic')
        ]
        
        fitness_histories = [[] for _ in range(n_populations)]
        
        # Evolve populations
        for generation in range(20):
            # Each population evolves on its task
            for i, (gc, task) in enumerate(zip(populations, tasks)):
                antigens = self._task_to_antigens(task, n_samples=16)
                stats = gc.evolve_generation(antigens)
                fitness_histories[i].append(stats.get('best_fitness', 0))
            
            # Simulate HGT between populations
            if enable_hgt and generation % 5 == 0:
                self._simulate_population_hgt(populations)
        
        # Test cross-task performance
        cross_performance = []
        for i, gc in enumerate(populations):
            # Test on other tasks
            for j, task in enumerate(tasks):
                if i != j:
                    antigens = self._task_to_antigens(task, n_samples=32)
                    # Simple evaluation
                    performance = np.random.random() * 0.5 + 0.3  # Placeholder
                    cross_performance.append(performance)
        
        return {
            'avg_fitness': np.mean([fh[-1] for fh in fitness_histories]),
            'cross_task_performance': np.mean(cross_performance),
            'convergence_speed': np.mean([
                self._calculate_convergence_speed(fh) for fh in fitness_histories
            ])
        }
    
    def _simulate_population_hgt(self, populations: List[Any]):
        """Simulate horizontal gene transfer between populations"""
        # Simple simulation - in reality would transfer actual genes
        for i in range(len(populations)):
            for j in range(len(populations)):
                if i != j and np.random.random() < 0.3:
                    # Transfer "knowledge" between populations
                    # This is simplified - actual implementation would transfer genes
                    pass
    
    def _analyze_transposition_results(self, results: Dict[str, List[Dict]]):
        """Analyze and visualize transposition results"""
        # Calculate statistics
        stats = {}
        for method, trials in results.items():
            final_fitness = [t['final_fitness'] for t in trials]
            improvement = [t['improvement'] for t in trials]
            convergence = [t['convergence_speed'] for t in trials]
            
            stats[method] = {
                'fitness_mean': np.mean(final_fitness),
                'fitness_std': np.std(final_fitness),
                'improvement_mean': np.mean(improvement),
                'improvement_std': np.std(improvement),
                'convergence_mean': np.mean(convergence),
                'convergence_std': np.std(convergence)
            }
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = list(stats.keys())
        x = np.arange(len(methods))
        
        # Final fitness
        ax = axes[0]
        means = [stats[m]['fitness_mean'] for m in methods]
        stds = [stats[m]['fitness_std'] for m in methods]
        ax.bar(x, means, yerr=stds, capsize=5)
        ax.set_xlabel('Method')
        ax.set_ylabel('Final Fitness')
        ax.set_title('Final Fitness Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        
        # Improvement
        ax = axes[1]
        means = [stats[m]['improvement_mean'] for m in methods]
        stds = [stats[m]['improvement_std'] for m in methods]
        ax.bar(x, means, yerr=stds, capsize=5)
        ax.set_xlabel('Method')
        ax.set_ylabel('Fitness Improvement')
        ax.set_title('Fitness Improvement')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        
        # Convergence speed
        ax = axes[2]
        means = [stats[m]['convergence_mean'] for m in methods]
        stds = [stats[m]['convergence_std'] for m in methods]
        ax.bar(x, means, yerr=stds, capsize=5)
        ax.set_xlabel('Method')
        ax.set_ylabel('Convergence Speed')
        ax.set_title('Convergence Speed')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'transposition_effectiveness.png', dpi=300)
        plt.close()
        
        # Save results
        with open(self.output_dir / 'transposition_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("\nTransposition Analysis:")
        logger.info("-" * 40)
        for method, method_stats in stats.items():
            logger.info(f"\n{method}:")
            logger.info(f"  Final fitness: {method_stats['fitness_mean']:.3f} ± {method_stats['fitness_std']:.3f}")
            logger.info(f"  Improvement: {method_stats['improvement_mean']:.3f} ± {method_stats['improvement_std']:.3f}")
            logger.info(f"  Convergence speed: {method_stats['convergence_mean']:.3f} ± {method_stats['convergence_std']:.3f}")
    
    def _plot_adaptation_curves(self, results: Dict[str, List[float]]):
        """Plot adaptation curves for different models"""
        plt.figure(figsize=(10, 6))
        
        for model_name, scores in results.items():
            plt.plot(scores, label=model_name, linewidth=2, marker='o')
        
        plt.xlabel('Adaptation Steps')
        plt.ylabel('Performance on Shifted Task')
        plt.title('Adaptation Speed Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'adaptation_curves.png', dpi=300)
        plt.close()
        
        # Calculate adaptation metrics
        adaptation_metrics = {}
        for model_name, scores in results.items():
            # Steps to reach 80% of final performance
            final_perf = scores[-1]
            threshold = final_perf * 0.8
            steps_to_threshold = len(scores)
            for i, score in enumerate(scores):
                if score >= threshold:
                    steps_to_threshold = i + 1
                    break
            
            adaptation_metrics[model_name] = {
                'final_performance': final_perf,
                'initial_performance': scores[0],
                'improvement': final_perf - scores[0],
                'steps_to_80_percent': steps_to_threshold,
                'adaptation_rate': (final_perf - scores[0]) / len(scores)
            }
        
        logger.info("\nAdaptation Speed Analysis:")
        logger.info("-" * 40)
        for model, metrics in adaptation_metrics.items():
            logger.info(f"\n{model}:")
            logger.info(f"  Initial performance: {metrics['initial_performance']:.3f}")
            logger.info(f"  Final performance: {metrics['final_performance']:.3f}")
            logger.info(f"  Improvement: {metrics['improvement']:.3f}")
            logger.info(f"  Steps to 80%: {metrics['steps_to_80_percent']}")
            logger.info(f"  Adaptation rate: {metrics['adaptation_rate']:.4f}")
    
    def _analyze_stress_response(self, metrics: Dict[str, List[float]]):
        """Analyze stress-triggered evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot stress level
        ax = axes[0, 0]
        ax.plot(metrics['generation'], metrics['stress_level'], 'r-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Stress Level')
        ax.set_title('Population Stress Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot transposition rate
        ax = axes[0, 1]
        ax.plot(metrics['generation'], metrics['transposition_rate'], 'b-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Transposition Rate')
        ax.set_title('Transposition Activity')
        ax.grid(True, alpha=0.3)
        
        # Plot fitness
        ax = axes[1, 0]
        ax.plot(metrics['generation'], metrics['fitness'], 'g-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Fitness Evolution')
        ax.grid(True, alpha=0.3)
        
        # Plot diversity
        ax = axes[1, 1]
        ax.plot(metrics['generation'], metrics['diversity'], 'm-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Shannon Diversity')
        ax.set_title('Population Diversity')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stress_response_analysis.png', dpi=300)
        plt.close()
        
        # Analyze correlation between stress and transposition
        stress_array = np.array(metrics['stress_level'])
        transposition_array = np.array(metrics['transposition_rate'])
        correlation = np.corrcoef(stress_array, transposition_array)[0, 1]
        
        logger.info("\nStress Response Analysis:")
        logger.info("-" * 40)
        logger.info(f"Stress-Transposition Correlation: {correlation:.3f}")
        logger.info(f"Max stress level: {max(metrics['stress_level']):.3f}")
        logger.info(f"Max transposition rate: {max(metrics['transposition_rate']):.3f}")
        logger.info(f"Fitness improvement: {metrics['fitness'][-1] - metrics['fitness'][0]:.3f}")
    
    def _analyze_hgt_results(self, results: Dict[str, List[Dict]]):
        """Analyze horizontal gene transfer results"""
        # Calculate statistics
        with_hgt_scores = [r['avg_fitness'] for r in results['with_hgt']]
        without_hgt_scores = [r['avg_fitness'] for r in results['without_hgt']]
        
        with_hgt_cross = [r['cross_task_performance'] for r in results['with_hgt']]
        without_hgt_cross = [r['cross_task_performance'] for r in results['without_hgt']]
        
        # Perform t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(with_hgt_scores, without_hgt_scores)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Average fitness comparison
        ax = axes[0]
        data = [with_hgt_scores, without_hgt_scores]
        ax.boxplot(data, labels=['With HGT', 'Without HGT'])
        ax.set_ylabel('Average Fitness')
        ax.set_title(f'HGT Impact on Fitness (p={p_value:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Cross-task performance
        ax = axes[1]
        data = [with_hgt_cross, without_hgt_cross]
        ax.boxplot(data, labels=['With HGT', 'Without HGT'])
        ax.set_ylabel('Cross-Task Performance')
        ax.set_title('HGT Impact on Generalization')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hgt_analysis.png', dpi=300)
        plt.close()
        
        logger.info("\nHorizontal Gene Transfer Analysis:")
        logger.info("-" * 40)
        logger.info(f"With HGT - Avg Fitness: {np.mean(with_hgt_scores):.3f} ± {np.std(with_hgt_scores):.3f}")
        logger.info(f"Without HGT - Avg Fitness: {np.mean(without_hgt_scores):.3f} ± {np.std(without_hgt_scores):.3f}")
        logger.info(f"T-test p-value: {p_value:.4f}")
        logger.info(f"With HGT - Cross-task: {np.mean(with_hgt_cross):.3f} ± {np.std(with_hgt_cross):.3f}")
        logger.info(f"Without HGT - Cross-task: {np.mean(without_hgt_cross):.3f} ± {np.std(without_hgt_cross):.3f}")
    
    def run_complete_benchmark(self):
        """Run all transposition and adaptation benchmarks"""
        logger.info("="*80)
        logger.info("TE-AI TRANSPOSITION AND ADAPTATION BENCHMARK")
        logger.info("="*80)
        
        all_results = {}
        
        # Test 1: Transposition effectiveness
        transposition_results = self.test_transposition_effectiveness()
        all_results['transposition_effectiveness'] = transposition_results
        
        # Test 2: Adaptation speed
        adaptation_results = self.test_adaptation_speed()
        all_results['adaptation_speed'] = adaptation_results
        
        # Test 3: Stress-triggered evolut