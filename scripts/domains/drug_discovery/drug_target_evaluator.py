"""
Drug Target Evaluator
=====================

Comprehensive evaluation system for drug targets using the full
TE-AI framework including quantum processing, dream consolidation,
and evolutionary dynamics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from scripts.core.quantum_gene import QuantumGeneModule
from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.phase_transition_detector import PhaseTransitionDetector
from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

from .drug_discovery_germinal_center import DrugDiscoveryGerminalCenter, DrugDiscoveryBCell
from .drug_target_antigen import DrugTargetAntigen

logger = get_logger()


@dataclass
class DrugTargetScore:
    """Comprehensive scoring for a drug target"""
    target_id: str
    overall_score: float
    components: Dict[str, float]
    quantum_coherence: float
    evolutionary_potential: float
    binding_profiles: List[Dict]
    phase_transition_data: Dict
    dream_insights: List[str]


class DrugTargetEvaluator:
    """
    Evaluates drug targets using the full power of TE-AI including
    quantum processing, phase transitions, and dream consolidation.
    """
    
    def __init__(
        self,
        germinal_center: Optional[DrugDiscoveryGerminalCenter] = None,
        use_phase_detection: bool = True,
        quantum_evaluation: bool = True,
        dream_analysis: bool = True
    ):
        """
        Initialize evaluator with optional custom germinal center.
        
        Args:
            germinal_center: Pre-configured germinal center or create new
            use_phase_detection: Enable phase transition detection
            quantum_evaluation: Use quantum genes for evaluation
            dream_analysis: Enable dream-based insights
        """
        self.germinal_center = germinal_center or DrugDiscoveryGerminalCenter(
            enable_quantum_dreams=dream_analysis
        )
        
        self.use_phase_detection = use_phase_detection
        self.quantum_evaluation = quantum_evaluation
        self.dream_analysis = dream_analysis
        
        if use_phase_detection:
            self.phase_detector = PhaseTransitionDetector(
                window_size=20
            )
            # Store phase threshold separately if needed
            self.phase_threshold = 0.15
            
        self.evaluation_history = []
        
    def evaluate_targets(
        self,
        targets: List[DrugTargetAntigen],
        generations: int = 20,
        parallel_evaluation: bool = True,
        stress_test: bool = True
    ) -> Dict[str, DrugTargetScore]:
        """
        Evaluate multiple drug targets comprehensively.
        
        Args:
            targets: List of drug targets to evaluate
            generations: Generations to evolve per target
            parallel_evaluation: Evaluate targets in parallel populations
            stress_test: Test with mutations and environmental stress
            
        Returns:
            Dictionary mapping target IDs to scores
        """
        logger.info(f"Evaluating {len(targets)} drug targets with TE-AI framework")
        scores = {}
        
        if parallel_evaluation and len(targets) > 1:
            # Create sub-populations for each target
            scores = self._parallel_evaluation(targets, generations, stress_test)
        else:
            # Sequential evaluation
            for target in targets:
                score = self._evaluate_single_target(target, generations, stress_test)
                scores[score.target_id] = score
                
        # Comparative analysis
        self._compute_relative_scores(scores)
        
        return scores
    
    def _evaluate_single_target(
        self,
        target: DrugTargetAntigen,
        generations: int,
        stress_test: bool
    ) -> DrugTargetScore:
        """Evaluate a single drug target"""
        target_id = target.protein_structure.pdb_id or f"target_{id(target)}"
        logger.info(f"\nEvaluating target: {target_id}")
        
        # Initialize tracking
        evolution_data = {
            'fitness_trajectory': [],
            'diversity_trajectory': [],
            'quantum_metrics': [],
            'phase_transitions': [],
            'transposition_events': []
        }
        
        # Convert to antigen batch
        target_batch = [target.to_graph() for _ in range(cfg.batch_size)]
        
        # Initial baseline
        initial_metrics = self._get_population_metrics()
        
        # Evolution phase
        for gen in range(generations):
            # Regular evolution
            self.germinal_center.evolve_generation(target_batch)
            
            # Track metrics
            evolution_data['fitness_trajectory'].append(
                self.germinal_center.fitness_landscape[-1]['mean_fitness']
            )
            evolution_data['diversity_trajectory'].append(
                self.germinal_center.diversity_metrics[-1]['shannon_index']
                if self.germinal_center.diversity_metrics else 0
            )
            
            # Quantum analysis
            if self.quantum_evaluation:
                quantum_metric = self._analyze_quantum_state()
                evolution_data['quantum_metrics'].append(quantum_metric)
                
            # Phase detection
            if self.use_phase_detection and len(evolution_data['fitness_trajectory']) > 5:
                phase = self.phase_detector.detect_phase_transition(
                    np.array(evolution_data['fitness_trajectory'][-20:])
                )
                if phase != 'stable':
                    evolution_data['phase_transitions'].append({
                        'generation': gen,
                        'phase': phase,
                        'fitness': evolution_data['fitness_trajectory'][-1]
                    })
                    
            # Track transpositions
            if hasattr(self.germinal_center, 'transposition_events'):
                evolution_data['transposition_events'].extend(
                    self.germinal_center.transposition_events
                )
                
            # Stress test at midpoint
            if stress_test and gen == generations // 2:
                logger.info(f"Applying stress test at generation {gen}")
                self._apply_stress_test(target)
                
            # Dream consolidation check
            if self.dream_analysis and gen % 5 == 0 and gen > 0:
                dream_insights = self._extract_dream_insights()
                logger.info(f"Dream insights at gen {gen}: {dream_insights}")
                
        # Final evaluation
        final_metrics = self._get_population_metrics()
        top_binders = self._get_top_binders(target, n=10)
        
        # Compute comprehensive score
        score_components = self._compute_score_components(
            target,
            evolution_data,
            initial_metrics,
            final_metrics,
            top_binders
        )
        
        # Extract binding profiles
        binding_profiles = self._extract_binding_profiles(top_binders, target)
        
        # Get dream insights
        dream_insights = []
        if self.dream_analysis and hasattr(self.germinal_center, 'dream_engine'):
            dream_insights = self._extract_dream_insights()
            
        # Create score object
        score = DrugTargetScore(
            target_id=target_id,
            overall_score=self._compute_overall_score(score_components),
            components=score_components,
            quantum_coherence=np.mean(evolution_data['quantum_metrics']) if evolution_data['quantum_metrics'] else 0,
            evolutionary_potential=self._compute_evolutionary_potential(evolution_data),
            binding_profiles=binding_profiles,
            phase_transition_data={
                'transitions': evolution_data['phase_transitions'],
                'stability': len(evolution_data['phase_transitions']) == 0
            },
            dream_insights=dream_insights
        )
        
        # Store in history
        self.evaluation_history.append({
            'target_id': target_id,
            'score': score,
            'evolution_data': evolution_data,
            'timestamp': time.time()
        })
        
        return score
    
    def _parallel_evaluation(
        self,
        targets: List[DrugTargetAntigen],
        generations: int,
        stress_test: bool
    ) -> Dict[str, DrugTargetScore]:
        """Evaluate multiple targets in parallel sub-populations"""
        logger.info("Starting parallel evaluation with sub-populations")
        
        # Split population into sub-populations
        cells_per_target = len(self.germinal_center.population) // len(targets)
        sub_populations = {}
        
        all_cells = list(self.germinal_center.population.items())
        for i, target in enumerate(targets):
            start_idx = i * cells_per_target
            end_idx = (i + 1) * cells_per_target if i < len(targets) - 1 else len(all_cells)
            
            sub_pop = dict(all_cells[start_idx:end_idx])
            sub_populations[target.protein_structure.pdb_id or f"target_{i}"] = {
                'target': target,
                'cells': sub_pop,
                'metrics': []
            }
            
        # Evolve sub-populations
        for gen in range(generations):
            for target_id, sub_data in sub_populations.items():
                # Create mini germinal center for sub-population
                mini_gc = DrugDiscoveryGerminalCenter(
                    population_size=len(sub_data['cells']),
                    enable_quantum_dreams=self.dream_analysis
                )
                mini_gc.population = sub_data['cells']
                
                # Evolve on specific target
                target_batch = [sub_data['target'].to_graph() for _ in range(cfg.batch_size)]
                mini_gc.evolve_generation(target_batch)
                
                # Update cells
                sub_data['cells'] = mini_gc.population
                sub_data['metrics'].append({
                    'fitness': mini_gc.fitness_landscape[-1]['mean_fitness'],
                    'diversity': mini_gc.diversity_metrics[-1] if mini_gc.diversity_metrics else {}
                })
                
            # Periodic cross-population exchange (migration)
            if gen % 10 == 0 and gen > 0:
                self._cross_population_exchange(sub_populations)
                
        # Evaluate each sub-population
        scores = {}
        for target_id, sub_data in sub_populations.items():
            # Create evaluation from sub-population data
            score = self._evaluate_from_subpopulation(
                sub_data['target'],
                sub_data['cells'],
                sub_data['metrics']
            )
            scores[target_id] = score
            
        return scores
    
    def _apply_stress_test(self, target: DrugTargetAntigen):
        """Apply stress test with mutations"""
        # Create mutated versions
        mutated_targets = []
        for _ in range(5):
            mutated = target.apply_disease_mutations()
            mutated_targets.append(mutated.to_graph())
            
        # Test population against mutants
        mutation_batch = mutated_targets * (cfg.batch_size // 5)
        self.germinal_center.evolve_generation(mutation_batch)
        
        # Trigger stress response
        self.germinal_center.current_stress = 0.8
        
    def _analyze_quantum_state(self) -> float:
        """Analyze quantum coherence in population"""
        quantum_cells = []
        
        for cell in self.germinal_center.population.values():
            for gene in cell.genes:
                if isinstance(gene, QuantumGeneModule) and gene.is_active:
                    if hasattr(gene, 'quantum_state') and gene.quantum_state is not None:
                        # Measure coherence
                        coherence = torch.abs(gene.quantum_state).mean().item()
                        quantum_cells.append(coherence)
                        
        return np.mean(quantum_cells) if quantum_cells else 0.0
    
    def _extract_dream_insights(self) -> List[str]:
        """Extract insights from dream consolidation"""
        insights = []
        
        if hasattr(self.germinal_center, 'dream_engine'):
            dream_engine = self.germinal_center.dream_engine
            
            # Check recent dreams
            if hasattr(dream_engine, 'dream_log') and dream_engine.dream_log:
                recent_dreams = dream_engine.dream_log[-5:]
                
                for dream in recent_dreams:
                    if 'insights' in dream:
                        insights.extend(dream['insights'])
                    
                    # Analyze dream patterns
                    if 'pattern' in dream:
                        if dream['pattern'] == 'convergent':
                            insights.append("Population converging on solution")
                        elif dream['pattern'] == 'divergent':
                            insights.append("Exploring alternative binding modes")
                        elif dream['pattern'] == 'oscillatory':
                            insights.append("Bistable binding states detected")
                            
        return insights
    
    def _compute_score_components(
        self,
        target: DrugTargetAntigen,
        evolution_data: Dict,
        initial_metrics: Dict,
        final_metrics: Dict,
        top_binders: List
    ) -> Dict[str, float]:
        """Compute individual scoring components"""
        components = {}
        
        # Affinity component
        if top_binders:
            components['max_affinity'] = top_binders[0][1]
            components['mean_top5_affinity'] = np.mean([b[1] for b in top_binders[:5]])
        else:
            components['max_affinity'] = 0.0
            components['mean_top5_affinity'] = 0.0
            
        # Evolution dynamics
        fitness_trajectory = evolution_data['fitness_trajectory']
        if fitness_trajectory:
            components['fitness_improvement'] = fitness_trajectory[-1] - fitness_trajectory[0]
            components['convergence_rate'] = self._calculate_convergence_rate(fitness_trajectory)
            components['stability'] = 1.0 - np.std(fitness_trajectory[-10:]) if len(fitness_trajectory) > 10 else 0.5
        else:
            components['fitness_improvement'] = 0.0
            components['convergence_rate'] = 0.0
            components['stability'] = 0.0
            
        # Diversity maintenance
        diversity_trajectory = evolution_data['diversity_trajectory']
        if diversity_trajectory:
            components['final_diversity'] = diversity_trajectory[-1]
            components['diversity_retention'] = diversity_trajectory[-1] / (diversity_trajectory[0] + 1e-6)
        else:
            components['final_diversity'] = 0.0
            components['diversity_retention'] = 0.0
            
        # Quantum coherence
        if evolution_data['quantum_metrics']:
            components['quantum_coherence'] = np.mean(evolution_data['quantum_metrics'])
            components['quantum_stability'] = 1.0 - np.std(evolution_data['quantum_metrics'])
        else:
            components['quantum_coherence'] = 0.0
            components['quantum_stability'] = 0.0
            
        # Phase transition adaptability
        phase_transitions = evolution_data['phase_transitions']
        components['adaptability'] = min(1.0, len(phase_transitions) * 0.2)  # More transitions = more adaptive
        components['phase_recovery'] = self._calculate_phase_recovery(phase_transitions, fitness_trajectory)
        
        # Target-specific features
        components['pocket_druggability'] = target.global_druggability
        components['selectivity_potential'] = target.selectivity_score
        
        return components
    
    def _calculate_convergence_rate(self, trajectory: List[float]) -> float:
        """Calculate how quickly fitness converges"""
        if len(trajectory) < 2:
            return 0.0
            
        # Find generation where 90% of final fitness is reached
        final_fitness = trajectory[-1]
        target_fitness = 0.9 * final_fitness
        
        for i, fitness in enumerate(trajectory):
            if fitness >= target_fitness:
                return 1.0 - (i / len(trajectory))
                
        return 0.0
    
    def _calculate_phase_recovery(
        self,
        transitions: List[Dict],
        fitness_trajectory: List[float]
    ) -> float:
        """Calculate recovery rate from phase transitions"""
        if not transitions or len(fitness_trajectory) < 2:
            return 1.0  # No transitions = perfect stability
            
        recovery_scores = []
        
        for transition in transitions:
            gen = transition['generation']
            if gen < len(fitness_trajectory) - 5:
                # Check fitness recovery after transition
                pre_fitness = fitness_trajectory[gen]
                post_fitness = max(fitness_trajectory[gen+1:gen+6])
                recovery = post_fitness / (pre_fitness + 1e-6)
                recovery_scores.append(min(1.0, recovery))
                
        return np.mean(recovery_scores) if recovery_scores else 0.5
    
    def _compute_overall_score(self, components: Dict[str, float]) -> float:
        """Compute weighted overall score"""
        weights = {
            'max_affinity': 0.25,
            'mean_top5_affinity': 0.15,
            'fitness_improvement': 0.10,
            'convergence_rate': 0.05,
            'stability': 0.05,
            'final_diversity': 0.05,
            'diversity_retention': 0.05,
            'quantum_coherence': 0.10,
            'quantum_stability': 0.05,
            'adaptability': 0.05,
            'phase_recovery': 0.05,
            'pocket_druggability': 0.025,
            'selectivity_potential': 0.025
        }
        
        # Ensure all weights sum to 1.0
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Compute weighted sum
        score = 0.0
        for component, value in components.items():
            if component in weights:
                score += weights[component] * value
                
        return score
    
    def _compute_evolutionary_potential(self, evolution_data: Dict) -> float:
        """Compute potential for further evolution"""
        # High diversity + stable fitness + successful transpositions = high potential
        
        potential = 0.0
        
        # Diversity component
        if evolution_data['diversity_trajectory']:
            final_diversity = evolution_data['diversity_trajectory'][-1]
            potential += min(1.0, final_diversity / 3.0) * 0.3
            
        # Transposition activity
        if evolution_data['transposition_events']:
            transposition_rate = len(evolution_data['transposition_events']) / len(evolution_data['fitness_trajectory'])
            potential += min(1.0, transposition_rate * 10) * 0.3
            
        # Phase transition success
        if evolution_data['phase_transitions']:
            # Successfully navigating phase transitions indicates adaptability
            potential += min(1.0, len(evolution_data['phase_transitions']) * 0.2) * 0.2
            
        # Quantum activity
        if evolution_data['quantum_metrics']:
            quantum_activity = np.mean(evolution_data['quantum_metrics'])
            potential += quantum_activity * 0.2
            
        return potential
    
    def _get_population_metrics(self) -> Dict:
        """Get current population metrics"""
        return {
            'population_size': len(self.germinal_center.population),
            'mean_fitness': self.germinal_center.fitness_landscape[-1]['mean_fitness']
            if self.germinal_center.fitness_landscape else 0.0,
            'diversity': self.germinal_center.diversity_metrics[-1]
            if self.germinal_center.diversity_metrics else {}
        }
    
    def _get_top_binders(
        self,
        target: DrugTargetAntigen,
        n: int = 10
    ) -> List[Tuple[DrugDiscoveryBCell, float]]:
        """Get top binding cells"""
        return self.germinal_center._get_top_drug_binders(target, n)
    
    def _extract_binding_profiles(
        self,
        top_binders: List[Tuple[DrugDiscoveryBCell, float]],
        target: DrugTargetAntigen
    ) -> List[Dict]:
        """Extract detailed binding profiles"""
        profiles = []
        
        for cell, affinity in top_binders[:5]:
            profile = {
                'cell_id': cell.cell_id,
                'affinity': affinity,
                'gene_signature': self.germinal_center._get_gene_signature(cell),
                'binding_pockets': [],
                'pharmacophores': {},
                'quantum_features': {}
            }
            
            # Extract pocket targeting
            if hasattr(cell, 'binding_profiles'):
                target_id = target.protein_structure.pdb_id or target.antigen_type
                if target_id in cell.binding_profiles:
                    binding_data = cell.binding_profiles[target_id]
                    if binding_data:
                        latest = binding_data[-1]
                        profile['binding_pockets'] = latest.get('pockets_targeted', [])
                        profile['pharmacophores'] = latest.get('pharmacophores', {})
                        
            # Extract quantum features
            for gene in cell.genes:
                if isinstance(gene, QuantumGeneModule) and gene.is_active:
                    profile['quantum_features'][gene.gene_id] = {
                        'coherence': torch.abs(gene.quantum_state).mean().item()
                        if hasattr(gene, 'quantum_state') and gene.quantum_state is not None else 0,
                        'entanglement': gene.entanglement_strength
                        if hasattr(gene, 'entanglement_strength') else 0
                    }
                    
            profiles.append(profile)
            
        return profiles
    
    def _compute_relative_scores(self, scores: Dict[str, DrugTargetScore]):
        """Compute relative rankings among all evaluated targets"""
        if len(scores) < 2:
            return
            
        # Normalize scores relative to best
        all_scores = [s.overall_score for s in scores.values()]
        max_score = max(all_scores)
        min_score = min(all_scores)
        
        for target_id, score in scores.items():
            # Add relative metrics
            score.components['relative_score'] = (score.overall_score - min_score) / (max_score - min_score + 1e-6)
            score.components['percentile'] = sum(1 for s in all_scores if s <= score.overall_score) / len(all_scores)
            
    def _evaluate_from_subpopulation(
        self,
        target: DrugTargetAntigen,
        cells: Dict,
        metrics: List[Dict]
    ) -> DrugTargetScore:
        """Create evaluation score from sub-population data"""
        target_id = target.protein_structure.pdb_id or f"target_{id(target)}"
        
        # Extract evolution data from metrics
        evolution_data = {
            'fitness_trajectory': [m['fitness'] for m in metrics],
            'diversity_trajectory': [m['diversity'].get('shannon_index', 0) for m in metrics],
            'quantum_metrics': [],
            'phase_transitions': [],
            'transposition_events': []
        }
        
        # Get top binders from final population
        top_binders = []
        for cell_id, cell in cells.items():
            if isinstance(cell, DrugDiscoveryBCell):
                affinity, _ = cell.process_drug_target(target)
                top_binders.append((cell, affinity.item()))
                
        top_binders.sort(key=lambda x: x[1], reverse=True)
        top_binders = top_binders[:10]
        
        # Compute score components
        initial_metrics = {'mean_fitness': metrics[0]['fitness'] if metrics else 0}
        final_metrics = {'mean_fitness': metrics[-1]['fitness'] if metrics else 0}
        
        score_components = self._compute_score_components(
            target, evolution_data, initial_metrics, final_metrics, top_binders
        )
        
        return DrugTargetScore(
            target_id=target_id,
            overall_score=self._compute_overall_score(score_components),
            components=score_components,
            quantum_coherence=0.0,  # Would need quantum analysis
            evolutionary_potential=self._compute_evolutionary_potential(evolution_data),
            binding_profiles=self._extract_binding_profiles(top_binders, target),
            phase_transition_data={'transitions': [], 'stability': True},
            dream_insights=[]
        )
    
    def _cross_population_exchange(self, sub_populations: Dict):
        """Exchange cells between sub-populations (migration)"""
        logger.info("Performing cross-population exchange")
        
        # Get best cells from each population
        migrants = {}
        for target_id, sub_data in sub_populations.items():
            # Sort cells by fitness
            sorted_cells = sorted(
                sub_data['cells'].items(),
                key=lambda x: x[1].fitness_history[-1] if x[1].fitness_history else 0,
                reverse=True
            )
            # Take top 10%
            n_migrants = max(1, len(sorted_cells) // 10)
            migrants[target_id] = sorted_cells[:n_migrants]
            
        # Exchange migrants
        target_ids = list(sub_populations.keys())
        for i, source_id in enumerate(target_ids):
            target_id = target_ids[(i + 1) % len(target_ids)]  # Next population
            
            # Add migrants to target population
            for cell_id, cell in migrants[source_id]:
                # Clone the cell
                new_cell = copy.deepcopy(cell)
                new_cell.cell_id = f"{cell_id}_migrant_{source_id}_to_{target_id}"
                
                # Add to target population
                sub_populations[target_id]['cells'][new_cell.cell_id] = new_cell
                
        logger.info(f"Migrated {sum(len(m) for m in migrants.values())} cells between populations")