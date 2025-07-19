# scripts/core/fitness_functions.py

import numpy as np
from typing import Dict
from scripts.core.production_b_cell import ProductionBCell
import torch
import numpy as np
from typing import Dict, List
from Levenshtein import distance as levenshtein_distance

from scripts.core.production_b_cell import ProductionBCell
from scripts.config import cfg

CHAMPION_ARCHIVE = []


def calculate_accuracy_fitness(cell: ProductionBCell, eval_results: Dict) -> float:
    """Fitness for the 'Scientist'. Rewards pure predictive accuracy."""
    return eval_results.get('raw_fitness', 0.0)

def calculate_speed_fitness(cell: ProductionBCell, eval_results: Dict) -> float:
    """Fitness for the 'Engineer'. Rewards low inference time."""
    raw_fitness = eval_results.get('raw_fitness', 0.0)
    inference_time = eval_results.get('inference_time', 1.0)
    speed_score = np.exp(-inference_time / 0.001)
    return (0.1 * raw_fitness) + (0.9 * speed_score)

# --- Advanced Specialist Functions ---

def calculate_novelty_fitness(cell: ProductionBCell, eval_results: Dict) -> float:
    """
    Fitness for the 'Innovator'. Rewards architectural difference from known champions.
    """
    raw_fitness = eval_results.get('raw_fitness', 0.0)
    
    # 1. Create the cell's architectural fingerprint
    active_genes = [g for g in cell.genes if g.is_active]
    fingerprint = "-".join(sorted([g.gene_type for g in active_genes]))
    
    # 2. Compare to the archive of champions
    if not CHAMPION_ARCHIVE:
        novelty_score = 1.0 # The first one is always novel
    else:
        # Find the minimum distance to any known champion fingerprint
        min_dist = min([levenshtein_distance(fingerprint, champ_fp) for champ_fp in CHAMPION_ARCHIVE])
        # Normalize the score (longer fingerprints can have larger distances)
        max_len = max(len(fingerprint), max(map(len, CHAMPION_ARCHIVE)))
        novelty_score = min_dist / max_len if max_len > 0 else 0.0
        
    # 3. Combine with a small amount of accuracy to prevent total nonsense
    return (0.2 * raw_fitness) + (0.8 * novelty_score)

def calculate_robustness_fitness(cell: ProductionBCell, eval_results: Dict) -> float:
    """
    Fitness for the 'Adversary'. Rewards stability under data perturbation.
    NOTE: This requires the evaluator to be modified to return attack results.
    For now, we'll use a placeholder. A real implementation would be more complex.
    """
    raw_fitness = eval_results.get('raw_fitness', 0.0)
    # This value would be calculated by the evaluator after running attacks
    robustness_penalty = eval_results.get('robustness_penalty', 0.0)
    
    return raw_fitness - robustness_penalty

def calculate_parsimony_fitness(cell: ProductionBCell, eval_results: Dict) -> float:
    """
    Fitness for the 'Minimalist'. Rewards accuracy while heavily penalizing complexity.
    """
    raw_fitness = eval_results.get('raw_fitness', 0.0)
    
    # Count active genes and total parameters
    active_genes = [g for g in cell.genes if g.is_active]
    num_genes = len(active_genes)
    num_params = sum(p.numel() for g in active_genes for p in g.parameters())
    
    # Penalize based on a target (e.g., we want models under 500k params and 5 genes)
    param_penalty = max(0, (num_params - 500000) / 500000)
    gene_penalty = max(0, (num_genes - 5) / 5)
    
    # The final score is accuracy minus the penalties
    return raw_fitness - (0.5 * param_penalty) - (0.5 * gene_penalty)

def calculate_constrained_fitness(cell: ProductionBCell, eval_results: Dict) -> float:
    """
    Fitness for the 'Biologist'. Rewards accuracy while enforcing real-world rules.
    NOTE: This requires the evaluator to predict molecular properties. Placeholder for now.
    """
    raw_fitness = eval_results.get('raw_fitness', 0.0)
    # This value would be calculated by the evaluator
    constraint_violation_penalty = eval_results.get('constraint_penalty', 0.0)
    
    return raw_fitness - constraint_violation_penalty



def calculate_balanced_fitness(cell: ProductionBCell, evaluation_results: Dict) -> float:
    """Fitness for the 'Generalist' island. Balances accuracy and speed."""
    raw_fitness = evaluation_results.get('raw_fitness', 0.0)
    inference_time = evaluation_results.get('inference_time', 1.0)
    speed_score = np.exp(-inference_time / 0.001)
    return (0.6 * raw_fitness) + (0.4 * speed_score)

# --- The Master Dictionary ---
# This MUST contain keys that match your config file.
FITNESS_FUNCTIONS = {
    "accuracy": calculate_accuracy_fitness,
    "speed": calculate_speed_fitness,
    "novelty": calculate_novelty_fitness,
    "robustness": calculate_robustness_fitness,
    "parsimony": calculate_parsimony_fitness,
    "constrained": calculate_constrained_fitness,
    "balanced": calculate_balanced_fitness
}