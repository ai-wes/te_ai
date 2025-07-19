# scripts/core/fitness_functions.py

import numpy as np
from typing import Dict
from scripts.core.production_b_cell import ProductionBCell

def calculate_accuracy_fitness(cell: ProductionBCell, evaluation_results: Dict) -> float:
    """Fitness for the 'Scientist' island. Purely rewards predictive accuracy."""
    return evaluation_results.get('raw_fitness', 0.0)

def calculate_speed_fitness(cell: ProductionBCell, evaluation_results: Dict) -> float:
    """Fitness for the 'Engineer' island. Heavily rewards low inference time."""
    raw_fitness = evaluation_results.get('raw_fitness', 0.0)
    inference_time = evaluation_results.get('inference_time', 1.0)
    speed_score = np.exp(-inference_time / 0.001)
    return (0.1 * raw_fitness) + (0.9 * speed_score)

def calculate_balanced_fitness(cell: ProductionBCell, evaluation_results: Dict) -> float:
    """Fitness for the 'Generalist' island. Balances accuracy and speed."""
    raw_fitness = evaluation_results.get('raw_fitness', 0.0)
    inference_time = evaluation_results.get('inference_time', 1.0)
    speed_score = np.exp(-inference_time / 0.001)
    return (0.6 * raw_fitness) + (0.4 * speed_score)

FITNESS_FUNCTIONS = {
    "accuracy": calculate_accuracy_fitness,
    "speed": calculate_speed_fitness,
    "balanced": calculate_balanced_fitness,
}