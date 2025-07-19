# scripts/core/fitness_functions.py

import numpy as np
import time
from typing import Dict
from scripts.core.production_b_cell import ProductionBCell

def calculate_accuracy_fitness(cell: ProductionBCell, evaluation_results: Dict) -> float:
    """Standard fitness based on predictive accuracy."""
    return evaluation_results.get('mean_affinity', 0.0)

def calculate_speed_fitness(cell: ProductionBCell, evaluation_results: Dict) -> float:
    """Fitness that heavily rewards low inference time."""
    mean_affinity = evaluation_results.get('mean_affinity', 0.0)
    inference_time = evaluation_results.get('inference_time', 1.0) # Default to 1s to avoid division by zero
    
    # Create a speed score. Lower is better. We want to maximize the score.
    # A time of 10ms (0.01s) should be a near-perfect score.
    speed_score = np.clip(1.0 - (inference_time / 0.01), 0, 1)
    
    # Heavily weight the speed score
    return (0.2 * mean_affinity) + (0.8 * speed_score)

def calculate_balanced_fitness(cell: ProductionBCell, evaluation_results: Dict) -> float:
    """A balanced fitness between accuracy and speed."""
    mean_affinity = evaluation_results.get('mean_affinity', 0.0)
    inference_time = evaluation_results.get('inference_time', 1.0)
    
    speed_score = np.clip(1.0 - (inference_time / 0.01), 0, 1)
    
    # Balance the two objectives
    return (0.6 * mean_affinity) + (0.4 * speed_score)

# A dictionary to easily access the functions by name
FITNESS_FUNCTIONS = {
    "accuracy": calculate_accuracy_fitness,
    "speed": calculate_speed_fitness,
    "balanced": calculate_balanced_fitness,
}