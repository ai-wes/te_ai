# scripts/core/benchmark_evaluator.py

import torch
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.data import Batch
from scripts.core.production_b_cell import ProductionBCell
from typing import List, Dict

def benchmark_champion(champion_cell: ProductionBCell, test_data: List, device) -> Dict:
    """
    Evaluates a single champion cell on the unseen test set and returns
    a dictionary of standard performance metrics.
    """
    if not champion_cell or not test_data:
        return {}

    champion_cell.to(device)
    champion_cell.eval() # Set the model to evaluation mode

    test_batch = Batch.from_data_list([graph.to(device) for graph in test_data])
    true_labels = test_batch.y.cpu().numpy()
    
    all_predictions = []
    total_inference_time = 0.0

    with torch.no_grad():
        start_time = time.perf_counter()
        # Get the model's raw output scores (logits)
        predicted_scores, _, _ = champion_cell(test_batch)
        total_inference_time = time.perf_counter() - start_time

    predicted_scores_cpu = predicted_scores.cpu().numpy()
    # Convert raw scores to binary predictions (0 or 1)
    predicted_labels = (predicted_scores_cpu > 0.5).astype(int)

    avg_inference_ms = (total_inference_time / len(test_data)) * 1000

    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels) * 100,
        "precision": precision_score(true_labels, predicted_labels, zero_division=0) * 100,
        "recall": recall_score(true_labels, predicted_labels, zero_division=0) * 100,
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0) * 100,
        "roc_auc": roc_auc_score(true_labels, predicted_scores_cpu) * 100,
        "inference_time_ms": avg_inference_ms
    }
    
    return metrics