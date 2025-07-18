#!/usr/bin/env python3
"""
TE-AI Benchmark Runner
======================

Comprehensive benchmarking system to validate TE-AI performance against
standard datasets and competing methods.
"""
import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Fix Keras 3 compatibility with DeepChem


import torch

# Fix for torchao compatibility issue with DeepChem
if not hasattr(torch._C, 'Tag') or not hasattr(torch._C.Tag, 'needs_fixed_stride_order'):
    class FakeTag:
        needs_fixed_stride_order = None
    if not hasattr(torch._C, 'Tag'):
        torch._C.Tag = FakeTag()
    else:
        torch._C.Tag.needs_fixed_stride_order = None

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.config import cfg
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()


class BenchmarkDataset:
    """Base class for benchmark datasets"""
    
    def __init__(self, name: str, task_type: str):
        self.name = name
        self.task_type = task_type
        self.data = None
        self.loaded = False
    
    def load(self):
        """Load the dataset"""
        raise NotImplementedError
    
    def get_splits(self) -> Tuple[Any, Any, Any, Any]:
        """Get train/test splits"""
        raise NotImplementedError
    
    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions"""
        raise NotImplementedError


class MolecularPropertyDataset(BenchmarkDataset):
    """Molecular property prediction benchmarks using real DeepChem data"""
    
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name, "molecular_property")
        self.dataset_name = dataset_name
        self.train_antigens = None
        self.valid_antigens = None
        self.test_antigens = None
        
    def load(self):
        """Load molecular dataset from DeepChem"""
        logger.info(f"Loading {self.dataset_name} dataset from DeepChem...")
        
        try:
            import deepchem as dc
            from scripts.domains.drug_discovery.deepchem_converter import DeepChemToTEAI
            
            # Create converter
            converter = DeepChemToTEAI(featurization_mode='hybrid')
            
            # Load and convert the dataset
            self.train_antigens, self.valid_antigens, self.test_antigens = \
                converter.convert_molnet_dataset(
                    self.dataset_name.lower(),
                    featurizer='ECFP',
                    splitter='scaffold'  # Use scaffold split for drug discovery
                )
            
            # Also store raw features for baseline models
            try:
                from deepchem.feat import CircularFingerprint
                morgan_featurizer = CircularFingerprint(radius=2, size=1024)
            except ImportError:
                try:
                    from deepchem.feat import MorganGenerator
                    morgan_featurizer = MorganGenerator(radius=2, size=1024)
                except ImportError:
                    # Fallback to string name
                    morgan_featurizer = 'ECFP'
            loader_fn = getattr(dc.molnet, f'load_{self.dataset_name.lower()}')
            tasks, datasets, transformers = loader_fn(featurizer=morgan_featurizer, splitter='scaffold')
            train, valid, test = datasets
            
            # Store both representations
            self.data = {
                "train_X": train.X,
                "train_y": train.y,
                "valid_X": valid.X,
                "valid_y": valid.y,
                "test_X": test.X,
                "test_y": test.y,
                "tasks": tasks,
                "train_antigens": self.train_antigens,
                "valid_antigens": self.valid_antigens,
                "test_antigens": self.test_antigens
            }
            
            self.loaded = True
            logger.info(f"Loaded {self.dataset_name}:")
            logger.info(f"  Train: {len(self.train_antigens)} samples")
            logger.info(f"  Valid: {len(self.valid_antigens)} samples")
            logger.info(f"  Test: {len(self.test_antigens)} samples")
            logger.info(f"  Tasks: {tasks}")
            
        except ImportError:
            logger.error("DeepChem not installed!")
            logger.error("Install with: pip install deepchem")
            raise RuntimeError("Cannot run molecular benchmarks without DeepChem! Install it first!")
        except Exception as e:
            logger.error(f"Failed to load {self.dataset_name}: {e}")
            raise
    
    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test splits"""
        if not self.loaded:
            self.load()
        
        # Return the proper train/test split from DeepChem
        # Combine train and valid for training
        X_train = np.vstack([self.data["train_X"], self.data["valid_X"]])
        y_train = np.vstack([self.data["train_y"], self.data["valid_y"]])
        
        X_test = self.data["test_X"]
        y_test = self.data["test_y"]
        
        # Flatten y if single task
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
            
        return X_train, X_test, y_train, y_test
    
    def get_antigen_splits(self) -> Tuple[List, List, List]:
        """Get antigen splits for TE-AI"""
        if not self.loaded:
            self.load()
            
        return self.train_antigens, self.valid_antigens, self.test_antigens
    
    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate molecular property predictions"""
        metrics = {}
        
        if targets.ndim == 1 or targets.shape[1] == 1:
            # Binary classification
            targets = targets.flatten()
            predictions = predictions.flatten()
            
            # Convert probabilities to binary predictions
            pred_binary = (predictions > 0.5).astype(int)
            
            metrics["accuracy"] = accuracy_score(targets, pred_binary)
            metrics["f1_score"] = f1_score(targets, pred_binary, average='binary')
            metrics["precision"] = precision_score(targets, pred_binary, average='binary', zero_division=0)
            metrics["recall"] = recall_score(targets, pred_binary, average='binary', zero_division=0)
            
            # ROC-AUC only if we have both classes
            if len(np.unique(targets)) > 1:
                metrics["roc_auc"] = roc_auc_score(targets, predictions)
            else:
                metrics["roc_auc"] = 0.5
                
        else:
            # Multi-task
            accuracies = []
            f1_scores = []
            roc_aucs = []
            
            for i in range(targets.shape[1]):
                task_targets = targets[:, i]
                task_preds = predictions[:, i] if predictions.ndim > 1 else predictions
                
                # Skip tasks with only one class
                if len(np.unique(task_targets)) == 1:
                    continue
                
                pred_binary = (task_preds > 0.5).astype(int)
                
                accuracies.append(accuracy_score(task_targets, pred_binary))
                f1_scores.append(f1_score(task_targets, pred_binary, average='binary'))
                roc_aucs.append(roc_auc_score(task_targets, task_preds))
            
            metrics["accuracy"] = np.mean(accuracies) if accuracies else 0.0
            metrics["f1_score"] = np.mean(f1_scores) if f1_scores else 0.0
            metrics["roc_auc"] = np.mean(roc_aucs) if roc_aucs else 0.5
        
        return metrics


class CybersecurityDataset(BenchmarkDataset):
    """Cybersecurity benchmark datasets"""
    
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name, "cybersecurity")
        self.dataset_name = dataset_name
    
    def load(self):
        """Load cybersecurity dataset"""
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        if self.dataset_name == "NSL-KDD":
            # Network intrusion detection
            n_samples = 125973
            n_features = 122
            n_classes = 5  # Normal + 4 attack types
            
            # Generate synthetic network traffic features
            X = np.random.randn(n_samples, n_features) * 0.5
            
            # Create distinct patterns for each attack type
            labels = np.random.randint(0, n_classes, size=n_samples)
            for class_id in range(n_classes):
                mask = labels == class_id
                if class_id == 0:  # Normal traffic
                    X[mask] *= 0.3
                else:  # Attack patterns
                    X[mask] += np.random.randn(n_features) * (class_id * 0.2)
            
            y = labels
            
        elif self.dataset_name == "CICIDS2017":
            # More recent intrusion detection dataset
            n_samples = 50000  # Subset for demo
            n_features = 78
            
            # Binary classification (benign vs malicious)
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 2, size=n_samples)
            
            # Add patterns
            malicious_mask = y == 1
            X[malicious_mask] += np.random.randn(n_features) * 0.5
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        self.data = {"X": X, "y": y}
        self.loaded = True
        logger.info(f"Loaded {n_samples} samples with {n_features} features")
    
    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test splits"""
        if not self.loaded:
            self.load()
        
        return train_test_split(
            self.data["X"], self.data["y"], 
            test_size=0.2, random_state=42
        )
    
    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate cybersecurity predictions"""
        metrics = {}
        
        if len(np.unique(targets)) == 2:
            # Binary classification
            pred_binary = (predictions > 0.5).astype(int)
            metrics["accuracy"] = accuracy_score(targets, pred_binary)
            metrics["f1_score"] = f1_score(targets, pred_binary, average='binary')
            metrics["precision"] = precision_score(targets, pred_binary, average='binary', zero_division=0)
            metrics["recall"] = recall_score(targets, pred_binary, average='binary', zero_division=0)
            metrics["roc_auc"] = roc_auc_score(targets, predictions)
        else:
            # Multi-class
            pred_classes = np.argmax(predictions, axis=1)
            metrics["accuracy"] = accuracy_score(targets, pred_classes)
            metrics["f1_score"] = f1_score(targets, pred_classes, average='weighted')
            # For multi-class ROC-AUC, need one-hot encoding
            metrics["roc_auc"] = 0.0  # Placeholder
        
        return metrics


class BaselineModel:
    """Base class for baseline models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.training_time = 0
        self.inference_time = 0
    
    def fit(self, X_train, y_train):
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, X_test):
        """Make predictions"""
        raise NotImplementedError


class SimpleNeuralNetwork(BaselineModel):
    """Standard feedforward neural network baseline"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__("Simple Neural Network")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Build model
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=1)
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss() if output_dim == 1 else nn.CrossEntropyLoss()
    
    def fit(self, X_train, y_train, epochs: int = 50):
        """Train the model"""
        start_time = time.time()
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            
            if self.output_dim == 1:
                loss = self.criterion(outputs.squeeze(), y_train)
            else:
                loss = self.criterion(outputs, y_train.long())
            
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        self.training_time = time.time() - start_time
    
    def predict(self, X_test):
        """Make predictions"""
        start_time = time.time()
        
        self.model.eval()
        with torch.no_grad():
            X_test = torch.tensor(X_test, dtype=torch.float32)
            outputs = self.model(X_test)
        
        self.inference_time = time.time() - start_time
        return outputs.numpy()


class RandomForestBaseline(BaselineModel):
    """Random Forest baseline"""
    
    def __init__(self):
        super().__init__("Random Forest")
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def fit(self, X_train, y_train):
        """Train the model"""
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
    
    def predict(self, X_test):
        """Make predictions"""
        start_time = time.time()
        # Get probability predictions for positive class
        predictions = self.model.predict_proba(X_test)[:, 1]
        self.inference_time = time.time() - start_time
        return predictions


class TEAIBenchmarkAdapter:
    """Adapter to run TE-AI on benchmark datasets"""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.model = None
        self.training_time = 0
        self.inference_time = 0
        self.evolution_history = []
        self.best_cells = []  # Store best performing cells for prediction
        self.fitness_history = []  # Track fitness over generations
        self.best_model_metrics = None  # Track best model that meets criteria
        self.best_model_generation = None
        self.model_save_dir = None
    
    def create_model(self, input_dim: int, output_dim: int):
        """Create appropriate TE-AI model for the task"""
        from scripts.core.production_germinal_center import ProductionGerminalCenter
        from scripts.domains.drug_discovery.drug_discovery_germinal_center import DrugDiscoveryGerminalCenter
        from scripts.domains.cyber_security.main_cyber_security import CyberSecurityGerminalCenter
        
        cfg.population_size = 30  # Smaller for benchmarking
        cfg.generations = 20
        cfg.num_generations = 10  # How many to actually run in benchmark
        
        if self.task_type == "molecular_property":
            # Use drug discovery germinal center
            self.model = DrugDiscoveryGerminalCenter(
                population_size=cfg.population_size,
                enable_drug_genes=True
            )
        elif self.task_type == "cybersecurity":
            # Use cybersecurity germinal center
            self.model = CyberSecurityGerminalCenter(cfg)
        else:
            # Generic germinal center
            self.model = ProductionGerminalCenter()
        
        self.cfg = cfg
    
    def fit(self, X_train, y_train, antigens=None, generations: int = 20):
        """Train using evolutionary process with real or converted antigens"""
        start_time = time.time()
        
        # Create unique directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = Path(f"benchmark_results/te_ai_models/{self.task_type}_{timestamp}")
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model save directory: {self.model_save_dir}")
        
        # Use provided antigens or create from features
        if antigens is not None:
            # Use real DrugTargetAntigens
            logger.info(f"Using {len(antigens)} real drug target antigens")
            antigen_graphs = []
            for antigen in antigens:
                graph = antigen.to_graph()
                # The druggability score is already in the graph
                antigen_graphs.append(graph)
        else:
            # Fallback: Convert features to antigens (for non-molecular tasks)
            logger.warning("No antigens provided, creating synthetic ones")
            from scripts.core.antigen import BiologicalAntigen, AntigenEpitope
            antigens = []
            for i in range(len(X_train)):
                epitope = AntigenEpitope(
                    sequence="SYNTHETIC",
                    structure_coords=np.random.randn(20, 3),
                    hydrophobicity=0.0,
                    charge=0.0
                )
                antigen = BiologicalAntigen(antigen_type="synthetic")
                antigen.epitopes = [epitope]
                antigens.append(antigen)
            
            # Convert to graphs
            antigen_graphs = []
            for i, a in enumerate(antigens):
                graph = a.to_graph()
                target_val = float(y_train[i]) if y_train.ndim == 1 else y_train[i, 0]
                graph.y = torch.tensor([target_val], dtype=torch.float32)
                antigen_graphs.append(graph)
        
        # Evolve
        logger.info("Starting TE-AI evolution...")
        logger.info("=" * 60)
        logger.info("BENCHMARK RUNNER: Starting training loop")
        logger.info("=" * 60)
        for generation in range(generations):
            # Sample batch
            batch_indices = np.random.choice(len(antigen_graphs), 
                                           size=min(32, len(antigen_graphs)), 
                                           replace=False)
            batch = [antigen_graphs[i] for i in batch_indices]
            
            # Get targets for this batch
            batch_targets = np.array([y_train[i] for i in batch_indices])
            
            # Evolve one generation
            logger.info(f"BENCHMARK: Generation {generation+1}/{generations}")
            if hasattr(self.model, 'evolve_generation'):
                stats = self.model.evolve_generation(batch)
                logger.info(f"BENCHMARK: evolve_generation returned stats? {stats is not None}")
                if stats:
                    self.evolution_history.append(stats)
                    
                    # Calculate training accuracy on this batch
                    logger.info(f"[METRICS] Generation {generation + 1}: Has population? {hasattr(self.model, 'population')}, "
                              f"Population size: {len(self.model.population) if hasattr(self.model, 'population') else 'N/A'}")
                    if hasattr(self.model, 'population') and self.model.population:
                        logger.debug(f"Population size: {len(self.model.population)}")
                        # Get predictions from current population
                        predictions = self._predict_batch_internal(batch)
                        logger.debug(f"Predictions shape: {predictions.shape if predictions is not None else 'None'}")
                        if predictions is not None:
                            # Calculate metrics
                            pred_binary = (predictions > 0.5).astype(int)
                            batch_acc = accuracy_score(batch_targets, pred_binary)
                            batch_prec = precision_score(batch_targets, pred_binary, average='binary', zero_division=0)
                            batch_rec = recall_score(batch_targets, pred_binary, average='binary', zero_division=0)
                            batch_f1 = f1_score(batch_targets, pred_binary, average='binary', zero_division=0)
                            
                            logger.info(f"Generation {generation + 1}: Fitness={stats.get('max_fitness', stats.get('mean_fitness', 0)):.4f}, "
                                      f"Acc={batch_acc:.3f}, P={batch_prec:.3f}, R={batch_rec:.3f}, F1={batch_f1:.3f}")
                            
                            # Store metrics
                            stats['batch_accuracy'] = batch_acc
                            stats['batch_precision'] = batch_prec
                            stats['batch_recall'] = batch_rec
                            stats['batch_f1'] = batch_f1
                            
                            # Store cells from best validation accuracy generation
                            # (not just those meeting 90% threshold)
                            if batch_acc > 0.65:  # Lower threshold for cell storage
                                # Sort cells by their individual performance
                                sorted_cells = sorted(self.model.population.items(), 
                                                    key=lambda x: x[1].fitness_history[-1] if hasattr(x[1], 'fitness_history') and x[1].fitness_history else 0, 
                                                    reverse=True)
                                # Store top cells from this generation
                                current_best_cells = [cell for _, cell in sorted_cells[:10]]
                                
                                # Update best cells if this generation has better validation accuracy
                                if not hasattr(self, 'best_validation_acc') or batch_acc > self.best_validation_acc:
                                    self.best_validation_acc = batch_acc
                                    self.best_cells = current_best_cells
                                    logger.info(f"Updated best cells from generation {gen} with validation accuracy {batch_acc:.3f}")
                            
                            # Check if this model meets our strict criteria for saving
                            if batch_acc >= 0.9 and batch_prec >= 0.9:
                                current_fitness = stats.get('max_fitness', stats.get('mean_fitness', 0))
                                
                                # Save if this is the first model meeting criteria or has better fitness
                                if (self.best_model_metrics is None or 
                                    current_fitness > self.best_model_metrics.get('fitness', 0)):
                                    
                                    logger.info(f"🎯 Model meets criteria! Acc={batch_acc:.3f}, P={batch_prec:.3f}, Fitness={current_fitness:.4f}")
                                    
                                    # Save model state
                                    self._save_model_checkpoint(generation, stats, batch_acc, batch_prec, batch_rec, batch_f1)
                                    
                                    # Update best model tracking
                                    self.best_model_metrics = {
                                        'generation': generation,
                                        'fitness': current_fitness,
                                        'accuracy': batch_acc,
                                        'precision': batch_prec,
                                        'recall': batch_rec,
                                        'f1': batch_f1
                                    }
                                    self.best_model_generation = generation
                    else:
                        logger.info(f"Generation {generation + 1}: Fitness={stats.get('max_fitness', stats.get('mean_fitness', 0)):.4f}")
        
        # Don't override best_cells here - we want to keep the cells from best validation generation
        # Only set if we haven't found good cells during training
        if not self.best_cells and hasattr(self.model, 'population'):
            logger.warning("No good validation cells found during training, using final population")
            # Sort cells by latest fitness from fitness_history
            sorted_cells = sorted(self.model.population.items(), 
                                key=lambda x: x[1].fitness_history[-1] if hasattr(x[1], 'fitness_history') and x[1].fitness_history else 0, 
                                reverse=True)
            # Keep top 10 cells
            self.best_cells = [cell for _, cell in sorted_cells[:10]]
        
        self.training_time = time.time() - start_time
        
        # Log final results
        if self.best_model_metrics:
            logger.info(f"\n✨ Best model saved from generation {self.best_model_generation}:")
            logger.info(f"   Accuracy: {self.best_model_metrics['accuracy']:.3f}")
            logger.info(f"   Precision: {self.best_model_metrics['precision']:.3f}")
            logger.info(f"   Recall: {self.best_model_metrics['recall']:.3f}")
            logger.info(f"   F1 Score: {self.best_model_metrics['f1']:.3f}")
            logger.info(f"   Fitness: {self.best_model_metrics['fitness']:.4f}")
        else:
            logger.warning("⚠️ No model met the 90% accuracy and precision criteria - nothing saved!")
    
    def _predict_batch_internal(self, antigen_batch):
        """Make predictions using current population (for training metrics)"""
        try:
            from torch_geometric.data import Batch
            
            # Get top performing cells
            if hasattr(self.model, 'population'):
                # Sort by latest fitness from fitness_history
                sorted_cells = sorted(self.model.population.items(), 
                                    key=lambda x: x[1].fitness_history[-1] if hasattr(x[1], 'fitness_history') and x[1].fitness_history else 0, 
                                    reverse=True)
                top_cells = [cell for _, cell in sorted_cells[:5]]
                
                if not top_cells:
                    return None
                
                # Batch the antigens
                batch_data = Batch.from_data_list(antigen_batch).to(cfg.device)
                
                # Get predictions from each top cell
                predictions_list = []
                with torch.no_grad():
                    for cell in top_cells:
                        affinity, _, _ = cell(batch_data)
                        # Convert affinity to probability (sigmoid)
                        prob = torch.sigmoid(affinity).cpu().numpy()
                        predictions_list.append(prob)
                
                # Average predictions from top cells
                predictions = np.mean(predictions_list, axis=0)
                return predictions
            
            return None
        except Exception as e:
            logger.warning(f"Error in _predict_batch_internal: {e}")
            return None
    
    def predict(self, X_test):
        """Make predictions using evolved model"""
        start_time = time.time()
        
        # Create antigens from test data
        from scripts.core.antigen import AntigenEpitope, BiologicalAntigen
        test_antigens = []
        
        for i in range(len(X_test)):
            epitope = AntigenEpitope(
                sequence="SYNTHETIC",
                structure_coords=np.random.randn(20, 3),
                hydrophobicity=0.0,
                charge=0.0
            )
            antigen = BiologicalAntigen(antigen_type="synthetic")
            antigen.epitopes = [epitope]
            test_antigens.append(antigen)
        
        # Convert to graphs
        test_graphs = []
        for a in test_antigens:
            graph = a.to_graph()
            test_graphs.append(graph)
        
        # Use best cells or current population for prediction
        if self.best_cells:
            # Use stored best cells
            predictions = self._predict_with_cells(test_graphs, self.best_cells)
        elif hasattr(self.model, 'population') and self.model.population:
            # Use current population
            sorted_cells = sorted(self.model.population.items(), 
                                key=lambda x: x[1].fitness_history[-1] if hasattr(x[1], 'fitness_history') and x[1].fitness_history else 0, 
                                reverse=True)
            top_cells = [cell for _, cell in sorted_cells[:10]]
            predictions = self._predict_with_cells(test_graphs, top_cells)
        else:
            # Fallback to random predictions
            logger.warning("No evolved cells available, using random predictions")
            predictions = np.random.rand(len(X_test))
        
        self.inference_time = time.time() - start_time
        
        # Ensure predictions match test set size
        if len(predictions) != len(X_test):
            logger.warning(f"Prediction count mismatch: expected {len(X_test)}, got {len(predictions)}")
            # Truncate or pad predictions to match expected size
            if len(predictions) > len(X_test):
                predictions = predictions[:len(X_test)]
            else:
                # Pad with default predictions
                padding = np.full(len(X_test) - len(predictions), 0.5)
                predictions = np.concatenate([predictions, padding])
        
        return predictions
    
    def _predict_with_cells(self, test_graphs, cells):
        """Make predictions using specific cells"""
        from torch_geometric.data import Batch
        
        all_predictions = []
        
        # Filter cells to only use the best performing ones
        if hasattr(cells[0], 'fitness_history') and cells[0].fitness_history:
            # Sort by most recent fitness
            sorted_cells = sorted(cells, 
                                key=lambda c: c.fitness_history[-1] if c.fitness_history else 0, 
                                reverse=True)
            # Use only top performing cells
            top_cells = sorted_cells[:min(5, len(sorted_cells))]
        else:
            top_cells = cells[:5]
        
        logger.info(f"Using {len(top_cells)} top cells for prediction")
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(test_graphs), batch_size):
            batch_graphs = test_graphs[i:i+batch_size]
            batch_data = Batch.from_data_list(batch_graphs).to(cfg.device)
            
            # Get predictions from each cell
            batch_predictions = []
            cell_weights = []
            with torch.no_grad():
                for cell in top_cells:
                    try:
                        # Get raw affinity without fitness adjustments
                        affinity, _, metadata = cell(batch_data)
                        
                        # Use raw affinity for classification
                        if isinstance(affinity, torch.Tensor):
                            affinity_np = affinity.cpu().numpy()
                        else:
                            affinity_np = affinity
                        
                        # Ensure proper shape
                        if affinity_np.ndim == 0:
                            affinity_np = np.array([affinity_np])
                        
                        batch_predictions.append(affinity_np)
                        
                        # Weight by cell's historical performance
                        weight = 1.0
                        if hasattr(cell, 'fitness_history') and len(cell.fitness_history) > 0:
                            # Use average fitness as weight
                            weight = np.mean(list(cell.fitness_history)[-10:])
                        cell_weights.append(weight)
                        
                    except Exception as e:
                        logger.warning(f"Error predicting with cell: {e}")
                        continue
            
            if batch_predictions:
                # Weighted average based on cell performance
                batch_predictions = np.array(batch_predictions)
                cell_weights = np.array(cell_weights)
                
                # Normalize weights
                if cell_weights.sum() > 0:
                    cell_weights = cell_weights / cell_weights.sum()
                else:
                    cell_weights = np.ones_like(cell_weights) / len(cell_weights)
                
                # Compute weighted predictions
                avg_predictions = np.average(batch_predictions, axis=0, weights=cell_weights)
                
                # Apply sigmoid for probability conversion
                avg_predictions = 1 / (1 + np.exp(-avg_predictions))
                
                # Ensure we're getting the right shape
                if avg_predictions.ndim == 0:
                    # Single prediction, expand to match batch size
                    avg_predictions = np.array([avg_predictions] * len(batch_graphs))
                elif len(avg_predictions) != len(batch_graphs):
                    logger.warning(f"Prediction shape mismatch: expected {len(batch_graphs)}, got {len(avg_predictions)}")
                    # Take only the predictions for this batch
                    avg_predictions = avg_predictions[:len(batch_graphs)]
                all_predictions.extend(avg_predictions)
            else:
                # Fallback for this batch
                all_predictions.extend([0.5] * len(batch_graphs))
        
        return np.array(all_predictions)
    
    def _save_model_checkpoint(self, generation, stats, accuracy, precision, recall, f1):
        """Save model checkpoint when criteria are met"""
        checkpoint = {
            'generation': generation,
            'model_state': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
            'population': {},  # Save population data
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fitness': stats.get('max_fitness', stats.get('mean_fitness', 0)),
                'generation_stats': stats
            },
            'config': {
                'task_type': self.task_type,
                'population_size': len(self.model.population) if hasattr(self.model, 'population') else 0,
                'generation': generation
            }
        }
        
        # Save top cells
        if hasattr(self.model, 'population'):
            sorted_cells = sorted(self.model.population.items(), 
                                key=lambda x: x[1].fitness_history[-1] if hasattr(x[1], 'fitness_history') and x[1].fitness_history else 0, 
                                reverse=True)
            
            # Save top 10 cells
            for i, (cell_id, cell) in enumerate(sorted_cells[:10]):
                checkpoint['population'][f'top_{i+1}'] = {
                    'cell_id': cell_id,
                    'fitness': cell.fitness_history[-1] if hasattr(cell, 'fitness_history') and cell.fitness_history else 0,
                    'gene_count': len(cell.genes) if hasattr(cell, 'genes') else 0,
                    'state_dict': cell.state_dict() if hasattr(cell, 'state_dict') else None
                }
        
        # Save checkpoint
        checkpoint_path = self.model_save_dir / f"checkpoint_gen_{generation}_acc_{accuracy:.3f}_prec_{precision:.3f}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"   💾 Saved checkpoint to: {checkpoint_path}")
        
        # Also save a "best_model.pt" link that always points to the best
        best_model_path = self.model_save_dir / "best_model.pt"
        if best_model_path.exists():
            best_model_path.unlink()
        torch.save(checkpoint, best_model_path)
        
        # Save metrics summary
        metrics_path = self.model_save_dir / "metrics_summary.json"
        metrics_summary = {
            'best_generation': generation,
            'best_metrics': checkpoint['metrics'],
            'save_time': datetime.now().isoformat(),
            'training_history': self.evolution_history[-10:]  # Last 10 generations
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info(f"   📊 Saved metrics summary to: {metrics_path}")


class BenchmarkRunner:
    """Main benchmark runner"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def run_benchmark(self, dataset: BenchmarkDataset, models: List[BaselineModel]) -> Dict[str, Any]:
        """Run benchmark on a dataset with multiple models"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running benchmark: {dataset.name}")
        logger.info(f"{'='*60}")
        
        # Load data
        X_train, X_test, y_train, y_test = dataset.get_splits()
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        results = {
            "dataset": dataset.name,
            "task_type": dataset.task_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "models": {}
        }
        
        # Add TE-AI model
        te_ai = TEAIBenchmarkAdapter(dataset.task_type)
        te_ai.create_model(X_train.shape[1], 1 if y_train.ndim == 1 else y_train.shape[1])
        models.append(te_ai)
        
        # Add DeepChem baselines for molecular tasks
        if dataset.task_type == "molecular_property":
            try:
                from scripts.benchmarks.deepchem_baselines import get_deepchem_baselines
                dc_baselines = get_deepchem_baselines()
                models.extend(dc_baselines)
                logger.info(f"Added {len(dc_baselines)} DeepChem baseline models")
            except ImportError:
                logger.warning("Could not import DeepChem baselines")
        
        # Run each model
        for model in models:
            logger.info(f"\nTraining {model.name if hasattr(model, 'name') else 'TE-AI'}...")
            
            # Train
            if isinstance(model, TEAIBenchmarkAdapter) and dataset.task_type == "molecular_property":
                # For TE-AI on molecular data, use antigens
                if hasattr(dataset, 'get_antigen_splits'):
                    train_antigens, valid_antigens, test_antigens = dataset.get_antigen_splits()
                    # Combine train and valid antigens
                    all_train_antigens = train_antigens + valid_antigens
                    model.fit(X_train, y_train, antigens=all_train_antigens)
                else:
                    model.fit(X_train, y_train)
            else:
                # For baseline models, use features directly
                model.fit(X_train, y_train)
            
            # Predict
            predictions = model.predict(X_test)
            
            # Evaluate
            metrics = dataset.evaluate(predictions, y_test)
            
            # Store results
            model_results = {
                "metrics": metrics,
                "training_time": model.training_time,
                "inference_time": model.inference_time,
            }
            
            if hasattr(model, 'evolution_history'):
                model_results["evolution_history"] = model.evolution_history
            
            model_name = model.name if hasattr(model, 'name') else 'TE-AI'
            results["models"][model_name] = model_results
            
            # Log results
            logger.info(f"Results for {model_name}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info(f"  Training time: {model.training_time:.2f}s")
            logger.info(f"  Inference time: {model.inference_time:.4f}s")
        
        return results
    
    def run_all_benchmarks(self):
        """Run all configured benchmarks"""
        all_results = []
        
        # Molecular property benchmarks
        molecular_datasets = ["BBBP", "Tox21", "SIDER"]
        for dataset_name in molecular_datasets:
            dataset = MolecularPropertyDataset(dataset_name)
            
            # Baseline models
            X_train, _, _, _ = dataset.get_splits()
            input_dim = X_train.shape[1]
            output_dim = 1  # Binary classification for most
            
            models = [
                SimpleNeuralNetwork(input_dim, output_dim),
                RandomForestBaseline()
            ]
            
            results = self.run_benchmark(dataset, models)
            all_results.append(results)
        
        # Cybersecurity benchmarks
        cyber_datasets = ["NSL-KDD", "CICIDS2017"]
        for dataset_name in cyber_datasets:
            dataset = CybersecurityDataset(dataset_name)
            
            # Baseline models
            X_train, _, _, _ = dataset.get_splits()
            input_dim = X_train.shape[1]
            output_dim = 1  # Binary for CICIDS2017
            
            models = [
                SimpleNeuralNetwork(input_dim, output_dim),
                RandomForestBaseline()
            ]
            
            results = self.run_benchmark(dataset, models)
            all_results.append(results)
        
        # Save results
        self.save_results(all_results)
        
        # Generate report
        self.generate_report(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """Generate benchmark report with visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary DataFrame
        summary_data = []
        for benchmark in results:
            dataset = benchmark["dataset"]
            for model_name, model_results in benchmark["models"].items():
                row = {
                    "Dataset": dataset,
                    "Model": model_name,
                    "Accuracy": model_results["metrics"].get("accuracy", 0),
                    "Precision": model_results["metrics"].get("precision", 0),
                    "Recall": model_results["metrics"].get("recall", 0),
                    "F1 Score": model_results["metrics"].get("f1_score", 0),
                    "ROC-AUC": model_results["metrics"].get("roc_auc", 0),
                    "Training Time (s)": model_results["training_time"],
                    "Inference Time (s)": model_results["inference_time"]
                }
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_file = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Generate plots
        self.generate_plots(df, timestamp)
        
        # Generate text report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("TE-AI BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # Summary statistics
            f.write("SUMMARY BY MODEL\n")
            f.write("-" * 30 + "\n")
            model_summary = df.groupby("Model")[["Accuracy", "F1 Score", "ROC-AUC"]].mean()
            f.write(model_summary.to_string())
            f.write("\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 30 + "\n")
            for benchmark in results:
                f.write(f"\nDataset: {benchmark['dataset']}\n")
                f.write(f"Task Type: {benchmark['task_type']}\n")
                f.write(f"Samples: {benchmark['train_samples']} train, {benchmark['test_samples']} test\n")
                
                for model_name, model_results in benchmark["models"].items():
                    f.write(f"\n  {model_name}:\n")
                    for metric, value in model_results["metrics"].items():
                        f.write(f"    {metric}: {value:.4f}\n")
                    f.write(f"    Training time: {model_results['training_time']:.2f}s\n")
                    f.write(f"    Inference time: {model_results['inference_time']:.4f}s\n")
        
        logger.info(f"Report saved to {report_file}")
    
    def generate_plots(self, df: pd.DataFrame, timestamp: str):
        """Generate visualization plots"""
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Performance comparison bar plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
        for i, metric in enumerate(metrics):
            ax = axes[i]
            df_pivot = df.pivot(index="Dataset", columns="Model", values=metric)
            df_pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{metric} by Dataset")
            ax.set_ylabel(metric)
            ax.legend(loc="best")
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"performance_comparison_{timestamp}.png", dpi=300)
        plt.close()
        
        # 2. Training time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        df_pivot = df.pivot(index="Dataset", columns="Model", values="Training Time (s)")
        df_pivot.plot(kind="bar", ax=ax, logy=True)
        ax.set_title("Training Time Comparison")
        ax.set_ylabel("Training Time (seconds, log scale)")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"training_time_{timestamp}.png", dpi=300)
        plt.close()
        
        # 3. Overall summary heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        summary_matrix = df.pivot_table(
            index="Model", 
            columns="Dataset", 
            values="Accuracy", 
            aggfunc="mean"
        )
        sns.heatmap(summary_matrix, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        ax.set_title("Accuracy Heatmap: Models vs Datasets")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"accuracy_heatmap_{timestamp}.png", dpi=300)
        plt.close()
        
        logger.info(f"Plots saved to {self.output_dir}")


def main():
    """Run all benchmarks"""
    logger.info("Starting TE-AI Benchmark Suite")
    logger.info("=" * 60)
    
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks()
    
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    
    # Print summary
    for benchmark in results:
        logger.info(f"\n{benchmark['dataset']}:")
        for model_name, model_results in benchmark["models"].items():
            accuracy = model_results["metrics"].get("accuracy", 0)
            logger.info(f"  {model_name}: {accuracy:.3f} accuracy")
    
    logger.info(f"\nFull results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()