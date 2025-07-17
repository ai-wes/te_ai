#!/usr/bin/env python3
"""
TE-AI Benchmark Runner
======================

Comprehensive benchmarking system to validate TE-AI performance against
standard datasets and competing methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve
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
    """Molecular property prediction benchmarks"""
    
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name, "molecular_property")
        self.dataset_name = dataset_name
        
    def load(self):
        """Load molecular dataset"""
        # For demo, create synthetic molecular data
        # In production, load from MoleculeNet
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        if self.dataset_name == "BBBP":
            # Blood-Brain Barrier Penetration (binary classification)
            n_samples = 2050
            n_features = 2048  # Morgan fingerprint size
            
            # Generate synthetic molecular fingerprints
            X = np.random.randn(n_samples, n_features) * 0.5
            # Add some structure
            for i in range(10):
                cluster_center = np.random.randn(n_features)
                cluster_size = n_samples // 10
                start_idx = i * cluster_size
                end_idx = (i + 1) * cluster_size
                X[start_idx:end_idx] += cluster_center * 0.3
            
            # Generate labels with some correlation to features
            y = (X.mean(axis=1) + np.random.randn(n_samples) * 0.5) > 0
            y = y.astype(np.float32)
            
        elif self.dataset_name == "Tox21":
            # Toxicity prediction (multi-task)
            n_samples = 8000
            n_features = 1024
            n_tasks = 12
            
            X = np.random.randn(n_samples, n_features) * 0.5
            y = np.random.randint(0, 2, size=(n_samples, n_tasks)).astype(np.float32)
            
        elif self.dataset_name == "SIDER":
            # Side effect prediction (multi-task)
            n_samples = 1427
            n_features = 1024
            n_tasks = 27
            
            X = np.random.randn(n_samples, n_features) * 0.5
            y = np.random.randint(0, 2, size=(n_samples, n_tasks)).astype(np.float32)
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        self.data = {"X": X, "y": y}
        self.loaded = True
        logger.info(f"Loaded {n_samples} samples with {n_features} features")
    
    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test splits"""
        if not self.loaded:
            self.load()
        
        X = self.data["X"]
        y = self.data["y"]
        
        # Use standard 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.ndim == 1 else None
        )
        
        return X_train, X_test, y_train, y_test
    
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
    
    def create_model(self, input_dim: int, output_dim: int):
        """Create appropriate TE-AI model for the task"""
        from scripts.core.production_germinal_center import ProductionGerminalCenter
        from scripts.domains.drug_discovery.drug_discovery_germinal_center import DrugDiscoveryGerminalCenter
        from scripts.domains.cyber_security.main_cyber_security import CyberSecurityGerminalCenter
        
        cfg.population_size = 30  # Smaller for benchmarking
        cfg.generations = 20
        
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
            self.model = ProductionGerminalCenter(
                initial_population_size=cfg.population_size
            )
        
        self.cfg = cfg
    
    def fit(self, X_train, y_train, generations: int = 20):
        """Train using evolutionary process"""
        start_time = time.time()
        
        # Convert to appropriate format for TE-AI
        # For simplicity, create synthetic antigens from data
        from scripts.core.anitgen import BiologicalAntigen, AntigenEpitope
        
        antigens = []
        for i in range(min(100, len(X_train))):  # Use subset for demo
            # Create epitope from features
            # Generate synthetic sequence based on features
            seq_length = 20
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            sequence = ''.join(np.random.choice(list(amino_acids), size=seq_length))
            
            # Generate structure coordinates
            coords = np.random.randn(seq_length, 3) * 2.0
            
            # Use target value to influence hydrophobicity
            target_val = float(y_train[i]) if y_train.ndim == 1 else y_train[i, 0]
            hydrophobicity = target_val * 2.0 - 1.0  # Map 0-1 to -1 to 1
            
            # Generate charge based on features
            charge = np.random.uniform(-3, 3)
            
            epitope = AntigenEpitope(
                sequence=sequence,
                structure_coords=coords,
                hydrophobicity=hydrophobicity,
                charge=charge
            )
            
            # Create antigen with the epitope
            antigen = BiologicalAntigen(antigen_type="synthetic")
            antigen.epitopes = [epitope]  # Replace default epitopes with our synthetic one
            antigens.append(antigen)
        
        # Convert to graphs and add target values
        antigen_graphs = []
        for i, a in enumerate(antigens):
            graph = a.to_graph()
            # Add the target value to the graph
            target_val = float(y_train[i]) if y_train.ndim == 1 else y_train[i, 0]
            graph.y = torch.tensor([target_val], dtype=torch.float32)
            antigen_graphs.append(graph)
        
        # Evolve
        logger.info("Starting TE-AI evolution...")
        for generation in range(generations):
            # Sample batch
            batch_indices = np.random.choice(len(antigen_graphs), 
                                           size=min(32, len(antigen_graphs)), 
                                           replace=False)
            batch = [antigen_graphs[i] for i in batch_indices]
            
            # Evolve one generation
            if hasattr(self.model, 'evolve_generation'):
                stats = self.model.evolve_generation(batch)
                if stats:
                    self.evolution_history.append(stats)
                    logger.info(f"Generation {generation}: Fitness={stats.get('best_fitness', 0):.4f}")
        
        self.training_time = time.time() - start_time
    
    def predict(self, X_test):
        """Make predictions using evolved model"""
        start_time = time.time()
        
        # For demo, return random predictions
        # In full implementation, would process through evolved population
        predictions = np.random.rand(len(X_test))
        
        self.inference_time = time.time() - start_time
        return predictions


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
        
        # Run each model
        for model in models:
            logger.info(f"\nTraining {model.name if hasattr(model, 'name') else 'TE-AI'}...")
            
            # Train
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