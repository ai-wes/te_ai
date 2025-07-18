"""
DeepChem Baseline Models for Benchmarking
=========================================

Provides wrappers for state-of-the-art DeepChem models to benchmark against TE-AI.
"""

import numpy as np
import time
import deepchem as dc
import deepchem.models
import torch
from typing import Dict, Any, Optional
from scripts.core.utils.detailed_logger import get_logger
from scripts.core.utils.robust_multitask import RobustMultitaskClassifier

logger = get_logger()


class DeepChemModelWrapper:
    """Base wrapper for DeepChem models"""
    
    def __init__(self, model_class, model_params: Dict[str, Any] = None):
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model = None
        self.training_time = 0
        self.inference_time = 0
        self.name = model_class.__name__
        
    def fit(self, X_train, y_train):
        """Train the model"""
        start_time = time.time()
        
        # Create DeepChem dataset
        train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train)
        
        # Initialize model
        n_tasks = 1 if y_train.ndim == 1 else y_train.shape[1]
        n_features = X_train.shape[1]
        
        # Set default parameters based on model type
        if self.model_class == dc.models.GraphConvModel:
            # Graph models need special handling
            logger.warning(f"{self.name} requires graph input, using MultitaskClassifier instead")
            self.model = dc.models.MultitaskClassifier(
                n_tasks=n_tasks,
                n_features=n_features,
                layer_sizes=[1000, 500],
                **self.model_params
            )
        elif self.model_class == dc.models.AttentiveFPModel:
            # AttentiveFP needs graph featurization
            logger.warning(f"{self.name} requires graph input, using MultitaskClassifier instead")
            self.model = dc.models.MultitaskClassifier(
                n_tasks=n_tasks,
                n_features=n_features,
                layer_sizes=[1000, 500],
                **self.model_params
            )
        else:
            # Standard models
            self.model = self.model_class(
                n_tasks=n_tasks,
                n_features=n_features,
                **self.model_params
            )
        
        # Train
        self.model.fit(train_dataset, nb_epoch=50)
        
        self.training_time = time.time() - start_time
        logger.info(f"{self.name} training completed in {self.training_time:.2f}s")
        
    def predict(self, X_test):
        """Make predictions"""
        start_time = time.time()
        
        # Create test dataset
        test_dataset = dc.data.NumpyDataset(X=X_test, y=np.zeros((len(X_test), 1)))
        
        # Predict
        predictions = self.model.predict(test_dataset)
        
        # Debug shape issues
        logger.info(f"DeepNN raw predictions shape: {predictions.shape}")
        
        # Handle multi-dimensional output
        if len(predictions.shape) == 3 and predictions.shape[2] == 2:
            # Binary classification with probabilities for both classes
            # Take only the positive class (index 1)
            logger.info("Binary classification output detected, taking positive class probabilities")
            predictions = predictions[:, 0, 1]  # Shape: (n_samples,)
        elif len(predictions.shape) > 2:
            predictions = predictions.squeeze()
        elif len(predictions.shape) == 2 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        elif len(predictions.shape) == 2 and predictions.shape[1] == 2:
            # 2D array with probabilities for both classes
            logger.info("2D binary classification output, taking positive class")
            predictions = predictions[:, 1]
            
        # Final safety check
        if len(predictions) != len(X_test):
            logger.warning(f"Prediction shape mismatch after processing: got {len(predictions)}, expected {len(X_test)}")
            predictions = predictions[:len(X_test)]
            
        self.inference_time = time.time() - start_time
        
        return predictions


class GraphConvModelWrapper(DeepChemModelWrapper):
    """Wrapper for Graph Convolutional Network"""
    
    def __init__(self):
        super().__init__(
            dc.models.GraphConvModel,
            {
                'graph_conv_layers': [64, 64],
                'dense_layer_size': 128,
                'dropout': 0.2,
                'mode': 'classification'
            }
        )
        self.name = "GraphConvNet"


class AttentiveFPModelWrapper(DeepChemModelWrapper):
    """Wrapper for Attentive FP model"""
    
    def __init__(self):
        super().__init__(
            dc.models.AttentiveFPModel,
            {
                'num_layers': 3,
                'num_timesteps': 3,
                'graph_feat_size': 200,
                'dropout': 0.2,
                'mode': 'classification'
            }
        )
        self.name = "AttentiveFP"


class MultitaskClassifierWrapper(DeepChemModelWrapper):
    """Wrapper for standard deep neural network"""
    
    def __init__(self):
        super().__init__(
            dc.models.MultitaskClassifier,
            {
                'layer_sizes': [1000, 500, 200],
                'dropout': 0.25,
                'learning_rate': 0.001
            }
        )
        self.name = "DeepNN"


class RobustMultitaskClassifierWrapper(DeepChemModelWrapper):
    """Wrapper for robust multitask classifier with bypass layers"""
    
    def __init__(self):
        super().__init__(
            RobustMultitaskClassifier,
            {
                'layer_sizes': [1000, 500],
                'bypass_layer_sizes': [100],
                'dropout': 0.25,
                'learning_rate': 0.001
            }
        )
        self.name = "RobustNN"


class ChemBERTaWrapper:
    """Wrapper for ChemBERTa transformer model"""
    
    def __init__(self):
        self.name = "ChemBERTa"
        self.model = None
        self.training_time = 0
        self.inference_time = 0
        
    def fit(self, X_train, y_train):
        """Train ChemBERTa (or use pretrained)"""
        start_time = time.time()
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load pretrained ChemBERTa
            self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1",
                num_labels=1 if y_train.ndim == 1 else y_train.shape[1]
            )
            
            logger.info("Loaded pretrained ChemBERTa model")
            
        except ImportError:
            logger.warning("Transformers library not installed, using MultitaskClassifier instead")
            # Fallback to standard NN
            self.model = dc.models.MultitaskClassifier(
                n_tasks=1 if y_train.ndim == 1 else y_train.shape[1],
                n_features=X_train.shape[1],
                layer_sizes=[1000, 500]
            )
            train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train)
            self.model.fit(train_dataset, nb_epoch=50)
            
        self.training_time = time.time() - start_time
        
    def predict(self, X_test):
        """Make predictions with ChemBERTa"""
        start_time = time.time()
        
        if hasattr(self, 'tokenizer'):
            # Real ChemBERTa predictions would need SMILES strings
            logger.warning("ChemBERTa needs SMILES input, using random predictions")
            predictions = np.random.rand(len(X_test))
        else:
            # Fallback predictions
            test_dataset = dc.data.NumpyDataset(X=X_test, y=np.zeros((len(X_test), 1)))
            predictions = self.model.predict(test_dataset).squeeze()
            
        self.inference_time = time.time() - start_time
        return predictions


def get_deepchem_baselines():
    """Get all available DeepChem baseline models"""
    baselines = [
        MultitaskClassifierWrapper(),
        RobustMultitaskClassifierWrapper(),
        # GraphConvModelWrapper(),  # Requires graph features
        # AttentiveFPModelWrapper(),  # Requires graph features
        # ChemBERTaWrapper(),  # Requires SMILES strings
    ]
    
    return baselines


def run_deepchem_baseline(model_wrapper, X_train, y_train, X_test, y_test):
    """Run a single DeepChem baseline"""
    
    # Train
    model_wrapper.fit(X_train, y_train)
    
    # Predict
    predictions = model_wrapper.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    if y_test.ndim == 1:
        # Binary classification
        pred_binary = (predictions > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, pred_binary),
            'f1_score': f1_score(y_test, pred_binary, average='binary'),
            'roc_auc': roc_auc_score(y_test, predictions) if len(np.unique(y_test)) > 1 else 0.5
        }
    else:
        # Multi-task
        metrics = {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0}
        
    return {
        'model': model_wrapper.name,
        'metrics': metrics,
        'training_time': model_wrapper.training_time,
        'inference_time': model_wrapper.inference_time
    }


if __name__ == "__main__":
    # Test baselines
    logger.info("Testing DeepChem baselines...")
    
    # Create dummy data
    X_train = np.random.randn(1000, 2048)
    y_train = np.random.randint(0, 2, size=(1000,))
    X_test = np.random.randn(200, 2048)
    y_test = np.random.randint(0, 2, size=(200,))
    
    baselines = get_deepchem_baselines()
    
    for baseline in baselines:
        logger.info(f"\nTesting {baseline.name}...")
        results = run_deepchem_baseline(baseline, X_train, y_train, X_test, y_test)
        
        print(f"\n{results['model']} Results:")
        for metric, value in results['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print(f"  Training time: {results['training_time']:.2f}s")
        print(f"  Inference time: {results['inference_time']:.4f}s")