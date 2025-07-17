# Set matplotlib backend before import to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

import asyncio
import websockets
import json
import os
from threading import Thread
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from torch_geometric.utils import to_undirected, add_self_loops
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import copy
import uuid
import json
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
import os
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hashlib
from scripts.core.utils.telemetry import write_visualization_state, _write_enhanced_visualization_state
from scripts.config import cfg

# Removed circular import - StemGeneModule will be imported dynamically when needed
# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


from scripts.core.utils.detailed_logger import get_logger, trace

logger = get_logger()

# ============================================================================# ============================================================================
# Self-Modifying Neural Architecture with Complete Implementation
# ============================================================================

class ArchitectureModification:
    """Represents a structural modification to neural architecture"""
    
    def __init__(self, mod_type: str, target_module: str, parameters: Dict):
        self.mod_type = mod_type
        self.target_module = target_module
        self.parameters = parameters
        self.timestamp = datetime.now()
        self.success = False
        self.performance_delta = 0.0

class SelfModifyingArchitecture(nn.Module):
    """Neural architecture that can modify its own structure"""
    
    def __init__(self, base_dim: int = cfg.hidden_dim):
        super().__init__()
        self.base_dim = base_dim
        self.modification_history = []
        self.architecture_dna = self._generate_architecture_dna()
        
        # Meta-controller for architecture decisions
        self.meta_controller = nn.LSTM(
            input_size=base_dim * 3,  # Current state, gradient info, performance
            hidden_size=base_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Architecture modification networks
        self.mod_networks = nn.ModuleDict({
            'layer_addition': self._create_layer_addition_network(),
            'layer_removal': self._create_layer_removal_network(),
            'connection_rewiring': self._create_rewiring_network(),
            'dimension_resizing': self._create_resizing_network(),
            'activation_change': self._create_activation_network()
        })
        
        # Dynamic module registry
        self.dynamic_modules = OrderedDict()
        self.module_connections = {}  # Adjacency list of connections
        
        # Initialize with base architecture
        self._initialize_base_architecture()
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.gradient_history = deque(maxlen=50)
        
    
    def _generate_architecture_dna(self) -> str:
        """Generate unique DNA string representing architecture"""
        # DNA encodes layer types, connections, and parameters
        dna_segments = [
            'START',
            f'DIM:{self.base_dim}',
            f'LAYERS:3',
            'CONN:SEQUENTIAL',
            'ACT:RELU',
            'NORM:LAYER',
            'END'
        ]
        return '-'.join(dna_segments)
    
    
    def _create_layer_addition_network(self) -> nn.Module:
        """Network that generates new layer parameters"""
        return nn.Sequential(
            nn.Linear(self.base_dim, self.base_dim * 2),
            nn.ReLU(),
            nn.Linear(self.base_dim * 2, self.base_dim * self.base_dim + self.base_dim),
            # Output is flattened weights + bias for new layer
        )
    
    
    def _create_layer_removal_network(self) -> nn.Module:
        """Network that decides which layers to remove"""
        return nn.Sequential(
            nn.Linear(self.base_dim, self.base_dim),
            nn.ReLU(),
            nn.Linear(self.base_dim, 1),
            nn.Sigmoid()  # Removal probability
        )
    
    
    def _create_rewiring_network(self) -> nn.Module:
        """Network that modifies connections between layers"""
        return nn.Sequential(
            nn.Linear(self.base_dim * 2, self.base_dim),
            nn.ReLU(),
            nn.Linear(self.base_dim, self.base_dim * self.base_dim),
            nn.Tanh()  # Connection weight matrix
        )
    
    
    def _create_resizing_network(self) -> nn.Module:
        """Network that changes layer dimensions"""
        return nn.Sequential(
            nn.Linear(self.base_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Min size, max size, expand factor, contract factor
        )
    
    
    def _create_activation_network(self) -> nn.Module:
        """Network that selects activation functions"""
        return nn.Sequential(
            nn.Linear(self.base_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # ReLU, Tanh, Sigmoid, ELU, GELU, Swish
        )
    
    
    def _initialize_base_architecture(self):
        """Create initial architecture"""
        # Base transformation layers
        for i in range(3):
            layer_name = f'transform_{i}'
            self.dynamic_modules[layer_name] = nn.Sequential(
                nn.Linear(self.base_dim, self.base_dim),
                nn.LayerNorm(self.base_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Sequential connections
            if i > 0:
                prev_layer = f'transform_{i-1}'
                self.module_connections[prev_layer] = [layer_name]
            
        # Output projection
        self.dynamic_modules['output'] = nn.Linear(self.base_dim, self.base_dim)
        self.module_connections['transform_2'] = ['output']
    
    
    def analyze_performance(self, loss_history: List[float], 
                          gradient_norms: List[float]) -> Dict[str, float]:
        """Analyze recent performance trends"""
        if len(loss_history) < 5:
            return {'trend': 0.0, 'stability': 1.0, 'gradient_health': 1.0}
        
        # Loss trend (negative is good)
        recent_losses = loss_history[-10:]
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # Stability (low variance is good)
        loss_variance = np.var(recent_losses)
        stability = 1.0 / (1.0 + loss_variance)
        
        # Gradient health
        if gradient_norms:
            recent_grads = gradient_norms[-10:]
            grad_mean = np.mean(recent_grads)
            grad_variance = np.var(recent_grads)
            
            # Healthy gradients: not too small, not too large, consistent
            if grad_mean < 1e-6:
                gradient_health = 0.1  # Vanishing gradients
            elif grad_mean > 10:
                gradient_health = 0.2  # Exploding gradients
            else:
                gradient_health = 1.0 / (1.0 + grad_variance)
        else:
            gradient_health = 0.5
        
        return {
            'trend': loss_trend,
            'stability': stability,
            'gradient_health': gradient_health
        }
        
        
    
    def decide_modification(self, performance_metrics: Dict[str, float],
                        current_state: torch.Tensor) -> ArchitectureModification:
        """Decide what architectural modification to make"""
        # Prepare input for meta-controller
        
        # --- FIX: Explicitly set dtype to match the model's expected input type ---
        # The original code implicitly created a float64 tensor.
        # We now create a float32 tensor (or whatever dtype current_state is)
        # to match the rest of the model's inputs under AMP.
        perf_tensor = torch.tensor([
            performance_metrics['trend'],
            performance_metrics['stability'],
            performance_metrics['gradient_health']
        ], dtype=current_state.dtype, device=current_state.device)
        
        # Get gradient statistics
        # --- FIX: Ensure the returned tensor is also cast to the correct dtype ---
        grad_stats = self._compute_gradient_statistics().to(dtype=current_state.dtype, device=current_state.device)
        
        # Combine inputs
        meta_input = torch.cat([
            current_state.flatten(),
            perf_tensor,
            grad_stats
        ]).unsqueeze(0).unsqueeze(0)
        
        # Pad to expected size
        if meta_input.shape[-1] < self.base_dim * 3:
            padding = torch.zeros(1, 1, self.base_dim * 3 - meta_input.shape[-1],
                                dtype=meta_input.dtype, device=meta_input.device) # Ensure padding matches dtype
            meta_input = torch.cat([meta_input, padding], dim=-1)
        
        # Get modification decision
        output, _ = self.meta_controller(meta_input) # This call will now succeed
        decision_logits = output[0, -1, :5]  # 5 modification types
        
        # Temperature-based sampling for exploration
        temperature = 1.0 if performance_metrics['gradient_health'] > 0.7 else 2.0
        probs = F.softmax(decision_logits / temperature, dim=0)
        
        mod_types = ['add_layer', 'remove_layer', 'rewire', 'resize', 'change_activation']
        mod_idx = torch.multinomial(probs, 1).item()
        mod_type = mod_types[mod_idx]
        
        # Generate modification parameters
        if mod_type == 'add_layer':
            params = self._generate_layer_addition_params(current_state)
        elif mod_type == 'remove_layer':
            params = self._generate_layer_removal_params()
        elif mod_type == 'rewire':
            params = self._generate_rewiring_params(current_state)
        elif mod_type == 'resize':
            params = self._generate_resizing_params(current_state)
        else:  # change_activation
            params = self._generate_activation_params(current_state)
        
        return ArchitectureModification(mod_type, params.get('target'), params)


    
    
    def _compute_gradient_statistics(self) -> torch.Tensor:
        """Compute gradient flow statistics"""
        grad_stats = []
        
        for name, module in self.dynamic_modules.items():
            grad_norm = 0.0
            param_count = 0
            
            for param in module.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    param_count += 1
            
            avg_grad = grad_norm / max(param_count, 1)
            grad_stats.append(avg_grad)
        
        # Pad or truncate to fixed size
        while len(grad_stats) < 10:
            grad_stats.append(0.0)
        grad_stats = grad_stats[:10]
        
        return torch.tensor(grad_stats).to(cfg.device)
    
    
    def _generate_layer_addition_params(self, state: torch.Tensor) -> Dict:
        """Generate parameters for adding a new layer"""
        # Generate layer parameters
        param_vector = self.mod_networks['layer_addition'](state.flatten())
        
        # Split into weights and bias
        weight_size = self.base_dim * self.base_dim
        weights = param_vector[:weight_size].reshape(self.base_dim, self.base_dim)
        bias = param_vector[weight_size:]
        
        # Decide where to insert
        positions = list(self.dynamic_modules.keys())
        insert_after = random.choice(positions[:-1])  # Not after output
        
        return {
            'target': insert_after,
            'weights': weights,
            'bias': bias,
            'layer_type': 'linear',
            'activation': 'relu'
        }
    
    
    def _generate_layer_removal_params(self) -> Dict:
        """Decide which layer to remove"""
        removal_probs = {}
        
        for name, module in self.dynamic_modules.items():
            if name != 'output' and len(self.dynamic_modules) > 2:
                # Don't remove output or if too few layers
                state_summary = torch.randn(self.base_dim).to(cfg.device)
                prob = self.mod_networks['layer_removal'](state_summary)
                removal_probs[name] = prob.item()
        
        if removal_probs:
            # Remove layer with highest removal probability
            target = max(removal_probs, key=removal_probs.get)
        else:
            target = None
        
        return {'target': target}
    
    
    def _generate_rewiring_params(self, state: torch.Tensor) -> Dict:
        """Generate new connection patterns"""
        # Pick two layers to rewire between
        layer_names = list(self.dynamic_modules.keys())
        if len(layer_names) < 2:
            return {'target': None}
        
        source = random.choice(layer_names[:-1])
        target = random.choice(layer_names[1:])
        
        # Generate connection weights
        combined_state = torch.cat([state.flatten(), torch.randn_like(state.flatten())])
        if combined_state.shape[0] < self.base_dim * 2:
            padding = torch.zeros(self.base_dim * 2 - combined_state.shape[0]).to(combined_state.device)
            combined_state = torch.cat([combined_state, padding])
        
        connection_weights = self.mod_networks['connection_rewiring'](
            combined_state[:self.base_dim * 2]
        )
        connection_matrix = connection_weights.reshape(self.base_dim, self.base_dim)
        
        return {
            'target': source,
            'source': source,
            'destination': target,
            'connection_matrix': connection_matrix,
            'connection_type': 'residual' if random.random() > 0.5 else 'sequential'
        }
    
    
    def _generate_resizing_params(self, state: torch.Tensor) -> Dict:
        """Generate layer resizing parameters"""
        resize_params = self.mod_networks['dimension_resizing'](state.flatten())
        
        min_size = int(torch.clamp(resize_params[0] * 32 + 16, 16, 512).item())
        max_size = int(torch.clamp(resize_params[1] * 32 + min_size, min_size + 16, 1024).item())
        expand_factor = torch.sigmoid(resize_params[2]).item() + 0.5
        contract_factor = torch.sigmoid(resize_params[3]).item() * 0.5 + 0.5
        
        # Choose layer to resize
        target = random.choice(list(self.dynamic_modules.keys())[:-1])
        
        return {
            'target': target,
            'min_size': min_size,
            'max_size': max_size,
            'expand_factor': expand_factor,
            'contract_factor': contract_factor
        }
    
    
    def _generate_activation_params(self, state: torch.Tensor) -> Dict:
        """Select new activation function"""
        activation_logits = self.mod_networks['activation_change'](state.flatten())
        activation_probs = F.softmax(activation_logits, dim=0)
        
        activations = ['relu', 'tanh', 'sigmoid', 'elu', 'gelu', 'swish']
        act_idx = torch.multinomial(activation_probs, 1).item()
        
        target = random.choice(list(self.dynamic_modules.keys())[:-1])
        
        return {
            'target': target,
            'activation': activations[act_idx]
        }
    
    
    def apply_modification(self, modification: ArchitectureModification, cell_id: str = None) -> bool:
        """Apply architectural modification and emit telemetry."""
        try:
            # We pass the cell_id for logging/visualization purposes
            success = False
            if modification.mod_type == 'add_layer':
                success = self._add_layer(modification.parameters)
            elif modification.mod_type == 'remove_layer':
                success = self._remove_layer(modification.parameters)
            elif modification.mod_type == 'rewire':
                success = self._rewire_connections(modification.parameters)
            elif modification.mod_type == 'resize':
                success = self._resize_layer(modification.parameters)
            elif modification.mod_type == 'change_activation':
                success = self._change_activation(modification.parameters)
            
            modification.success = success
            self.modification_history.append(modification)
            
            if success:
                self.architecture_dna = self._generate_architecture_dna()
                # --- TELEMETRY EMITTER ---
                # Broadcast the new state after a successful change
                if cell_id:
                    write_visualization_state(cell_id, self)
                    _write_enhanced_visualization_state(cell_id, self)
                # --- END TELEMETRY ---
            
            return success
                
        except Exception as e:
            print(f"Modification failed: {e}")
            print(f"  Debug: mod_type={modification.mod_type}, target={modification.target_module}")
            print(f"  Debug: dynamic_modules keys={list(self.dynamic_modules.keys())}")
            if modification.target_module in self.dynamic_modules:
                module = self.dynamic_modules[modification.target_module]
                print(f"  Debug: target module type={type(module)}")
                print(f"  Debug: is Sequential? {isinstance(module, nn.Sequential)}")
            import traceback
            traceback.print_exc()
            modification.success = False
            self.modification_history.append(modification)
            return False
            

    
    
    def _add_layer(self, params: Dict) -> bool:
        """Add new layer to architecture"""
        if params['target'] is None:
            return False
        
        # Create new layer
        layer_name = f"transform_{len(self.dynamic_modules)}"
        
        new_layer = nn.Sequential(
            nn.Linear(self.base_dim, self.base_dim),
            nn.LayerNorm(self.base_dim),
            self._get_activation(params.get('activation', 'relu')),
            nn.Dropout(0.1)
        )
        
        # Initialize with generated parameters
        with torch.no_grad():
            new_layer[0].weight.data = params['weights']
            new_layer[0].bias.data = params['bias']
        
        # Insert into architecture
        self.dynamic_modules[layer_name] = new_layer
        
        # Update connections
        insert_after = params['target']
        if insert_after in self.module_connections:
            next_layers = self.module_connections[insert_after]
            self.module_connections[insert_after] = [layer_name]
            self.module_connections[layer_name] = next_layers
        else:
            self.module_connections[layer_name] = ['output']
        
        print(f"Added layer {layer_name} after {insert_after}")
        return True
    
    
    def _remove_layer(self, params: Dict) -> bool:
        """Remove layer from architecture"""
        target = params.get('target')
        if target is None or target not in self.dynamic_modules:
            return False
        
        if len(self.dynamic_modules) <= 2:  # Keep minimum architecture
            return False
        
        # Find connections to update
        prev_layers = [k for k, v in self.module_connections.items() if target in v]
        next_layers = self.module_connections.get(target, [])
        
        # Remove layer
        del self.dynamic_modules[target]
        del self.module_connections[target]
        
        # Update connections
        for prev in prev_layers:
            self.module_connections[prev] = next_layers
        
        print(f"Removed layer {target}")
        return True
    
    
    def _rewire_connections(self, params: Dict) -> bool:
        """Modify connections between layers"""
        source = params.get('source')
        destination = params.get('destination')
        
        if not source or not destination:
            return False
        
        connection_type = params.get('connection_type', 'sequential')
        
        if connection_type == 'residual':
            # Add residual connection
            if source not in self.module_connections:
                self.module_connections[source] = []
            
            if destination not in self.module_connections[source]:
                self.module_connections[source].append(destination)
                print(f"Added residual connection {source} -> {destination}")
        else:
            # Rewire sequential connection
            self.module_connections[source] = [destination]
            # Reduced debug spam
            if random.random() < 0.1:  # Only print 10% of rewiring events
                print(f"Rewired {source} -> {destination}")
        
        return True
    
    
    def _resize_layer(self, params: Dict) -> bool:
        """Resize layer dimensions"""
        target = params.get('target')
        if target not in self.dynamic_modules:
            return False
        
        module = self.dynamic_modules[target]
        
        # Handle both Sequential and Linear modules
        if isinstance(module, nn.Sequential):
            # Find the Linear layer in the Sequential
            linear_layer = None
            linear_idx = None
            for i, layer in enumerate(module):
                if isinstance(layer, nn.Linear):
                    linear_layer = layer
                    linear_idx = i
                    break
            if linear_layer is None:
                return False
            old_in = linear_layer.in_features
            old_out = linear_layer.out_features
        elif isinstance(module, nn.Linear):
            linear_layer = module
            linear_idx = None
            old_in = module.in_features
            old_out = module.out_features
        else:
            return False
        
        # Determine new size
        if old_out < params['min_size']:
            new_size = int(old_out * params['expand_factor'])
        elif old_out > params['max_size']:
            new_size = int(old_out * params['contract_factor'])
        else:
            return False  # No change needed
        
        new_size = max(params['min_size'], min(params['max_size'], new_size))
        
        # Create new layer with different size
        new_layer = nn.Linear(old_in, new_size).to(linear_layer.weight.device)
        
        # Initialize intelligently
        with torch.no_grad():
            if new_size > old_out:
                # Expanding: copy old weights and add noise to new
                new_layer.weight.data[:old_out] = linear_layer.weight.data
                new_layer.weight.data[old_out:] = torch.randn(
                    new_size - old_out, old_in
                ).to(new_layer.weight.device) * 0.01
                new_layer.bias.data[:old_out] = linear_layer.bias.data
            else:
                # Contracting: use PCA or importance sampling
                weight_importance = linear_layer.weight.data.norm(dim=1)
                _, indices = torch.topk(weight_importance, new_size)
                new_layer.weight.data = linear_layer.weight.data[indices]
                new_layer.bias.data = linear_layer.bias.data[indices]
        
        # Replace the module
        if isinstance(module, nn.Sequential):
            module[linear_idx] = new_layer
        else:
            self.dynamic_modules[target] = new_layer
        
        # Reduced debug spam
        if random.random() < 0.1:  # Only print 10% of resize events
            print(f"Resized {target}: {old_out} -> {new_size}")
        return True
    
    
    def _change_activation(self, params: Dict) -> bool:
        """Change activation function"""
        target = params.get('target')
        activation = params.get('activation', 'relu')
        
        if target not in self.dynamic_modules:
            return False
        
        module = self.dynamic_modules[target]
        
        # Check if this is a Sequential module
        if isinstance(module, nn.Sequential):
            # Find and replace activation
            for i, layer in enumerate(module):
                if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.GELU)):
                    module[i] = self._get_activation(activation)
                    # Reduced debug spam
                    if random.random() < 0.1:  # Only print 10% of activation changes
                        print(f"Changed activation in {target} to {activation}")
                    return True
        
        # If it's just a Linear layer (like 'output'), we can't change activation
        return False
    
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(name, nn.ReLU())
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dynamic architecture"""
        # Start from first layer
        current = x
        processed_layers = set()
        
        # Find input layer (no incoming connections)
        input_layers = set(self.dynamic_modules.keys()) - \
                      set(sum(self.module_connections.values(), []))
        
        if not input_layers:
            input_layers = [list(self.dynamic_modules.keys())[0]]
        
        # Process through architecture
        layer_outputs = {}
        to_process = list(input_layers)
        
        while to_process:
            layer_name = to_process.pop(0)
            
            if layer_name in processed_layers:
                continue
            
            # Get inputs for this layer
            incoming = [k for k, v in self.module_connections.items() if layer_name in v]
            
            if incoming:
                # Combine inputs
                inputs = []
                for in_layer in incoming:
                    if in_layer in layer_outputs:
                        inputs.append(layer_outputs[in_layer])
                
                if len(inputs) == 0:
                    continue  # Dependencies not ready
                elif len(inputs) == 1:
                    layer_input = inputs[0]
                else:
                    # Multiple inputs - combine
                    layer_input = sum(inputs) / len(inputs)
            else:
                layer_input = current
            
            # Process through layer
            layer_output = self.dynamic_modules[layer_name](layer_input)
            layer_outputs[layer_name] = layer_output
            processed_layers.add(layer_name)
            
            # Add next layers to process
            if layer_name in self.module_connections:
                to_process.extend(self.module_connections[layer_name])
        
        # Return output layer result
        if 'output' in layer_outputs:
            return layer_outputs['output']
        else:
            # Return last processed layer
            return layer_output
