import torch
import numpy as np
import random
import torch.nn as nn
from scripts.core.utils.detailed_logger import get_logger, trace

logger = get_logger()
import threading

# Helper class for terminal colors
class TermColors:
    """Utility class for terminal colors and styles."""
    # Basic Colors
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright Colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\032[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
# ============================================================================
# Production Configuration
# ============================================================================
from scripts.config import cfg

# ============================================================================
# Telemetry System for Live Visualization
# ============================================================================

# A lock to prevent race conditions when writing to the state file
state_lock = threading.Lock()

# Global reference to current germinal center for visualization
_current_germinal_center = None

# Global run ID for unique file naming
_run_id = None



import json
import time
import threading
import os
from datetime import datetime
from collections import defaultdict

# Global variables for visualization
_current_germinal_center = None
_run_id = None
state_lock = threading.Lock()








def write_visualization_state(cell_id, architecture_modifier):
    """Writes the current architectural state including full population data to JSON file."""
    # Write visualization state regardless of mode
    
    try:
        _write_population_visualization_state(cell_id, architecture_modifier)
        _write_single_cell_architecture_state(cell_id, architecture_modifier)
        logger.info("write_visualization_state completed successfully")
    except Exception as e:
        logger.warning(f"write_visualization_state failed: {e}")
        # Don't crash the training, just skip visualization
        return











def set_germinal_center(gc):
    """Set the global germinal center reference"""
    global _current_germinal_center
    _current_germinal_center = gc



def _write_single_cell_architecture_state(cell_id, architecture_modifier):
    """Writes detailed architecture state for a single cell to a unique file."""
    print("Writing enhanced visualization state")
    # Capture current architecture state
    architecture_state = {
        'modules': {},
        'connections': dict(architecture_modifier.module_connections),
        'modification_history': []
    }
    
    # Analyze each module in detail
    for name, module in architecture_modifier.dynamic_modules.items():
        module_info = {
            'name': name,
            'type': 'sequential' if isinstance(module, nn.Sequential) else 'linear',
            'layers': [],
            'position': _calculate_module_position(name, architecture_modifier),
            'size': 0,
            'activation': None,
            'color': '#4A90E2'  # Default blue
        }
        
        if isinstance(module, nn.Sequential):
            # Analyze Sequential module structure
            for i, layer in enumerate(module):
                layer_info = {
                    'index': i,
                    'type': type(layer).__name__,
                    'params': {}
                }
                
                if isinstance(layer, nn.Linear):
                    layer_info['params'] = {
                        'in_features': layer.in_features,
                        'out_features': layer.out_features
                    }
                    module_info['size'] = layer.out_features
                elif isinstance(layer, nn.LayerNorm):
                    layer_info['params'] = {
                        'normalized_shape': layer.normalized_shape
                    }
                elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.GELU, nn.SiLU)):
                    module_info['activation'] = type(layer).__name__.lower().replace('silu', 'swish')
                    module_info['color'] = _get_activation_color(type(layer).__name__)
                elif isinstance(layer, nn.Dropout):
                    layer_info['params'] = {
                        'p': layer.p
                    }
                
                module_info['layers'].append(layer_info)
        
        elif isinstance(module, nn.Linear):
            # Handle standalone Linear module (like output)
            module_info['size'] = module.out_features
            module_info['layers'] = [{
                'index': 0,
                'type': 'Linear',
                'params': {
                    'in_features': module.in_features,
                    'out_features': module.out_features
                }
            }]
        
        architecture_state['modules'][name] = module_info
    
    # Process modification history
    for i, mod in enumerate(architecture_modifier.modification_history[-50:]):  # Last 50 mods
        mod_info = {
            'index': i,
            'type': mod.mod_type,
            'target': mod.target_module,
            'success': mod.success,
            'timestamp': time.time() - (50 - i) * 0.5,  # Simulate timing
            'parameters': {}
        }
        
        # Extract specific parameters based on modification type
        if hasattr(mod, 'parameters'):
            params = mod.parameters
            
            if mod.mod_type == 'add_layer':
                mod_info['parameters'] = {
                    'insert_after': params.get('target'),
                    'activation': params.get('activation', 'relu'),
                    'layer_name': f"transform_{len(architecture_modifier.dynamic_modules) - 1}"
                }
            
            elif mod.mod_type == 'remove_layer':
                mod_info['parameters'] = {
                    'removed_layer': params.get('target')
                }
            
            elif mod.mod_type == 'rewire':
                mod_info['parameters'] = {
                    'source': params.get('source'),
                    'destination': params.get('destination'),
                    'connection_type': params.get('connection_type', 'sequential')
                }
            
            elif mod.mod_type == 'resize':
                # Find the actual new size from the current state
                target = params.get('target')
                new_size = architecture_modifier.base_dim  # Default
                if target in architecture_modifier.dynamic_modules:
                    module = architecture_modifier.dynamic_modules[target]
                    if isinstance(module, nn.Sequential):
                        for layer in module:
                            if isinstance(layer, nn.Linear):
                                new_size = layer.out_features
                                break
                    elif isinstance(module, nn.Linear):
                        new_size = module.out_features
                
                mod_info['parameters'] = {
                    'target': target,
                    'new_size': new_size
                }
            
            elif mod.mod_type == 'change_activation':
                mod_info['parameters'] = {
                    'target': params.get('target'),
                    'new_activation': params.get('activation')
                }
        
        architecture_state['modification_history'].append(mod_info)
    
    # Get population data - capture ALL cells like in _write_population_visualization_state
    cells_data = []
    generation = 0
    
    if _current_germinal_center and hasattr(_current_germinal_center, 'population'):
        generation = getattr(_current_germinal_center, 'generation', 0)
        
        # Capture ALL cells in the population
        for idx, (cid, cell) in enumerate(list(_current_germinal_center.population.items())):
            # Determine cell type
            type_counts = defaultdict(int)
            cell_type = 'balanced'
            
            if hasattr(cell, 'genes'):
                active_genes = [g for g in cell.genes if g.is_active]
                if active_genes:
                    for gene in active_genes:
                        type_counts[gene.gene_type] += 1
                    
                    if type_counts:
                        dominant_type = max(type_counts, key=type_counts.get)
                        type_mapping = {
                            'S': 'stem',
                            'V': 'biosensor',
                            'D': 'effector',
                            'J': 'controller',
                            'Q': 'quantum'
                        }
                        cell_type = type_mapping.get(dominant_type, 'balanced')
            
            cell_info = {
                'cell_id': cid,
                'index': idx,
                'fitness': getattr(cell, 'fitness', 0.5),
                'generation': getattr(cell, 'generation', generation),
                'lineage': getattr(cell, 'lineage', []),
                'type': cell_type,
                'genes': [],
                'architecture': None,
                'connections': []
            }
            
            # Collect gene information
            if hasattr(cell, 'genes'):
                for gene in cell.genes:
                    gene_info = {
                        'gene_id': str(getattr(gene, 'gene_id', str(id(gene)))),
                        'gene_type': str(getattr(gene, 'gene_type', 'V')),
                        'position': int(getattr(gene, 'position', 0)),
                        'is_active': bool(getattr(gene, 'is_active', False)),
                        'is_quantum': 'Quantum' in gene.__class__.__name__,
                        'depth': float(gene.compute_depth().item()) if hasattr(gene, 'compute_depth') else 1.0,
                        'activation': float(getattr(gene, 'activation_ema', 0.0)),
                        'variant_id': int(getattr(gene, 'variant_id', 0)),
                        'methylation': float(gene.methylation_state.mean().item()) if hasattr(gene, 'methylation_state') else 0.0
                    }
                    cell_info['genes'].append(gene_info)
                
                # Track gene connections
                active_genes = [g for g in cell.genes if g.is_active]
                for idx1, gene1 in enumerate(active_genes):
                    for idx2, gene2 in enumerate(active_genes[idx1+1:], idx1+1):
                        cell_info['connections'].append({
                            'source': str(gene1.gene_id),
                            'target': str(gene2.gene_id),
                            'strength': float(abs(idx1 - idx2) / len(active_genes)) if active_genes else 0.0
                        })
            
            # Add architecture information if this is the current cell
            if cid == cell_id:
                cell_info['architecture'] = architecture_state
            
            cells_data.append(cell_info)
    
    # Create visualization state
    state = {
        'timestamp': time.time(),
        'generation': generation,
        'current_cell_id': cell_id,
        'cells': cells_data,
        'architecture_state': architecture_state
    }
    
    # Ensure visualization directory exists
    global _run_id
    if _run_id is None:
        from datetime import datetime
        _run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    
    viz_dir = os.path.join("scripts", "visualization_data", _run_id)
    os.makedirs(viz_dir, exist_ok=True)

    # Write state to unique file
    unique_filename = os.path.join(viz_dir, f"architecture_cell_{cell_id}_gen_{generation}.json")
    
    with state_lock:
        # Write to unique filename for archival
        with open(unique_filename, 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug(f"Wrote single cell architecture state to {unique_filename}")
        
        architecture_json_path = os.path.join("scripts", "te_ai_state.json")
        with open(architecture_json_path, 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug(f"Updated architecture_state.json at {architecture_json_path}")
        architecture_json_path = os.path.join("scripts", "visualization_data", 'architecture_state.json')
        with open(architecture_json_path, 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug(f"Updated architecture_state.json at {architecture_json_path}")


def _calculate_module_position(module_name, architecture_modifier):
    """Calculate 3D position based on connection graph"""
    # Build graph for topological analysis
    connections = architecture_modifier.module_connections
    all_modules = set(architecture_modifier.dynamic_modules.keys())
    
    # Find layers with no incoming connections (input layers)
    input_layers = all_modules - set(sum(connections.values(), []))
    
    # Calculate layer depth (distance from input)
    depths = {}
    visited = set()
    queue = [(layer, 0) for layer in input_layers]
    
    while queue:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        
        visited.add(current)
        depths[current] = depth
        
        # Add connected layers
        if current in connections:
            for next_layer in connections[current]:
                if next_layer not in visited:
                    queue.append((next_layer, depth + 1))
    
    # Get depth for this module
    depth = depths.get(module_name, 0)
    
    # Calculate vertical position based on parallel paths
    layers_at_depth = [m for m, d in depths.items() if d == depth]
    y_offset = layers_at_depth.index(module_name) if module_name in layers_at_depth else 0
    
    # Calculate Z position based on connectivity
    incoming = sum(1 for k, v in connections.items() if module_name in v)
    outgoing = len(connections.get(module_name, []))
    
    return {
        'x': depth * 3.0,
        'y': y_offset * 2.0 - len(layers_at_depth) / 2.0,
        'z': (outgoing - incoming) * 0.5
    }

def _get_activation_color(activation_name):
    """Get color based on activation type"""
    colors = {
        'ReLU': '#FF6B6B',
        'Tanh': '#4ECDC4',
        'Sigmoid': '#F7DC6F',
        'ELU': '#BB8FCE',
        'GELU': '#85C1E2',
        'SiLU': '#50E3C2',  # Swish
        'Linear': '#FFFFFF'
    }
    return colors.get(activation_name, '#9013FE')




def _write_population_visualization_state(cell_id, architecture_modifier):
    """Writes population-wide visualization data including all cells and their architectures."""
    
    # First create the basic node/link structure for this cell
    nodes = []
    links = []
    
    # Define node positions in 3D space
    node_positions = {}
    layer_keys = list(architecture_modifier.dynamic_modules.keys())
    for i, key in enumerate(layer_keys):
        node_positions[key] = {'x': i * 150, 'y': 0, 'z': 0}

    for name, module in architecture_modifier.dynamic_modules.items():
        # Determine the size of the layer
        size = 128  # Default
        if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[0], nn.Linear):
            size = module[0].out_features
        elif isinstance(module, nn.Linear):
            size = module.out_features
        
        # Determine the activation function
        activation = 'None'
        if isinstance(module, nn.Sequential):
            for layer in module:
                if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.GELU, nn.SiLU)):
                    activation = type(layer).__name__
        elif isinstance(module, nn.Linear):
            # Linear layers don't have activation
            activation = 'Linear'

        nodes.append({
            'id': name,
            'size': size,
            'activation': activation,
            'fx': node_positions[name]['x'],  # Fixed x position for stability
            'fy': node_positions[name]['y'],
            'fz': node_positions[name]['z']
        })

    for source, destinations in architecture_modifier.module_connections.items():
        for dest in destinations:
            # Ensure both source and dest are still valid nodes
            if source in architecture_modifier.dynamic_modules and dest in architecture_modifier.dynamic_modules:
                links.append({'source': source, 'target': dest})

    # Get the full cell population data if available  
    cells_data = []
    population = None
    generation = 0
    current_stress = 0
    
    # Try to access the germinal center instance
    global _current_germinal_center
    if _current_germinal_center and hasattr(_current_germinal_center, 'population'):
        population = _current_germinal_center.population
        generation = getattr(_current_germinal_center, 'generation', 0)
        current_stress = getattr(_current_germinal_center, 'current_stress', 0)
        # Reduced debug spam - only print occasionally
        if len(population) % 100 == 0:
            print(f"[DEBUG] Found germinal center with {len(population)} cells")
    else:
        print(f"[DEBUG] No germinal center found: gc={_current_germinal_center}, has_pop={hasattr(_current_germinal_center, 'population') if _current_germinal_center else False}")
    
    # If we have population data, collect comprehensive cell information
    if population:
        for idx, (cid, cell) in enumerate(list(population.items())):  # Get ALL cells
            
            ### START: Inlined Cell Type Logic ###
            type_counts = defaultdict(int)
            cell_type = 'balanced'  # Default value

            if hasattr(cell, 'genes'):
                active_genes = [g for g in cell.genes if g.is_active]
                if active_genes:
                    for gene in active_genes:
                        type_counts[gene.gene_type] += 1
                    
                    if type_counts:
                        dominant_type = max(type_counts, key=type_counts.get)
                        
                        type_mapping = {
                            'S': 'stem',
                            'V': 'biosensor',
                            'D': 'effector',
                            'J': 'controller',
                            'Q': 'quantum'
                        }
                        cell_type = type_mapping.get(dominant_type, 'balanced')
            ### END: Inlined Cell Type Logic ###

            cell_info = {
                'cell_id': cid,
                'index': idx,
                'fitness': getattr(cell, 'fitness', 0.5),
                'generation': getattr(cell, 'generation', generation),
                'lineage': getattr(cell, 'lineage', []),
                'type': cell_type, # <-- The calculated type is added here
                'genes': [],
                'architecture': None,
                'connections': []
            }
            
            # Collect gene information
            if hasattr(cell, 'genes'):
                for gene in cell.genes:
                    gene_info = {
                        'gene_id': str(getattr(gene, 'gene_id', str(id(gene)))),
                        'gene_type': str(getattr(gene, 'gene_type', 'V')),
                        'position': int(getattr(gene, 'position', 0)),
                        'is_active': bool(getattr(gene, 'is_active', False)),
                        'is_quantum': 'Quantum' in gene.__class__.__name__,
                        'depth': float(gene.compute_depth().item()) if hasattr(gene, 'compute_depth') else 1.0,
                        'activation': float(getattr(gene, 'activation_ema', 0.0)),
                        'variant_id': int(getattr(gene, 'variant_id', 0)),
                        'methylation': float(gene.methylation_state.mean().item()) if hasattr(gene, 'methylation_state') else 0.0
                    }
                    cell_info['genes'].append(gene_info)
                
                # Track gene connections
                active_genes = [g for g in cell.genes if g.is_active]
                for idx1, gene1 in enumerate(active_genes):
                    for idx2, gene2 in enumerate(active_genes[idx1+1:], idx1+1):
                        cell_info['connections'].append({
                            'source': str(gene1.gene_id),
                            'target': str(gene2.gene_id),
                            'strength': float(abs(idx1 - idx2) / len(active_genes)) if active_genes else 0.0
                        })
            
            # Add architecture information if this is the current cell
            if cid == cell_id and hasattr(cell, 'architecture_modifier'):
                arch = cell.architecture_modifier
                try:
                    # Safely extract serializable architecture info
                    module_names = []
                    if hasattr(arch, 'dynamic_modules'):
                        module_names = list(arch.dynamic_modules.keys())
                    
                    connections = {}
                    if hasattr(arch, 'module_connections'):
                        # Convert defaultdict to regular dict and ensure all values are lists
                        for k, v in arch.module_connections.items():
                            connections[str(k)] = list(v) if isinstance(v, (list, set, tuple)) else [str(v)]
                    
                    cell_info['architecture'] = {
                        'dna': str(getattr(arch, 'architecture_dna', 'N/A')),
                        'modules': module_names,
                        'connections': connections,
                        'modifications': len(getattr(arch, 'modification_history', []))
                    }
                except Exception as arch_error:
                    print(f"[WARNING] Failed to serialize architecture info: {arch_error}")
                    cell_info['architecture'] = {
                        'dna': 'error',
                        'modules': [],
                        'connections': {},
                        'modifications': 0
                    }
            
            cells_data.append(cell_info)
    
    # Create comprehensive state
    state = {
        # Individual cells data for visualization
        'cells': cells_data,
        
        # Legacy single cell visualization (for backward compatibility)
        'cell_id': cell_id,
        'nodes': nodes,
        'links': links,
        
        # Population metrics
        'generation': generation,
        'population_size': len(population) if population else 1,
        'total_genes': sum(len(c.get('genes', [])) for c in cells_data),
        'active_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('is_active', False)),
        'quantum_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('is_quantum', False)),
        
        # Cell type distribution
        'cell_types': {
            'V_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('gene_type') == 'V' and g.get('is_active')),
            'D_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('gene_type') == 'D' and g.get('is_active')),
            'J_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('gene_type') == 'J' and g.get('is_active')),
            'Q_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('is_quantum', False) and g.get('is_active')),
            'S_genes': sum(1 for c in cells_data for g in c.get('genes', []) if g.get('gene_type') == 'S' and g.get('is_active'))
        },
        
        # System state
        'phase': 'normal',
        'stress_level': current_stress,
        'mean_fitness': sum(c.get('fitness', 0) for c in cells_data) / max(len(cells_data), 1),
        
        # Timestamp
        'timestamp': time.time()
    }

    # Create unique filename with run ID
    global _run_id
    if _run_id is None:
        from datetime import datetime
        _run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join("visualization_data", _run_id)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Write to both the unique file and the standard polling file
    unique_filename = os.path.join(viz_dir, f"generation_{generation:04d}_state.json")
    
    with state_lock:
        # Write to unique file for this run
        with open(unique_filename, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            
        # Also write to standard polling file for live visualization
        with open('te_ai_state.json', 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False)
            
        # Write metadata about current run
        metadata = {
            'run_id': _run_id,
            'current_generation': generation,
            'latest_file': unique_filename,
            'timestamp': time.time()
        }
        with open(os.path.join(viz_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        # Also write a pointer file that tells the frontend where to find the data
        pointer = {
            'current_run_id': _run_id,
            'current_generation': generation,
            'data_directory': viz_dir,
            'latest_state_file': unique_filename,
            'te_ai_state_file': 'te_ai_state.json',  # For live polling
            'timestamp': time.time()
        }
        with open('current_run_pointer.json', 'w', encoding='utf-8') as f:
            json.dump(pointer, f, ensure_ascii=False, indent=2)
