"""
Polling-based visualization bridge for TE-AI
Uses a simple file-based approach for communication
"""

import json
import time
import numpy as np
from pathlib import Path

# Global state file path
STATE_FILE = Path("te_ai_state.json")

class PollingBridge:
    """Simple polling-based bridge that writes state to a JSON file"""
    
    def __init__(self):
        self.state = {
            'generation': 0,
            'population': 0,
            'fitness': 0.0,
            'diversity': 0.0,
            'stress': 0.0,
            'phase': 'normal',
            'events': [],
            'timestamp': time.time()
        }
        self.write_state()
    
    def write_state(self):
        """Write current state to file"""
        self.state['timestamp'] = time.time()
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update_metrics(self, **kwargs):
        """Update metrics in the state"""
        for key, value in kwargs.items():
            if key in self.state:
                self.state[key] = value
        self.write_state()
    
    def add_event(self, event_type, data):
        """Add an event to the event list"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        # Keep only last 100 events
        self.state['events'] = self.state['events'][-99:] + [event]
        self.write_state()

# Global instance
polling_bridge = PollingBridge()

# Import original classes and wrap them
from scripts.depricated.transposable_immune_ai_production_complete import ContinuousDepthGeneModule, ProductionBCell
from scripts.run_optimized_simulation import OptimizedProductionGerminalCenter

class InstrumentedGeneModule(ContinuousDepthGeneModule):
    """Gene module that reports its state changes"""
    
    def forward(self, x, edge_index, batch=None):
        result = super().forward(x, edge_index, batch)
        # Don't emit every activation, too frequent
        return result
    
    def transpose(self, stress_level, diversity):
        child, action = super().transpose(stress_level, diversity)
        
        if action != 'none':
            polling_bridge.add_event('transposition', {
                'gene_id': self.gene_id,
                'action': action,
                'stress_level': stress_level
            })
        
        return child, action

class InstrumentedBCell(ProductionBCell):
    """B-cell that reports structure changes"""
    
    def __init__(self, initial_genes):
        super().__init__(initial_genes)
        # Report initial structure
        polling_bridge.add_event('cell_created', {
            'cell_id': id(self),
            'num_genes': len(self.genes)
        })

class PollingGerminalCenter(OptimizedProductionGerminalCenter):
    """Germinal center that updates polling state"""
    
    def __init__(self):
        super().__init__()
        # Set global reference for visualization
        import scripts.depricated.transposable_immune_ai_production_complete as prod
        prod._current_germinal_center = self
    
    def evolve_generation(self, antigens):
        # Update generation start
        polling_bridge.update_metrics(
            generation=self.generation + 1,
            population=len(self.population)
        )
        
        # Run normal evolution
        super().evolve_generation(antigens)
        
        # Calculate and update metrics
        if self.population:
            # Calculate mean fitness from the fitness_scores recorded in the last generation
            fitness_values = [score for score in self.history[-1]['fitness_scores'].values()] if self.history and 'fitness_scores' in self.history[-1] else [0]
            mean_fitness = np.mean(fitness_values) if fitness_values else 0
            
            # Assign fitness to each cell object for easy access
            if self.history and 'fitness_scores' in self.history[-1]:
                for cell_id, fitness in self.history[-1]['fitness_scores'].items():
                    if cell_id in self.population:
                        self.population[cell_id].fitness = fitness
        else:
            mean_fitness = 0
        
        # Collect detailed cell data for ALL cells
        cells_data = []
        population_list = list(self.population.values())
        
        # Process ALL cells, not just a sample
        for i, cell in enumerate(population_list):
            cell_info = {
                'cell_id': f'cell_{cell.cell_id}',
                'index': i,
                'fitness': getattr(cell, 'fitness', 0),
                'generation': getattr(cell, 'generation', self.generation),
                'lineage': getattr(cell, 'lineage', []),
                'cell_type': self._get_cell_specialization(cell),
                'zone': self._get_cell_zone(cell),
                'genes': [],
                'architecture': None,
                'connections': []
            }
            
            # Get gene information
            if hasattr(cell, 'genes'):
                for gene in cell.genes:
                    gene_info = {
                        'gene_id': getattr(gene, 'gene_id', str(id(gene))),
                        'gene_type': getattr(gene, 'gene_type', 'V'),
                        'is_active': getattr(gene, 'is_active', False),
                        'is_quantum': 'Quantum' in gene.__class__.__name__,
                        'depth': gene.compute_depth().item() if hasattr(gene, 'compute_depth') else 1.0,
                        'position': getattr(gene, 'position', 0),
                        'activation': getattr(gene, 'activation_ema', 0.5),
                        'variant_id': getattr(gene, 'variant_id', 0),
                        'methylation': gene.methylation_state.mean().item() if hasattr(gene, 'methylation_state') else 0.0
                    }
                    cell_info['genes'].append(gene_info)
                
                # Track gene connections/relationships
                active_genes = [g for g in cell.genes if g.is_active]
                for idx1, gene1 in enumerate(active_genes):
                    for idx2, gene2 in enumerate(active_genes[idx1+1:], idx1+1):
                        cell_info['connections'].append({
                            'source': gene1.gene_id,
                            'target': gene2.gene_id,
                            'strength': abs(idx1 - idx2) / len(active_genes) if active_genes else 0
                        })
            
            # Add architecture information if cell has it
            if hasattr(cell, 'architecture_modifier'):
                arch = cell.architecture_modifier
                cell_info['architecture'] = {
                    'dna': getattr(arch, 'architecture_dna', None),
                    'modules': list(arch.dynamic_modules.keys()) if hasattr(arch, 'dynamic_modules') else [],
                    'connections': dict(arch.module_connections) if hasattr(arch, 'module_connections') else {},
                    'modifications': len(getattr(arch, 'modification_history', []))
                }
            
            cells_data.append(cell_info)
        
        # Update state with comprehensive cell data
        polling_bridge.state['cells'] = cells_data
        polling_bridge.state['population_size'] = len(cells_data)
        polling_bridge.state['total_genes'] = sum(len(c['genes']) for c in cells_data)
        polling_bridge.state['active_genes'] = sum(1 for c in cells_data for g in c['genes'] if g['is_active'])
        polling_bridge.state['quantum_genes'] = sum(1 for c in cells_data for g in c['genes'] if g['is_quantum'])
        
        # Get current phase
        current_phase = getattr(self.cached_phase_detector, 'current_phase', 'normal')
        
        # Check for dream phase
        if current_phase == 'dream' and polling_bridge.state.get('phase') != 'dream':
            polling_bridge.add_event('dream_phase_start', {
                'num_realities': 5,
                'generation': self.generation
            })
        elif current_phase != 'dream' and polling_bridge.state.get('phase') == 'dream':
            polling_bridge.add_event('dream_phase_end', {
                'generation': self.generation
            })
        
        # Add architectural census data
        architectural_data = {
            'layer_counts': [],
            'connection_counts': [],
            'quantum_genes': [],
            'architecture_species': {}
        }
        for cell in population_list[:20]:  # Sample first 20 cells for performance
            quantum_count = sum(1 for gene in cell.genes if 'Quantum' in gene.__class__.__name__)
            architectural_data['quantum_genes'].append(quantum_count)
            
            if hasattr(cell, 'architecture_modifier'):
                mod = cell.architecture_modifier
                architectural_data['layer_counts'].append(len(mod.dynamic_modules))
                connections = sum(len(v) for v in mod.module_connections.values())
                architectural_data['connection_counts'].append(connections)
                
                # Track architecture DNA
                dna = getattr(mod, 'architecture_dna', 'unknown')
                architectural_data['architecture_species'][dna] = architectural_data['architecture_species'].get(dna, 0) + 1
        
        polling_bridge.update_metrics(
            fitness=mean_fitness,
            diversity=self.history[-1].get('diversity', 0),
            stress=self.history[-1].get('current_stress', 0),
            phase=current_phase,
            mean_layers=np.mean(architectural_data['layer_counts']) if architectural_data['layer_counts'] else 0,
            mean_connections=np.mean(architectural_data['connection_counts']) if architectural_data['connection_counts'] else 0,
            mean_quantum_genes=np.mean(architectural_data['quantum_genes']) if architectural_data['quantum_genes'] else 0,
            architecture_species=len(architectural_data['architecture_species'])
        )
    
    def _get_cell_specialization(self, cell):
        """Determine cell specialization based on gene composition."""
        if not hasattr(cell, 'genes'):
            return 'balanced'

        gene_types = {'V': 0, 'D': 0, 'J': 0, 'S': 0, 'Q': 0}
        active_genes = 0
        for gene in cell.genes:
            if getattr(gene, 'is_active', False):
                active_genes += 1
                gene_type = getattr(gene, 'gene_type', 'V')
                if 'Quantum' in gene.__class__.__name__:
                    gene_type = 'Q'
                elif 'Stem' in gene.__class__.__name__:
                    gene_type = 'S'
                
                if gene_type in gene_types:
                    gene_types[gene_type] += 1

        if active_genes == 0:
            return 'inactive'

        # Find primary specialization
        primary_type = max(gene_types, key=gene_types.get)
        
        # If the max count is 0, it's balanced/inactive
        if gene_types[primary_type] == 0:
            return 'balanced'

        # Check if it's truly dominant or balanced
        if gene_types[primary_type] < active_genes * 0.5:
            return 'balanced'
            
        return primary_type

    def _get_cell_zone(self, cell):
        """Determine the germinal center zone for a cell."""
        fitness = getattr(cell, 'fitness', 0.5)
        generation = getattr(self, 'generation', 0)
        stress = getattr(self, 'current_stress', 0.0)
        
        has_quantum = False
        active_gene_count = 0
        total_gene_count = 0
        if hasattr(cell, 'genes'):
            for gene in cell.genes:
                total_gene_count += 1
                if getattr(gene, 'is_active', False):
                    active_gene_count += 1
                    if 'Quantum' in gene.__class__.__name__:
                        has_quantum = True
        
        mutation_rate = (active_gene_count / max(total_gene_count, 1)) * 0.5 + stress * 0.5
        
        if fitness > 0.8 and generation > 20:
            return 'memoryZone'
        if has_quantum:
            return 'quantumLayer'
        if mutation_rate > 0.5 or stress > 0.6:
            return 'darkZone'
        if fitness > 0.65:
            return 'lightZone'
        
        return 'mantleZone'