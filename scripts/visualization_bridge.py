import asyncio
import json
import queue
import time 
from threading import Thread
import websockets
from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import cfg
from transposable_immune_ai_production_complete import ContinuousDepthGeneModule, QuantumGeneModule, ProductionBCell

from fast_optimized_te_ai import OptimizedProductionGerminalCenter









class VisualizationBridge:
    """Streams neural architecture events to visualization"""
    
    def __init__(self, port=8765):
        self.port = port
        self.event_queue = queue.Queue()
        self.clients = set()
        self.server_thread = None
        self.start_server()
        
    def start_server(self):
        """Start WebSocket server in background thread"""
        def run_server():
            async def handler(websocket):
                self.clients.add(websocket)
                try:
                    await websocket.wait_closed()
                finally:
                    self.clients.remove(websocket)
            
            async def broadcast_events():
                while True:
                    if not self.event_queue.empty():
                        event = self.event_queue.get()
                        if self.clients:
                            await asyncio.gather(
                                *[client.send(json.dumps(event)) for client in self.clients],
                                return_exceptions=True
                            )
                    await asyncio.sleep(0.01)
            
            async def main():
                async with websockets.serve(handler, "localhost", self.port):
                    await broadcast_events()
            
            asyncio.run(main())
        
        self.server_thread = Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def emit_event(self, event_type, data):
        """Queue event for broadcasting"""
        self.event_queue.put({
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        })
class InstrumentedGeneModule(ContinuousDepthGeneModule):
    """Gene module that reports its state changes"""

    def forward(self, x, edge_index, batch=None):
        result = super().forward(x, edge_index, batch)

        # Emit activation event
        viz_bridge.emit_event('gene_activation', {
            'gene_id': self.gene_id,
            'gene_type': self.gene_type,
            'position': self.position,
            'depth': self.compute_depth().item(),
            'activation': self.activation_ema,
            'is_quantum': isinstance(self, QuantumGeneModule),
            'chromatin_accessibility': self.chromatin_accessibility
        })

        return result

    def transpose(self, stress_level, diversity):
        child, action = super().transpose(stress_level, diversity)

        if action:
            # Emit transposition event
            viz_bridge.emit_event('transposition', {
                'gene_id': self.gene_id,
                'action': action,
                'old_position': self.position,
                'new_position': self.position if action != 'jump' else None,
                'stress_level': stress_level,
                'child_id': child.gene_id if child else None
            })

        return child, action

# Modified ProductionBCell
class InstrumentedBCell(ProductionBCell):
    """B-cell that reports structural changes"""

    def __init__(self, initial_genes, viz_bridge=None):
        super().__init__(initial_genes)
        self.viz_bridge = viz_bridge if viz_bridge else globals().get('viz_bridge')

        # Report initial structure
        self._report_structure()

    def _report_structure(self):
        """Send complete cell structure to visualization"""
        gene_data = []
        for gene in self.genes:
            if gene.is_active:
                gene_data.append({
                    'gene_id': gene.gene_id,
                    'gene_type': gene.gene_type,
                    'position': gene.position,
                    'depth': gene.compute_depth().item(),
                    'is_quantum': isinstance(gene, QuantumGeneModule),
                    'variant_id': gene.variant_id
                })

        if self.viz_bridge:
            self.viz_bridge.emit_event('cell_structure', {
                'cell_id': self.cell_id,
                'genes': gene_data,
                'fitness': self.fitness_history[-1] if self.fitness_history else 0,
                'generation': self.generation
            })

# Modified germinal center
class VisualizableGerminalCenter(OptimizedProductionGerminalCenter):
    """Germinal center with live visualization support"""
    
    def __init__(self, viz_bridge=None, **kwargs):
        # OptimizedProductionGerminalCenter doesn't accept any parameters
        super().__init__()
        self.viz_bridge = viz_bridge
        # Store the parameters that were passed but not used
        self.config_params = kwargs

    def evolve_generation(self, antigens):
        # Emit generation start
        if self.viz_bridge:
            self.viz_bridge.emit_event('generation_start', {
                'generation': self.generation + 1,
                'population_size': len(self.population),
                'stress_level': self.current_stress
            })

        # Run normal evolution
        super().evolve_generation(antigens)
        
        # Write comprehensive visualization state
        if hasattr(self, 'population') and self.population:
            import json
            
            # Count cell types
            gene_type_counts = {'V': 0, 'D': 0, 'J': 0, 'Q': 0, 'S': 0}
            total_cells = len(self.population)
            total_genes = 0
            active_genes = 0
            quantum_genes = 0
            
            # Collect population statistics
            for cell_id, cell in self.population.items():
                if hasattr(cell, 'genes'):
                    for gene in cell.genes:
                        total_genes += 1
                        if gene.is_active:
                            active_genes += 1
                            gene_type = gene.gene_type
                            if gene_type in gene_type_counts:
                                gene_type_counts[gene_type] += 1
                            if isinstance(gene, QuantumGeneModule):
                                quantum_genes += 1
            
            # Get current phase/zone
            current_phase = 'normal'
            if hasattr(self, 'cached_phase_detector') and hasattr(self.cached_phase_detector, 'current_phase'):
                current_phase = self.cached_phase_detector.current_phase
            elif hasattr(self, 'current_stress'):
                if self.current_stress > 0.8:
                    current_phase = 'critical'
                elif self.current_stress > 0.5:
                    current_phase = 'transitional'
            
            # Get a sample cell to visualize its structure
            cell_id = list(self.population.keys())[0] if self.population else 'unknown'
            cell = self.population[cell_id] if cell_id in self.population else None
            
            # Create comprehensive state
            state = {
                # Cell structure visualization
                'cell_id': cell_id,
                'nodes': [],
                'links': [],
                
                # Population metrics
                'generation': self.generation if hasattr(self, 'generation') else 0,
                'population_size': total_cells,
                'total_genes': total_genes,
                'active_genes': active_genes,
                'quantum_genes': quantum_genes,
                
                # Cell type distribution
                'cell_types': {
                    'V_genes': gene_type_counts['V'],
                    'D_genes': gene_type_counts['D'],
                    'J_genes': gene_type_counts['J'],
                    'Q_genes': gene_type_counts['Q'],
                    'S_genes': gene_type_counts['S']
                },
                
                # System state
                'phase': current_phase,
                'stress_level': self.current_stress if hasattr(self, 'current_stress') else 0,
                'mean_fitness': 0,
                
                # Timestamp
                'timestamp': time.time()
            }
            
            # Calculate mean fitness
            if hasattr(self, 'population'):
                fitness_values = []
                for c in self.population.values():
                    if hasattr(c, 'fitness'):
                        fitness_values.append(c.fitness)
                if fitness_values:
                    state['mean_fitness'] = sum(fitness_values) / len(fitness_values)
            
            # Add gene structure visualization for the sample cell
            if cell and hasattr(cell, 'genes'):
                # Create nodes for each gene
                for i, gene in enumerate(cell.genes):
                    if gene.is_active:
                        state['nodes'].append({
                            'id': gene.gene_id,
                            'size': int(gene.compute_depth().item() * 50) if hasattr(gene, 'compute_depth') else 50,
                            'activation': gene.gene_type,
                            'fx': i * 100,
                            'fy': 0,
                            'fz': 0,
                            'color': '#ff0000' if isinstance(gene, QuantumGeneModule) else None
                        })
                
                # Create links between consecutive genes
                active_genes_list = [g for g in cell.genes if g.is_active]
                for i in range(len(active_genes_list) - 1):
                    state['links'].append({
                        'source': active_genes_list[i].gene_id,
                        'target': active_genes_list[i+1].gene_id
                    })
            
            # Write to file
            with open('te_ai_state.json', 'w') as f:
                json.dump(state, f, indent=2)

        # Emit generation summary
        if self.viz_bridge:
            # Calculate metrics safely
            mean_fitness = 0
            if hasattr(self, 'population') and self.population:
                try:
                    mean_fitness = sum(getattr(c, 'fitness', 0) for c in self.population) / len(self.population)
                except:
                    mean_fitness = 0
            
            self.viz_bridge.emit_event('generation_complete', {
                'generation': self.generation,
                'metrics': {
                    'mean_fitness': mean_fitness,
                    'population_size': len(self.population) if hasattr(self, 'population') else 0,
                    'stress_level': getattr(self, 'current_stress', 0)
                },
                'phase_state': getattr(self, 'cached_phase_detector', self).current_phase if hasattr(getattr(self, 'cached_phase_detector', self), 'current_phase') else 'normal'
            })





viz_bridge = VisualizationBridge()
