"""
Instrumented neural components that report state changes for visualization
"""


from scripts.core.production_germinal_center import ProductionGerminalCenter
from scripts.core.production_b_cell import ProductionBCell

import time
import asyncio
import websockets
import json
from threading import Thread
from scripts.core.ode import ContinuousDepthGeneModule
from scripts.core.quantum_gene import QuantumGeneModule
import queue




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
            async def handler(websocket, path):
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
                                *[client.send(json.dumps(event)) for client in self.clients]
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
        
        

viz_bridge = VisualizationBridge

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

    def __init__(self, initial_genes):
        super().__init__(initial_genes)

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

        viz_bridge.emit_event('cell_structure', {
            'cell_id': self.cell_id,
            'genes': gene_data,
            'fitness': self.fitness_history[-1] if self.fitness_history else 0,
            'generation': self.generation
        })

# Modified germinal center
class VisualizableGerminalCenter(ProductionGerminalCenter):
    """Germinal center with live visualization support"""
    
    def __init__(self, viz_bridge=None, **kwargs):
        super().__init__(**kwargs)
        self.viz_bridge = viz_bridge

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

        # Emit generation summary
        if self.viz_bridge:
            self.viz_bridge.emit_event('generation_complete', {
                'generation': self.generation,
                'metrics': self._get_current_metrics(),
                'phase_state': self.phase_detector.current_phase
            })



class VisualizableGerminalCenter(ProductionGerminalCenter):
    """Germinal center with live visualization support"""
    
    def __init__(self, viz_bridge=None, **kwargs):
        super().__init__(**kwargs)
        self.viz_bridge = viz_bridge
    
    def create_b_cell(self, initial_genes):
        """Override to create instrumented B cells"""
        return InstrumentedBCell(initial_genes, self.viz_bridge)
    
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
        
        # Emit generation summary
        if self.viz_bridge:
            self.viz_bridge.emit_event('generation_complete', {
                'generation': self.generation,
                'metrics': self._get_current_metrics(),
                'phase_state': self.phase_detector.current_phase if hasattr(self, 'phase_detector') else 'unknown'
            })
    
    def _get_current_metrics(self):
        """Gather current metrics for visualization"""
        metrics = {
            'avg_fitness': sum(cell.fitness_history[-1] if cell.fitness_history else 0 
                             for cell in self.population) / len(self.population),
            'population_size': len(self.population),
            'stress_level': self.current_stress,
            'diversity': self._calculate_diversity(),
            'best_fitness': max(cell.fitness_history[-1] if cell.fitness_history else 0 
                              for cell in self.population)
        }
        return metrics
    
    def _calculate_diversity(self):
        """Calculate population diversity"""
        # Simple diversity metric based on unique gene configurations
        gene_signatures = set()
        for cell in self.population:
            signature = tuple(sorted([
                (g.gene_type, g.position, g.variant_id) 
                for g in cell.genes if g.is_active
            ]))
            gene_signatures.add(signature)
        
        return len(gene_signatures) / len(self.population)