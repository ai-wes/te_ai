import asyncio
import json
import queue
import time 
from threading import Thread
import websockets
from dataclasses import dataclass, field
from typing import List, Dict, Any
from scripts.config import cfg
from scripts.depricated.transposable_immune_ai_production_complete import ContinuousDepthGeneModule, QuantumGeneModule, ProductionBCell
from scripts.run_optimized_simulation import OptimizedProductionGerminalCenter
import os
import sys
from datetime import datetime









class VisualizationBridge:
    """Streams neural architecture events to visualization"""
    
    def __init__(self, port=8765):
        self.port = port
        self.event_queue = queue.Queue()
        self.clients = set()
        self.server_thread = None
        
        # Create unique run ID and visualization directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.viz_dir = os.path.join("visualization_data", self.run_id)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Current state file (for live polling)
        self.current_state_file = "te_ai_state.json"
        
        # Generation counter
        self.generation_counter = 0
        
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
    
    def save_generation_snapshot(self, state_data, generation):
        """Save a snapshot of the state for this generation"""
        # Save to generation-specific file
        gen_file = os.path.join(self.viz_dir, f"generation_{generation:04d}.json")
        with open(gen_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        # Also update the current state file for live viewing
        with open(self.current_state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        # Save a metadata file with run info
        metadata = {
            'run_id': self.run_id,
            'current_generation': generation,
            'timestamp': time.time(),
            'viz_dir': self.viz_dir,
            'total_files': generation + 1
        }
        with open(os.path.join(self.viz_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
            
            
            
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
                'fitness': self.fitness_history[-1] if hasattr(self, 'fitness_history') and self.fitness_history else 0,
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
        
        # Track changes between generations
        self.previous_cell_states = {}
        self.cell_change_history = []

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
            
            # Collect data for all cells with detailed tracking
            cells_data = []
            cell_ids = list(self.population.keys())
            
            for idx, cell_id in enumerate(cell_ids):
                cell = self.population[cell_id]
                if hasattr(cell, 'genes'):
                    cell_data = {
                        'cell_id': str(cell_id),  # Ensure it's a string
                        'index': idx,
                        'fitness': float(cell.fitness_history[-1]) if hasattr(cell, 'fitness_history') and cell.fitness_history else 0.5,
                        'generation': int(cell.generation) if hasattr(cell, 'generation') else 0,
                        'lineage': list(cell.lineage) if hasattr(cell, 'lineage') else [],
                        'genes': [],
                        'architecture': None,
                        'connections': []
                    }
                    
                    # Add detailed gene information
                    for gene in cell.genes:
                        gene_info = {
                            'gene_id': str(gene.gene_id),
                            'gene_type': str(gene.gene_type),
                            'position': int(gene.position) if hasattr(gene, 'position') else 0,
                            'is_active': bool(gene.is_active),
                            'is_quantum': isinstance(gene, QuantumGeneModule),
                            'depth': float(gene.compute_depth().item()) if hasattr(gene, 'compute_depth') else 1.0,
                            'activation': float(gene.activation_ema) if hasattr(gene, 'activation_ema') else 0.0,
                            'variant_id': int(gene.variant_id) if hasattr(gene, 'variant_id') else 0,
                            'methylation': float(gene.methylation_state.mean().item()) if hasattr(gene, 'methylation_state') else 0.0
                        }
                        cell_data['genes'].append(gene_info)
                    
                    # Add architecture information if cell has it
                    if hasattr(cell, 'architecture_modifier'):
                        arch = cell.architecture_modifier
                        cell_data['architecture'] = {
                            'dna': arch.architecture_dna if hasattr(arch, 'architecture_dna') else None,
                            'modules': list(arch.dynamic_modules.keys()) if hasattr(arch, 'dynamic_modules') else [],
                            'connections': dict(arch.module_connections) if hasattr(arch, 'module_connections') else {},
                            'modifications': len(arch.modification_history) if hasattr(arch, 'modification_history') else 0
                        }
                    
                    # Track gene connections/relationships
                    active_genes_list = [g for g in cell.genes if g.is_active]
                    for i, gene1 in enumerate(active_genes_list):
                        for j, gene2 in enumerate(active_genes_list[i+1:], i+1):
                            # Record pairwise gene relationships
                            cell_data['connections'].append({
                                'source': str(gene1.gene_id),
                                'target': str(gene2.gene_id),
                                'strength': float(abs(i - j) / len(active_genes_list)) if active_genes_list else 0.0
                            })
                    
                    cells_data.append(cell_data)
                    
                    # Track changes from previous generation
                    if str(cell_id) in self.previous_cell_states:
                        prev_state = self.previous_cell_states[str(cell_id)]
                        changes = []
                        
                        # Check fitness change
                        if abs(prev_state.get('fitness', 0) - cell_data['fitness']) > 0.01:
                            changes.append({
                                'type': 'fitness_change',
                                'from': prev_state.get('fitness', 0),
                                'to': cell_data['fitness']
                            })
                        
                        # Check gene changes
                        prev_gene_ids = {g['gene_id'] for g in prev_state.get('genes', [])}
                        curr_gene_ids = {g['gene_id'] for g in cell_data['genes']}
                        
                        # New genes
                        for gene_id in (curr_gene_ids - prev_gene_ids):
                            changes.append({
                                'type': 'gene_added',
                                'gene_id': gene_id
                            })
                        
                        # Lost genes
                        for gene_id in (prev_gene_ids - curr_gene_ids):
                            changes.append({
                                'type': 'gene_removed',
                                'gene_id': gene_id
                            })
                        
                        if changes:
                            self.cell_change_history.append({
                                'cell_id': str(cell_id),
                                'generation': int(self.generation),
                                'changes': changes
                            })
                    
                    # Store current state for next comparison
                    # Make a deep copy to avoid reference issues
                    self.previous_cell_states[str(cell_id)] = {
                        'fitness': cell_data['fitness'],
                        'genes': [{'gene_id': g['gene_id']} for g in cell_data['genes']]
                    }
            
            # Create comprehensive state
            state = {
                # Individual cells data for visualization
                'cells': cells_data,
                
                # Legacy single cell visualization (for backward compatibility)
                'cell_id': cells_data[0]['cell_id'] if cells_data else 'unknown',
                'nodes': [],
                'links': [],
                
                # Population metrics
                'generation': int(self.generation) if hasattr(self, 'generation') else 0,
                'population_size': int(total_cells),
                'total_genes': int(total_genes),
                'active_genes': int(active_genes),
                'quantum_genes': int(quantum_genes),
                
                # Cell type distribution
                'cell_types': {
                    'V_genes': gene_type_counts['V'],
                    'D_genes': gene_type_counts['D'],
                    'J_genes': gene_type_counts['J'],
                    'Q_genes': gene_type_counts['Q'],
                    'S_genes': gene_type_counts['S']
                },
                
                # System state
                'phase': str(current_phase),
                'stress_level': float(self.current_stress) if hasattr(self, 'current_stress') else 0.0,
                'mean_fitness': 0.0,
                
                # Cell actions and events
                'recent_actions': [],
                'architecture_modifications': [],
                'transposition_events': [],
                'cell_changes': self.cell_change_history[-100:],  # Last 100 changes
                
                # Timestamp
                'timestamp': time.time()
            }
            
            # Calculate mean fitness
            if hasattr(self, 'population'):
                fitness_values = []
                for c in self.population.values():
                    if hasattr(c, 'fitness_history') and c.fitness_history:
                        fitness_values.append(c.fitness_history[-1])
                if fitness_values:
                    state['mean_fitness'] = sum(fitness_values) / len(fitness_values)
            
            # Collect recent actions and events from the visualization bridge
            if self.viz_bridge and hasattr(self.viz_bridge, 'event_queue'):
                # Get recent events from the queue (non-blocking)
                recent_events = []
                try:
                    import queue
                    while not self.viz_bridge.event_queue.empty() and len(recent_events) < 50:
                        event = self.viz_bridge.event_queue.get_nowait()
                        recent_events.append(event)
                        
                        # Categorize events
                        if event['type'] == 'transposition':
                            state['transposition_events'].append(event['data'])
                        elif event['type'] == 'architecture_modification':
                            state['architecture_modifications'].append(event['data'])
                        elif event['type'] in ['gene_activation', 'cell_structure']:
                            state['recent_actions'].append(event)
                            
                except queue.Empty:
                    pass
                
                # Put events back for WebSocket broadcast
                for event in recent_events:
                    self.viz_bridge.event_queue.put(event)
            
            # Add gene structure visualization for the first cell (legacy support)
            cell = self.population[cells_data[0]['cell_id']] if cells_data else None
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
            
            # Save generation snapshot
            generation = self.generation if hasattr(self, 'generation') else 0
            self.viz_bridge.save_generation_snapshot(state, generation)
            
            # Log summary
            print(f"   📸 Saved visualization snapshot: generation_{generation:04d}.json")
            
            # Generate/update runs manifest periodically
            if generation % 10 == 0:  # Every 10 generations
                try:
                    import subprocess
                    subprocess.run([sys.executable, "generate_runs_manifest.py"], capture_output=True, text=True)
                    print("   📋 Updated runs manifest")
                except Exception as e:
                    print(f"   ⚠️  Failed to update runs manifest: {e}")

        # Emit generation summary
        if self.viz_bridge:
            # Calculate metrics safely
            mean_fitness = 0
            if hasattr(self, 'population') and self.population:
                try:
                    fitness_values = []
                    for c in self.population.values():
                        if hasattr(c, 'fitness_history') and c.fitness_history:
                            fitness_values.append(c.fitness_history[-1])
                        else:
                            fitness_values.append(0)
                    mean_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0
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
