"""
Instrumented neural components that report state changes for visualization
"""

from transposable_immune_ai_production_complete import (
    ContinuousDepthGeneModule, ProductionBCell, 
    OptimizedProductionGerminalCenter, QuantumGeneModule
)
import time


class InstrumentedGeneModule(ContinuousDepthGeneModule):
    """Gene module that reports its state changes"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viz_bridge = None  # Will be set by parent
    
    def forward(self, x, edge_index, batch=None):
        result = super().forward(x, edge_index, batch)
        
        # Emit activation event if visualization bridge is available
        if self.viz_bridge:
            self.viz_bridge.emit_event('gene_activation', {
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
        old_position = self.position
        child, action = super().transpose(stress_level, diversity)
        
        if action and self.viz_bridge:
            # Emit transposition event
            self.viz_bridge.emit_event('transposition', {
                'gene_id': self.gene_id,
                'action': action,
                'old_position': old_position,
                'new_position': self.position if action != 'jump' else child.position if child else None,
                'stress_level': stress_level,
                'child_id': child.gene_id if child else None
            })
        
        return child, action


class InstrumentedBCell(ProductionBCell):
    """B-cell that reports structural changes"""
    
    def __init__(self, initial_genes, viz_bridge=None):
        super().__init__(initial_genes)
        self.viz_bridge = viz_bridge
        
        # Pass viz_bridge to all genes
        for gene in self.genes:
            if hasattr(gene, 'viz_bridge'):
                gene.viz_bridge = viz_bridge
        
        # Report initial structure
        if self.viz_bridge:
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
        
        self.viz_bridge.emit_event('cell_structure', {
            'cell_id': self.cell_id,
            'genes': gene_data,
            'fitness': self.fitness_history[-1] if self.fitness_history else 0,
            'generation': self.generation
        })
    
    def forward(self, x, edge_index, batch=None):
        result = super().forward(x, edge_index, batch)
        
        # Report structure changes after forward pass
        if self.viz_bridge:
            self._report_structure()
        
        return result
    
    def add_gene(self, gene):
        super().add_gene(gene)
        if hasattr(gene, 'viz_bridge'):
            gene.viz_bridge = self.viz_bridge
    
    def undergo_somatic_hypermutation(self, rate=0.01):
        super().undergo_somatic_hypermutation(rate)
        
        # Report mutation event
        if self.viz_bridge:
            self.viz_bridge.emit_event('mutation', {
                'cell_id': self.cell_id,
                'mutation_rate': rate,
                'active_genes': sum(1 for g in self.genes if g.is_active)
            })


class VisualizableGerminalCenter(OptimizedProductionGerminalCenter):
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