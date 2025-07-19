# scripts/core/causal_tapestry.py

import networkx as nx
from typing import Dict, Any

class CausalTapestry:
    """
    A shared, evolving knowledge graph that tracks the lineage, genetics,
    and key events of the entire Symbiotic Swarm. Implemented with networkx.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.run_id = None # To track different experiments

    def reset(self, run_id: str):
        """Clears the graph for a new experimental run."""
        self.graph.clear()
        self.run_id = run_id
        print(f"Causal Tapestry reset for new run: {run_id}")

    # --- Methods for Adding Nodes ---

    def add_cell_node(self, cell_id: str, generation: int, island_name: str, fitness: float, genes: list):
        self.graph.add_node(
            cell_id,
            type='cell',
            generation=generation,
            island=island_name,
            fitness=fitness,
            genes=genes # List of gene_ids
        )

    def add_gene_node(self, gene_id: str, gene_type: str, variant_id: Any):
        if not self.graph.has_node(gene_id):
            self.graph.add_node(
                gene_id,
                type='gene',
                gene_type=gene_type,
                variant_id=str(variant_id)
            )

    def add_event_node(self, event_id: str, event_type: str, generation: int, details: Dict):
        self.graph.add_node(
            event_id,
            type='event',
            event_type=event_type,
            generation=generation,
            details=details
        )

    # --- Methods for Adding Edges (Relationships) ---

    def log_lineage(self, parent_id: str, child_id: str):
        """Records a parent-child relationship."""
        self.graph.add_edge(parent_id, child_id, type='PARENT_OF')

    def log_gene_composition(self, cell_id: str, gene_id: str):
        """Links a cell to the genes it contains."""
        self.graph.add_edge(cell_id, gene_id, type='CONTAINS_GENE')

    def log_event_participation(self, participant_id: str, event_id: str, role: str):
        """Links a cell or gene to an event."""
        self.graph.add_edge(participant_id, event_id, type='PARTICIPATED_IN', role=role)

    def log_event_output(self, event_id: str, output_id: str, role: str):
        """Links an event to its resulting cell or gene."""
        self.graph.add_edge(event_id, output_id, type='PRODUCED', role=role)

    # --- Methods for Saving and Loading ---

    def save_tapestry(self, filepath: str):
        """Saves the graph to a file."""
        nx.write_graphml(self.graph, filepath)
        print(f"Causal Tapestry saved to {filepath}")

    def load_tapestry(self, filepath: str):
        """Loads a graph from a file, replacing the current one."""
        self.graph = nx.read_graphml(filepath)
        print(f"Causal Tapestry loaded from {filepath}")

    def merge_tapestry(self, filepath: str):
        """Loads another graph and merges it with the current one."""
        other_graph = nx.read_graphml(filepath)
        self.graph = nx.compose(self.graph, other_graph)
        print(f"Causal Tapestry from {filepath} merged into current graph.")