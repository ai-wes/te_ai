#core/global_workspace.py
"""
Global Workspace for Collective Consciousness
=============================================

A shared memory space where cells can broadcast their internal states,
allowing the swarm to collaboratively solve problems. It is structured
as a causal graph to maintain a historical record of the swarm's cognition.
"""

from collections import defaultdict
import torch
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

class GlobalWorkspace:
    """
    A shared blackboard for the swarm. Manages real-time communication
    and maintains a causal history of problem-solving events.
    """
    def __init__(self):
        # The workspace stores states related to specific challenges (antigen hashes)
        self.workspace = defaultdict(lambda: {
            "confused_states": [],
            "confident_states": [],
            "discussion": []
        })
        # The Causal Graph would be a more complex graph structure in a full implementation
        self.causal_graph_log = []

    def broadcast(self, antigen_hash: str, cell_id: str, message_type: str, state_tensor: torch.Tensor):
        """
        A cell broadcasts its state and a message to the workspace.
        
        Args:
            antigen_hash: A unique identifier for the problem/antigen batch.
            cell_id: The ID of the broadcasting cell.
            message_type: 'CONFUSED' or 'CONFIDENT'.
            state_tensor: The cell's internal representation tensor.
        """
        log_message = f"   [Workspace] Cell {cell_id[:8]} broadcasted '{message_type}' for challenge {antigen_hash[:8]}"
        
        if message_type == 'CONFUSED':
            self.workspace[antigen_hash]["confused_states"].append((cell_id, state_tensor.detach().cpu()))
            log_message += f" (Total confused: {len(self.workspace[antigen_hash]['confused_states'])})"
        elif message_type == 'CONFIDENT':
            self.workspace[antigen_hash]["confident_states"].append((cell_id, state_tensor.detach().cpu()))
            log_message += f" (Total confident: {len(self.workspace[antigen_hash]['confident_states'])})"
        
        logger.info(log_message)
        
        # Log this event in the causal history
        self.causal_graph_log.append({
            "event": "broadcast",
            "antigen_hash": antigen_hash,
            "cell_id": cell_id,
            "message": message_type
        })

    def query(self, antigen_hash: str) -> dict:
        """
        Allows a cell to read the current state of the workspace for a given problem.
        """
        return self.workspace[antigen_hash]

    def clear_challenge(self, antigen_hash: str):
        """Clears the workspace for a given challenge after it's resolved."""
        if antigen_hash in self.workspace:
            del self.workspace[antigen_hash]
