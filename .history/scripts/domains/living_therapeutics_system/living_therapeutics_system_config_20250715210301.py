# ============================================================================
# THERAPEUTIC DOMAIN CONFIGURATION
# ============================================================================

import torch
from dataclasses import field
import os
import json
import logging





class TherapeuticConfig:
    # --- Merged Configuration from CFG and ProductionConfig ---
    
############################# MAIN TE-AI CONFIGURATION ############################################

    # 1. Device and Performance
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_mixed_precision: bool = True  # AMP for speed
    gradient_checkpointing: bool = True  # Memory efficiency
    num_workers: int = 1
    pin_memory: bool = True
    use_jit_compilation: bool = True
    use_amp: bool = True  # Automatic Mixed Precision



    # 2. Neural Architecture & ODEs
    feature_dim: int = 64
    hidden_dim: int = 128
    num_heads: int = 8  # For multi-head attention
    ode_solver: str = "dopri5"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    ode_time_points: int = 20
    min_depth: float = 0.1
    max_depth: float = 3.0

    # 3. Genome and Transposon Dynamics
    genome_size: int = 10000
    num_genes: int = 100
    gene_min_len: int = 50
    gene_max_len: int = 200
    num_transposons: int = 50
    transposon_min_len: int = 30
    transposon_max_len: int = 150
    base_transpose_prob: float = 0.01  # Renamed from transposition_rate
    stress_multiplier: float = 10.0
    duplication_cost: float = 0.1
    max_genes_per_clone: int = 10
    transposition_energy_cost: float = 0.05
    excision_rate: float = 0.03
    insertion_rate: float = 0.04

    # 4. Epigenetic System
    epigenetic_markers: int = 10
    methylation_rate: float = 0.02  # Updated value
    methylation_inheritance: float = 0.85
    methylation_effect_strength: float = 0.5
    demethylation_rate: float = 0.05
    histone_modification_types: int = 5
    histone_modification_rate: float = 0.01
    chromatin_remodeling_threshold: float = 0.7

    # 5. Gene Regulatory Network (GRN)
    grn_num_nodes: int = 150  # Updated calculation in __post_init__
    grn_edge_density: float = 0.1
    grn_update_steps: int = 10

    # 6. V(D)J Recombination
    vdj_v_segments: int = 50
    vdj_d_segments: int = 20
    vdj_j_segments: int = 10
    junctional_diversity_max: int = 5

    # 7. Quantum Dream System & Consolidation
    quantum_dream_enabled: bool = True
    quantum_state_dim: int = 128
    dream_consolidation_threshold: float = 0.7
    num_h_qubits: int = 10
    num_v_qubits: int = 10
    rbm_learning_rate: float = 0.1
    rbm_epochs: int = 50
    rbm_batch_size: int = 16
    dream_cycles_per_generation: int = 5
    dream_learning_rate: float = 0.001
    nightmare_adversarial_strength: float = 0.1
    memory_replay_batch_size: int = 64

    # 8. Population Dynamics & Evolution
    initial_population: int = 64  # Renamed from population_size
    max_population: int = 512
    num_generations: int = 50
    selection_pressure: float = 0.4  # Updated value
    mutation_rate: float = 0.02
    crossover_rate: float = 0.1  # Updated value
    elite_size: int = 5
    diversity_weight: float = 0.15
    shannon_entropy_target: float = 0.8
    niche_pressure: float = 0.1

    # 9. Horizontal Gene Transfer
    use_horizontal_transfer: bool = True
    horizontal_transfer_prob: float = 0.002  # Renamed and updated
    plasmid_stability: float = 0.95
    conjugation_efficiency: float = 0.8
    transformation_rate: float = 0.001

    # 10. Environment & Stress
    num_environments: int = 10
    env_complexity: int = 5
    stress_frequency: float = 0.1
    stress_strength: float = 1.5
    stress_window: int = 20
    stress_threshold: float = 0.1
    phase_transition_sensitivity: float = 0.9
    critical_slowing_threshold: float = 0.8

    # 11. Training Parameters
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # 12. Domain-Specific Settings (Immunology Example)
    use_domain_adaptation: bool = True
    antigen_shape_dim: int = 16
    receptor_shape_dim: int = 16
    affinity_threshold: float = 0.95

    # 13. Advanced Feature Flags
    use_meta_learning: bool = True
    use_self_modification: bool = True

    # 14. Logging, Checkpointing, and Visualization
    save_dir: str = "production_results"
    log_file: str = "production_log.txt"
    log_level: str = "INFO"
    plot_interval: int = 5
    checkpoint_interval: int = 5
    seed: int = 42


    ############################# THERAPEUTIC CONFIGURATION ############################################
    num_biomarkers = 50  # cytokines, metabolites, etc.
    critical_biomarkers = ['IL-6', 'TNF-α', 'CRP', 'glucose', 'pH']
    
    # Therapeutic targets
    therapeutic_modes = ['anti-inflammatory', 'immunomodulation', 
                        'metabolic_regulation', 'tissue_repair', 'targeted_killing']
    
    # Safety thresholds
    toxicity_threshold = 0.3
    max_therapeutic_strength = 0.9
    
    # Patient response dynamics
    response_time_constant = 6.0  # hours
    circadian_period = 24.0  # hours







    def __post_init__(self):
        """Validate and initialize derived configuration parameters."""
        # Ensure output directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Combine logic from both __post_init__ methods
        self.grn_num_nodes = self.num_genes + self.num_transposons

        # Validate ODE parameters
        assert self.ode_solver in ["dopri5", "dopri8", "adaptive_heun", "bosh3"], \
            f"Invalid ODE solver: {self.ode_solver}"
        assert 0 < self.min_depth < self.max_depth <= 5.0, "Invalid ODE depth settings"

        # Validate probabilities
        for attr in ['base_transpose_prob', 'methylation_rate', 'mutation_rate', 'horizontal_transfer_prob']:
            if not (0 <= getattr(self, attr) <= 1.0):
                logging.warning(f"Probability {attr} is outside the [0, 1] range: {getattr(self, attr)}")

        # Log the final configuration to a file for reproducibility
        self._log_config()

    def _log_config(self):
        """Logs the configuration to a JSON file in the save directory."""
        config_path = os.path.join(self.save_dir, "config.json")
        # Use a custom encoder to handle non-serializable fields if any
        def custom_encoder(o):
            if isinstance(o, (torch.device)):
                return str(o)
            return o.__dict__

        try:
            with open(config_path, 'w') as f:
                # Convert dataclass to dict for json serialization
                json.dump(self.__dict__, f, indent=2, default=custom_encoder)
            logging.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}", exc_info=True)
            

THERAPY_CFG = TherapeuticConfig()
