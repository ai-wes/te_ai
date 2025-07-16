# File: living_therapeutic_system.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque, defaultdict
import uuid
import random
import time
from torch_geometric.data import Data, Batch

# Import from production system
from scripts.depricated.transposable_immune_ai_production_complete import (
    ContinuousDepthGeneModule, QuantumGeneModule, cfg, ProductionBCell,
    ProductionGerminalCenter
)

# ============================================================================
# THERAPEUTIC DOMAIN CONFIGURATION
# ============================================================================

class TherapeuticConfig:
    """Configuration for living therapeutic system"""
    # Biomarkers and patient state
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

THERAPY_CFG = TherapeuticConfig()

# ============================================================================
# LIVING THERAPEUTIC SEED GENES
# ============================================================================

class TherapeuticSeedGene(ContinuousDepthGeneModule):
    """Base class for therapeutic seed genes"""
    
    def __init__(self, gene_type: str, therapeutic_function: Dict):
        super().__init__(gene_type, 0)
        
        self.therapeutic_function = therapeutic_function
        self.is_seed = True
        # Initialize with device support
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.safety_governor = nn.Parameter(torch.tensor(0.5, device=device))
        
        # Therapeutic output network
        self.therapeutic_network = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.Tanh()  # Bounded therapeutic output
        )
        
        # Patient state encoder
        self.biomarker_encoder = nn.Sequential(
            nn.Linear(THERAPY_CFG.num_biomarkers, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim)
        )
        
        # Safety monitoring
        self.safety_monitor = nn.Sequential(
            nn.Linear(cfg.hidden_dim + THERAPY_CFG.num_biomarkers, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Safety score 0-1
        )

class BiosensorGene(TherapeuticSeedGene):
    """Detects patient state and disease markers"""
    
    def __init__(self):
        super().__init__('BS', {
            'function': 'biosensing',
            'targets': THERAPY_CFG.critical_biomarkers,
            'sensitivity': 'high',
            'response_time': 'fast'
        })
        
        # Specialized sensing networks for different biomarkers
        self.biomarker_detectors = nn.ModuleDict({
            'inflammatory': nn.LSTM(cfg.hidden_dim, 64, batch_first=True),
            'metabolic': nn.GRU(cfg.hidden_dim, 64, batch_first=True),
            'cellular': nn.LSTM(cfg.hidden_dim, 64, batch_first=True)
        })
        
        # Pattern recognition for disease states
        self.disease_classifier = nn.Sequential(
            nn.Linear(192, 128),  # 3 detectors * 64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),  # 10 disease states
            nn.Softmax(dim=-1)
        )
        
        # Temporal tracking
        self.biomarker_history = deque(maxlen=100)
        self.detected_patterns = {}
    
    def sense_patient_state(self, biomarkers: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Comprehensive patient state detection"""
        # Ensure biomarkers are on the same device as the model
        device = next(self.biomarker_encoder.parameters()).device
        biomarkers = biomarkers.to(device)
        
        # Encode biomarkers
        encoded = self.biomarker_encoder(biomarkers)
        
        # Run specialized detectors
        # Ensure encoded has proper dimensions for LSTM/GRU
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
        
        inflammatory, _ = self.biomarker_detectors['inflammatory'](encoded.unsqueeze(0))
        metabolic, _ = self.biomarker_detectors['metabolic'](encoded.unsqueeze(0))
        cellular, _ = self.biomarker_detectors['cellular'](encoded.unsqueeze(0))
        
        # Combine detector outputs - handle different output shapes
        if inflammatory.dim() == 3:
            # Shape: (batch, seq, features)
            combined = torch.cat([
                inflammatory[:, -1, :],
                metabolic[:, -1, :],
                cellular[:, -1, :]
            ], dim=-1)
        else:
            # Shape: (batch, features)
            combined = torch.cat([
                inflammatory,
                metabolic,
                cellular
            ], dim=-1)
        
        # Classify disease state
        disease_probs = self.disease_classifier(combined)
        
        # Detect critical conditions
        critical_markers = self._detect_critical_conditions(biomarkers)
        
        return {
            'encoded_state': encoded,
            'disease_probabilities': disease_probs,
            'critical_conditions': critical_markers,
            'inflammatory_score': inflammatory[:, -1, :].mean(),
            'metabolic_score': metabolic[:, -1, :].mean()
        }
    
    def _detect_critical_conditions(self, biomarkers: torch.Tensor) -> Dict[str, bool]:
        """Detect critical patient conditions requiring immediate response"""
        # Simplified critical detection - in practice would use medical thresholds
        il6_idx = 0  # Would map to actual biomarker indices
        tnf_idx = 1
        glucose_idx = 3
        
        # Handle both 1D and 2D tensors
        if biomarkers.dim() == 2:
            biomarkers = biomarkers[0]  # Take first sample if batched
        
        return {
            'cytokine_storm': biomarkers[il6_idx].item() > 100 and biomarkers[tnf_idx].item() > 50,
            'hypoglycemia': biomarkers[glucose_idx].item() < 70,
            'hyperglycemia': biomarkers[glucose_idx].item() > 200,
            'acidosis': biomarkers[4].item() < 7.35  # pH
        }

class TherapeuticEffectorGene(TherapeuticSeedGene):
    """Produces therapeutic molecules/signals"""
    
    def __init__(self, therapeutic_mode: str):
        super().__init__('TE', {
            'function': 'therapeutic_delivery',
            'mode': therapeutic_mode,
            'potency': 'adaptive',
            'safety': 'high'
        })
        
        self.therapeutic_mode = therapeutic_mode
        
        # Mode-specific therapeutic networks
        self.mode_networks = nn.ModuleDict({
            'anti-inflammatory': self._create_antiinflammatory_network(),
            'immunomodulation': self._create_immunomod_network(),
            'metabolic_regulation': self._create_metabolic_network(),
            'tissue_repair': self._create_repair_network(),
            'targeted_killing': self._create_cytotoxic_network()
        })
        
        # Dose optimization network
        self.dose_controller = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 10, 128),  # +10 for patient features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 dose strength
        )
        
        # Combination therapy coordinator
        self.synergy_network = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh()
        )
        
        # Metabolic LSTM output projection
        self.metabolic_projection = nn.Linear(64, 15)
    
    def _create_antiinflammatory_network(self):
        """Network for anti-inflammatory responses"""
        return nn.Sequential(
            nn.Linear(cfg.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 anti-inflammatory factors
            nn.Softplus()  # Positive concentrations
        )
    
    def _create_immunomod_network(self):
        """Network for immune system modulation"""
        return nn.ModuleDict({
            'suppression': nn.Linear(cfg.hidden_dim, 20),  # Immunosuppressive factors
            'activation': nn.Linear(cfg.hidden_dim, 20),   # Immunostimulatory factors
            'balance': nn.Linear(40, 20)  # Balance between suppression/activation
        })
    
    def _create_metabolic_network(self):
        """Network for metabolic regulation"""
        return nn.LSTM(cfg.hidden_dim, 64, num_layers=2, batch_first=True)
    
    def _create_repair_network(self):
        """Network for tissue repair factors"""
        return nn.Sequential(
            nn.Linear(cfg.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 15)  # Growth factors, etc.
        )
    
    def _create_cytotoxic_network(self):
        """Network for targeted cell killing (e.g., cancer)"""
        return nn.Sequential(
            nn.Linear(cfg.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # Different cytotoxic mechanisms
            nn.Sigmoid()  # Killing probability
        )
    
    def generate_therapeutic(self, patient_state: Dict, target_cells: Optional[torch.Tensor] = None):
        """Generate therapeutic response based on patient state"""
        encoded_state = patient_state['encoded_state']
        
        # Ensure encoded_state is on the same device as the model
        model_device = next(self.parameters()).device
        encoded_state = encoded_state.to(model_device)
        
        # Get mode-specific therapeutic output
        if self.therapeutic_mode in self.mode_networks:
            if self.therapeutic_mode == 'immunomodulation':
                # Special handling for immune modulation
                supp = self.mode_networks[self.therapeutic_mode]['suppression'](encoded_state)
                activ = self.mode_networks[self.therapeutic_mode]['activation'](encoded_state)
                combined = torch.cat([supp, activ], dim=-1)
                therapeutic_output = self.mode_networks[self.therapeutic_mode]['balance'](combined)
            elif self.therapeutic_mode == 'metabolic_regulation':
                # LSTM returns tuple (output, hidden)
                lstm_input = encoded_state.unsqueeze(0) if encoded_state.dim() == 1 else encoded_state
                lstm_input = lstm_input.unsqueeze(0) if lstm_input.dim() == 2 else lstm_input
                therapeutic_output, _ = self.mode_networks[self.therapeutic_mode](lstm_input)
                therapeutic_output = therapeutic_output.squeeze(0).squeeze(0)  # Remove batch and seq dims
                # Project from LSTM hidden size (64) to therapeutic size
                therapeutic_output = self.metabolic_projection(therapeutic_output)
            else:
                therapeutic_output = self.mode_networks[self.therapeutic_mode](encoded_state)
        else:
            therapeutic_output = self.therapeutic_network(encoded_state)
        
        # Calculate optimal dose
        patient_features = self._extract_patient_features(patient_state)
        # Ensure both tensors have same dimensions
        if encoded_state.dim() == 2:
            encoded_state = encoded_state.squeeze(0)  # Remove batch dimension if present
        # Ensure both tensors are on the same device
        dose_input = torch.cat([encoded_state, patient_features.to(encoded_state.device)])
        dose_strength = self.dose_controller(dose_input)
        
        # Apply safety constraints
        safety_score = self._calculate_safety(therapeutic_output, patient_state)
        dose_strength = dose_strength * safety_score
        
        # Scale therapeutic output
        final_therapeutic = therapeutic_output * dose_strength
        
        return {
            'therapeutic': final_therapeutic,
            'dose': dose_strength,
            'safety_score': safety_score,
            'mode': self.therapeutic_mode
        }
    
    def _extract_patient_features(self, patient_state: Dict) -> torch.Tensor:
        """Extract key patient features for dosing decisions"""
        # In practice, would include age, weight, kidney/liver function, etc.
        # Get device from existing tensors in patient_state
        device = torch.device('cpu')
        if 'disease_probabilities' in patient_state and torch.is_tensor(patient_state['disease_probabilities']):
            device = patient_state['disease_probabilities'].device
        features = torch.zeros(10, device=device)
        
        # Disease severity
        features[0] = patient_state['disease_probabilities'].max()
        
        # Inflammatory state
        if 'inflammatory_score' in patient_state:
            features[1] = patient_state['inflammatory_score']
        
        # Critical conditions
        critical = patient_state.get('critical_conditions', {})
        features[2] = float(any(critical.values()))
        
        return features
    
    def _calculate_safety(self, therapeutic: torch.Tensor, patient_state: Dict) -> torch.Tensor:
        """Calculate safety score for therapeutic intervention"""
        # Check for contraindications
        if patient_state.get('critical_conditions', {}).get('cytokine_storm', False):
            if self.therapeutic_mode == 'immunomodulation':
                # Return tensor on same device as the model
                device = next(self.parameters()).device
                return torch.tensor(0.3, device=device)  # Reduce dose during cytokine storm
        
        # General safety calculation
        # Get device from model parameters
        device = next(self.parameters()).device
        biomarkers = patient_state.get('biomarkers', torch.zeros(THERAPY_CFG.num_biomarkers, device=device))
        # Use encoded_state instead of therapeutic output for safety calculation
        encoded_state = patient_state.get('encoded_state', torch.zeros(cfg.hidden_dim, device=device))
        if encoded_state.dim() > 1:
            encoded_state = encoded_state.squeeze(0)
        # Ensure both tensors are on the same device
        safety_input = torch.cat([encoded_state, biomarkers.to(encoded_state.device)])
        
        return self.safety_monitor(safety_input)

class AdaptiveControllerGene(TherapeuticSeedGene):
    """Controls and coordinates therapeutic response"""
    
    def __init__(self):
        super().__init__('AC', {
            'function': 'adaptive_control',
            'strategy': 'multi-objective',
            'learning': 'online',
            'safety': 'paramount'
        })
        
        # Multi-objective optimization
        # Initialize objective weights with device support
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.objective_weights = nn.Parameter(torch.tensor([
            0.4,  # Efficacy
            0.3,  # Safety
            0.2,  # Speed
            0.1   # Efficiency
        ], device=device))
        
        # Temporal planning network
        self.treatment_planner = nn.LSTM(
            cfg.hidden_dim + THERAPY_CFG.num_biomarkers,
            cfg.hidden_dim,
            num_layers=3,
            batch_first=True
        )
        
        # Circadian rhythm modulator
        self.circadian_modulator = nn.Sequential(
            nn.Linear(1, 32),  # Time input
            nn.ReLU(),  # Changed from Sin() which doesn't exist
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0-1 circadian factor
        )
        
        # Learning from patient response
        self.response_memory = deque(maxlen=1000)
        self.meta_learner = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 4)  # Update objective weights
        )
    
    def plan_treatment(self, patient_history: List[Dict], current_state: Dict, 
                      horizon: int = 24) -> Dict:
        """Plan treatment strategy over time horizon"""
        # Prepare sequence data
        history_tensor = self._prepare_history(patient_history)
        
        # Generate treatment plan
        hidden = None
        treatment_plan = []
        
        for t in range(horizon):
            # Add time features
            time_of_day = (t % 24) / 24.0
            # Create tensor on correct device
            device = next(self.circadian_modulator.parameters()).device
            circadian_factor = self.circadian_modulator(torch.tensor([[time_of_day]], device=device))
            
            # Plan next treatment step
            # Get treatment planner device
            planner_device = next(self.treatment_planner.parameters()).device
            
            if hidden is None:
                history_tensor = history_tensor.to(planner_device)
                output, hidden = self.treatment_planner(history_tensor)
            else:
                # Continue from previous hidden state
                encoded = current_state['encoded_state']
                if encoded.dim() > 1:
                    encoded = encoded.squeeze(0)
                # Ensure biomarkers are on the same device as encoded
                biomarkers = current_state.get('biomarkers', torch.zeros(THERAPY_CFG.num_biomarkers, device=encoded.device))
                if torch.is_tensor(biomarkers):
                    biomarkers = biomarkers.to(encoded.device)
                current_input = torch.cat([
                    encoded,
                    biomarkers
                ]).unsqueeze(0).unsqueeze(0)
                
                # Move to planner device
                current_input = current_input.to(planner_device)
                if hidden is not None:
                    hidden = (hidden[0].to(planner_device), hidden[1].to(planner_device))
                
                output, hidden = self.treatment_planner(current_input, hidden)
            
            # Modulate by circadian rhythm
            treatment_output = output[:, -1, :] * circadian_factor
            
            treatment_plan.append({
                'time': t,
                'treatment_vector': treatment_output,
                'circadian_factor': circadian_factor.item()
            })
        
        return {
            'plan': treatment_plan,
            'objectives': self.objective_weights,
            'horizon': horizon
        }
    
    def learn_from_response(self, treatment: Dict, response: Dict):
        """Learn from patient response to treatment"""
        # Store in memory
        self.response_memory.append({
            'treatment': treatment,
            'response': response,
            'outcome': self._evaluate_outcome(response)
        })
        
        # Meta-learning update
        if len(self.response_memory) >= 10:
            # Analyze recent responses
            recent = list(self.response_memory)[-10:]
            
            # Extract features
            # Ensure all tensors are on the same device
            device = next(self.parameters()).device
            treatment_tensors = [r['treatment']['therapeutic'].to(device) for r in recent]
            treatment_features = torch.stack(treatment_tensors).mean(0)
            
            response_tensors = [r['response']['biomarker_change'].to(device) for r in recent]
            response_features = torch.stack(response_tensors).mean(0)
            # Create tensor on model device
            device = next(self.parameters()).device
            outcome_scores = torch.tensor([r['outcome'] for r in recent], device=device).mean()
            
            # Meta-learn better objective weights
            meta_input = torch.cat([
                treatment_features,
                response_features,
                self.objective_weights
            ])
            
            weight_updates = self.meta_learner(meta_input)
            
            # Update objective weights
            with torch.no_grad():
                self.objective_weights.data += 0.01 * weight_updates
                self.objective_weights.data = F.softmax(self.objective_weights, dim=0)
    
    def _prepare_history(self, patient_history: List[Dict]) -> torch.Tensor:
        """Prepare patient history for sequential processing"""
        if not patient_history:
            # Return default tensor if no history
            # Return tensor on model device
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return torch.zeros(1, 1, cfg.hidden_dim + THERAPY_CFG.num_biomarkers, device=device)
        
        history_tensors = []
        device = torch.device('cpu')  # Default device
        
        for record in patient_history[-24:]:  # Last 24 hours
            state = record.get('encoded_state', torch.zeros(cfg.hidden_dim))
            biomarkers = record.get('biomarkers', torch.zeros(THERAPY_CFG.num_biomarkers))
            
            # Get device from first tensor found
            if torch.is_tensor(state):
                device = state.device
            elif torch.is_tensor(biomarkers):
                device = biomarkers.device
                
            # Ensure both tensors are on the same device
            if not torch.is_tensor(state):
                state = torch.tensor(state, device=device)
            else:
                state = state.to(device)
                
            if not torch.is_tensor(biomarkers):
                biomarkers = torch.tensor(biomarkers, device=device)
            else:
                biomarkers = biomarkers.to(device)
                
            combined = torch.cat([state, biomarkers])
            history_tensors.append(combined)
        
        return torch.stack(history_tensors).unsqueeze(0)
    
    def _evaluate_outcome(self, response: Dict) -> float:
        """Evaluate treatment outcome"""
        # Multi-objective evaluation
        efficacy = response.get('efficacy_score', 0.5)
        safety = response.get('safety_score', 0.5)
        speed = response.get('response_speed', 0.5)
        efficiency = response.get('efficiency_score', 0.5)
        
        # Create scores tensor on same device as objective_weights
        device = self.objective_weights.device
        scores = torch.tensor([efficacy, safety, speed, efficiency], device=device)
        
        return (self.objective_weights * scores).sum().item()

# ============================================================================
# THERAPEUTIC STEM GENE
# ============================================================================

class TherapeuticStemGene(ContinuousDepthGeneModule):
    """Stem gene that can differentiate into any therapeutic function"""
    
    def __init__(self):
        super().__init__('TS', 0)  # Therapeutic Stem
        
        # Differentiation state (can be partial)
        # Initialize with device support
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.differentiation_state = nn.Parameter(torch.zeros(5, device=device))  # [BS, TE, AC, Mixed, Stem]
        self.is_differentiated = False
        
        # Learning from seed genes
        self.observed_seeds = {}
        self.skill_memory = nn.Parameter(torch.zeros(20, cfg.hidden_dim, device=device))
        
        # Meta-network for differentiation decisions
        # Input size: encoded_state (128) + need_scores (3) + seed_influence (128) + skill_memory (128) = 387
        self.differentiation_network = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3 + 3, 256),  # 128*3 + 3 = 387
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Softmax(dim=-1)
        )
        
        # Morphable therapeutic components
        self.stem_components = nn.ModuleDict({
            'sensing': nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            'effector': nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            'control': nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            'universal': nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        })
        
        # Plasticity and commitment
        self.plasticity = nn.Parameter(torch.tensor(1.0, device=device))
        self.commitment_level = 0.0
        
        # Niche sensing
        self.niche_sensor = nn.Sequential(
            nn.Linear(THERAPY_CFG.num_biomarkers + 10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Niche scores for BS, TE, AC
        )
    
    def sense_therapeutic_needs(self, patient_state: Dict, population_state: Dict) -> Dict:
        """Analyze what therapeutic function is most needed"""
        # Count existing therapeutic genes
        gene_counts = {'BS': 0, 'TE': 0, 'AC': 0}
        gene_performance = {'BS': [], 'TE': [], 'AC': []}
        
        for cell in population_state.get('cells', []):
            for gene in cell.genes:
                if hasattr(gene, 'therapeutic_function'):
                    gene_type = gene.gene_type
                    if gene_type in gene_counts:
                        gene_counts[gene_type] += 1
                        gene_performance[gene_type].append(gene.fitness_contribution)
        
        # Analyze patient needs
        # Get device from model parameters
        device = next(self.parameters()).device
        biomarkers = patient_state.get('biomarkers', torch.zeros(THERAPY_CFG.num_biomarkers, device=device))
        critical = patient_state.get('critical_conditions', {})
        
        # Calculate need scores
        device = biomarkers.device if torch.is_tensor(biomarkers) else torch.device('cpu')
        need_features = torch.cat([
            biomarkers if torch.is_tensor(biomarkers) else torch.tensor(biomarkers, device=device),
            torch.tensor([
                float(any(critical.values())),  # Any critical condition
                float(gene_counts['BS'] == 0),  # No sensors
                float(gene_counts['TE'] < 2),    # Few effectors
                float(gene_counts['AC'] == 0),   # No controller
                patient_state.get('disease_severity', 0.5),
                patient_state.get('treatment_resistance', 0.0),
                len(population_state.get('cells', [])) / 100.0,  # Population size
                population_state.get('avg_fitness', 0.5),
                population_state.get('diversity', 0.5),
                float(self.commitment_level < 0.5)  # Still plastic
            ], device=device)
        ])
        
        # Ensure need_features is on the same device as the niche_sensor
        sensor_device = next(self.niche_sensor.parameters()).device
        need_scores = self.niche_sensor(need_features.float().to(sensor_device))
        
        return {
            'gene_counts': gene_counts,
            'gene_performance': gene_performance,
            'need_scores': need_scores,
            'recommended_type': ['BS', 'TE', 'AC'][need_scores.argmax()]
        }
    
    def guided_differentiation(self, patient_state: Dict, population_state: Dict):
        """Differentiate based on therapeutic needs"""
        needs = self.sense_therapeutic_needs(patient_state, population_state)
        
        # Learn from successful seeds
        seed_influence = self._learn_from_seeds(population_state)
        
        # Combine patient needs and seed learning
        # Ensure all tensors are 1D before concatenation and on the same device
        device = self.skill_memory.device  # Use skill_memory's device as reference
        
        encoded_state = patient_state['encoded_state']
        encoded_state = encoded_state.flatten() if encoded_state.dim() > 1 else encoded_state
        encoded_state = encoded_state.to(device)
        
        need_scores = needs['need_scores']
        need_scores = need_scores.flatten() if need_scores.dim() > 1 else need_scores
        need_scores = need_scores.to(device)
        
        seed_influence_flat = seed_influence.flatten() if seed_influence.dim() > 1 else seed_influence
        seed_influence_flat = seed_influence_flat.to(device)
        
        skill_mean = self.skill_memory.mean(dim=0)
        skill_mean = skill_mean.flatten() if skill_mean.dim() > 1 else skill_mean
        
        differentiation_input = torch.cat([
            encoded_state,
            need_scores,
            seed_influence_flat,
            skill_mean
        ])
        
        # Ensure differentiation_input is on the same device as the network
        network_device = next(self.differentiation_network.parameters()).device
        differentiation_input = differentiation_input.to(network_device)
        
        # Decide differentiation
        diff_probs = self.differentiation_network(differentiation_input)
        
        # Stochastic differentiation with commitment
        if random.random() < self.plasticity.item():
            if needs['recommended_type'] and random.random() < 0.7:
                # Follow recommendation
                type_map = {'BS': 0, 'TE': 1, 'AC': 2}
                target_idx = type_map.get(needs['recommended_type'], 4)
                
                # Partial differentiation
                commitment_strength = min(0.8, self.commitment_level + 0.2)
                self.differentiation_state.data = torch.zeros(5, device=self.differentiation_state.device)
                self.differentiation_state.data[target_idx] = commitment_strength
                self.differentiation_state.data[4] = 1 - commitment_strength  # Remain partly stem
                
                self.commitment_level = commitment_strength
                
                print(f"   💊 Therapeutic stem gene differentiating toward {needs['recommended_type']} "
                      f"(commitment: {commitment_strength:.2f})")
            else:
                # Explore novel combination
                self._explore_hybrid_therapeutic(diff_probs)
    
    def _learn_from_seeds(self, population_state: Dict) -> torch.Tensor:
        """Learn from successful seed genes"""
        seed_features = []
        
        for cell in population_state.get('cells', []):
            for gene in cell.genes:
                if hasattr(gene, 'is_seed') and gene.is_seed:
                    # Store successful patterns
                    if gene.fitness_contribution > 0.7:
                        self.observed_seeds[gene.gene_type] = {
                            'function': gene.therapeutic_function,
                            'performance': gene.fitness_contribution,
                            'features': gene.morphogen_field.detach() if hasattr(gene, 'morphogen_field') else None
                        }
                    
                    device = next(self.parameters()).device
                    seed_features.append(gene.morphogen_field if hasattr(gene, 'morphogen_field') 
                                       else torch.randn(cfg.hidden_dim, device=device))
        
        if seed_features:
            return torch.stack(seed_features).mean(0)
        device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
        return torch.zeros(cfg.hidden_dim, device=device)
    
    def _explore_hybrid_therapeutic(self, diff_probs: torch.Tensor):
        """Create novel therapeutic combination"""
        # Sample from distribution
        chosen = torch.multinomial(diff_probs, 2)
        
        # Create hybrid differentiation
        self.differentiation_state.data = torch.zeros(5, device=self.differentiation_state.device)
        self.differentiation_state.data[chosen[0]] = 0.4
        self.differentiation_state.data[chosen[1]] = 0.3
        self.differentiation_state.data[3] = 0.2  # Mixed
        self.differentiation_state.data[4] = 0.1  # Stem
        
        print(f"   🧬 Exploring hybrid therapeutic: {chosen.tolist()}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process with differentiation-weighted components"""
        # Get base processing
        h = self.input_projection(x)
        
        # Apply differentiation-specific processing
        diff_weights = F.softmax(self.differentiation_state, dim=0)
        
        # Weighted combination of therapeutic components
        h_therapeutic = (
            diff_weights[0] * self.stem_components['sensing'](h) +    # Biosensor
            diff_weights[1] * self.stem_components['effector'](h) +   # Effector
            diff_weights[2] * self.stem_components['control'](h) +    # Controller
            diff_weights[3] * (self.stem_components['sensing'](h) +   # Mixed
                              self.stem_components['effector'](h)) / 2 +
            diff_weights[4] * self.stem_components['universal'](h)    # Stem
        )
        
        # Continue with ODE processing
        return super().forward_from_hidden(h_therapeutic, edge_index, batch)
    
    def emergency_response(self, critical_condition: str):
        """Rapid differentiation in response to critical patient condition"""
        print(f"   🚨 EMERGENCY: {critical_condition} detected!")
        
        # Immediate differentiation based on condition
        emergency_map = {
            'cytokine_storm': 'TE',  # Need therapeutic effector
            'organ_failure': 'AC',    # Need adaptive control
            'unknown_pathogen': 'BS', # Need biosensing
            'treatment_resistance': 'Mixed'  # Need novel approach
        }
        
        target_type = emergency_map.get(critical_condition, 'TE')
        
        # Force rapid differentiation
        with torch.no_grad():
            self.commitment_level = 0.9
            self.plasticity.data = torch.tensor(0.1, device=self.plasticity.device)  # Lock in
            
            type_map = {'BS': 0, 'TE': 1, 'AC': 2, 'Mixed': 3}
            self.differentiation_state.data = torch.zeros(5, device=self.differentiation_state.device)
            self.differentiation_state.data[type_map.get(target_type, 1)] = 0.9
            self.differentiation_state.data[4] = 0.1

# ============================================================================
# INTEGRATION WITH TE-AI SYSTEM
# ============================================================================

def create_living_therapeutic_population(patient_profile: Dict) -> Dict:
    """Initialize population for specific patient"""
    population = {}
    
    # Analyze patient to determine initial composition
    disease_type = patient_profile.get('disease', 'unknown')
    severity = patient_profile.get('severity', 0.5)
    
    # Create founder population
    for i in range(cfg.initial_population):
        genes = []
        
        # Seed genes based on disease
        if i < 10:  # Biosensors
            genes.append(BiosensorGene())
        elif i < 25:  # Therapeutic effectors
            mode = random.choice(THERAPY_CFG.therapeutic_modes)
            genes.append(TherapeuticEffectorGene(mode))
        elif i < 30:  # Controllers
            genes.append(AdaptiveControllerGene())
        
        # Add stem genes
        num_stem = random.randint(2, 5)
        for _ in range(num_stem):
            genes.append(TherapeuticStemGene())
        
        # Add some quantum genes for exploration
        if random.random() < 0.1:
            genes.append(QuantumGeneModule('QT', 0))
        
        # Create therapeutic cell
        from scripts.depricated.transposable_immune_ai_production_complete import ProductionBCell
        cell = ProductionBCell(genes)
        cell.therapeutic_profile = {
            'specialization': 'therapeutic',
            'patient_id': patient_profile.get('id'),
            'disease_target': disease_type
        }
        
        population[cell.cell_id] = cell
    
    return population

# ============================================================================
# THERAPEUTIC EVOLUTION CYCLE
# ============================================================================

class LivingTherapeuticSystem:
    """Complete living therapeutic system that evolves with patient"""
    
    def __init__(self, patient_profile: Dict):
        self.patient_profile = patient_profile
        self.population = create_living_therapeutic_population(patient_profile)
        
        # Patient state tracking
        self.patient_history = deque(maxlen=1000)
        self.treatment_history = deque(maxlen=1000)
        
        # Initialize germinal center
        from scripts.depricated.transposable_immune_ai_production_complete import ProductionGerminalCenter
        self.germinal_center = ProductionGerminalCenter()
        self.germinal_center.population = self.population
        
    def therapeutic_cycle(self, current_biomarkers: torch.Tensor) -> Dict:
        """One cycle of sense → plan → treat → evolve"""
        
        # 1. Sense patient state
        patient_state = self._comprehensive_patient_assessment(current_biomarkers)
        
        # 2. Population generates therapeutic response
        therapeutic_response = self._generate_population_response(patient_state)
        
        # 3. Apply treatment and simulate patient response
        patient_response = self._simulate_patient_response(therapeutic_response)
        
        # 4. Evolve based on treatment success
        self._evolve_therapeutic_population(patient_response)
        
        # 5. Record history
        self.patient_history.append(patient_state)
        self.treatment_history.append(therapeutic_response)
        
        return {
            'patient_state': patient_state,
            'treatment': therapeutic_response,
            'response': patient_response,
            'population_size': len(self.population),
            'avg_fitness': np.mean([c.fitness_history[-1] if c.fitness_history else 0 
                                   for c in self.population.values()])
        }
    
    def _comprehensive_patient_assessment(self, biomarkers: torch.Tensor) -> Dict:
        """Comprehensive patient state assessment"""
        patient_state = {
            'biomarkers': biomarkers,
            'timestamp': time.time()
        }
        
        # Run all biosensor genes
        sensor_outputs = []
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, BiosensorGene):
                    sensor_output = gene.sense_patient_state(biomarkers)
                    sensor_outputs.append(sensor_output)
        
        # Aggregate sensor data
        if sensor_outputs:
            # Get device from biomarkers or use default
            device = biomarkers.device if torch.is_tensor(biomarkers) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Stack encoded states on same device
            encoded_states = [s['encoded_state'].to(device) for s in sensor_outputs]
            patient_state['encoded_state'] = torch.stack(encoded_states).mean(0)
            
            # Stack disease probabilities on same device
            disease_probs = [s['disease_probabilities'].to(device) for s in sensor_outputs]
            patient_state['disease_probabilities'] = torch.stack(disease_probs).mean(0)
            
            # Check for critical conditions
            critical_conditions = {}
            for output in sensor_outputs:
                for condition, detected in output['critical_conditions'].items():
                    if detected:
                        critical_conditions[condition] = True
            
            patient_state['critical_conditions'] = critical_conditions
            
            # Calculate severity
            patient_state['disease_severity'] = patient_state['disease_probabilities'].max().item()
        else:
            # No sensors yet - create default values
            device = biomarkers.device
            patient_state['encoded_state'] = torch.zeros(cfg.hidden_dim, device=device)
            patient_state['disease_probabilities'] = torch.zeros(3, device=device)
            patient_state['critical_conditions'] = {}
            patient_state['disease_severity'] = 0.5  # Default moderate severity
        
        return patient_state
    
    def _generate_population_response(self, patient_state: Dict) -> Dict:
        """Generate coordinated therapeutic response from population"""
        therapeutic_outputs = []
        controller_plans = []
        
        for cell in self.population.values():
            for gene in cell.genes:
                # Therapeutic effectors generate treatments
                if isinstance(gene, TherapeuticEffectorGene):
                    therapeutic = gene.generate_therapeutic(patient_state)
                    therapeutic_outputs.append(therapeutic)
                
                # Controllers create treatment plans
                elif isinstance(gene, AdaptiveControllerGene):
                    plan = gene.plan_treatment(
                        list(self.patient_history)[-24:],
                        patient_state
                    )
                    controller_plans.append(plan)
                
                # Stem genes might differentiate
                elif isinstance(gene, TherapeuticStemGene):
                    population_state = {
                        'cells': list(self.population.values()),
                        'avg_fitness': np.mean([c.fitness_history[-1] if c.fitness_history else 0 
                                              for c in self.population.values()]),
                        'diversity': len(set(g.gene_type for c in self.population.values() 
                                           for g in c.genes))
                    }
                    gene.guided_differentiation(patient_state, population_state)
        
        # Combine therapeutic outputs
        if therapeutic_outputs:
            combined_therapeutic = self._combine_therapeutics(therapeutic_outputs)
        else:
            # Create empty therapeutic on correct device
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            combined_therapeutic = {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0}
        
        # Use best treatment plan
        if controller_plans:
            best_plan = max(controller_plans, 
                          key=lambda p: p.get('objectives', [0])[0])  # Sort by efficacy
        else:
            best_plan = {'plan': [], 'horizon': 24}
        
        return {
            'therapeutic': combined_therapeutic,
            'plan': best_plan,
            'num_effectors': len(therapeutic_outputs),
            'num_controllers': len(controller_plans)
        }
    
    def _combine_therapeutics(self, therapeutic_outputs: List[Dict]) -> Dict:
        """Intelligently combine multiple therapeutic outputs"""
        if not therapeutic_outputs:
            # Return empty therapeutic on correct device
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return {'therapeutic': torch.zeros(20, device=device), 'dose': 0.0, 'safety_score': 1.0}
        
        # Group by mode
        by_mode = defaultdict(list)
        for output in therapeutic_outputs:
            by_mode[output['mode']].append(output)
        
        # Find max therapeutic size across all outputs
        # Handle both 1D and 2D tensors
        max_size = 0
        for output in therapeutic_outputs:
            therapeutic = output['therapeutic']
            if therapeutic.dim() == 2:
                size = therapeutic.shape[1]  # For [1, X] tensors, use the second dimension
            else:
                size = therapeutic.shape[0]  # For [X] tensors, use the first dimension
            max_size = max(max_size, size)
        # Max therapeutic size determined
        
        # Combine within modes (avoid overdosing)
        # Get device from first therapeutic tensor
        device = therapeutic_outputs[0]['therapeutic'].device if therapeutic_outputs else torch.device('cpu')
        combined = {
            'therapeutic': torch.zeros(max_size, device=device),
            'dose': 0.0,
            'safety_score': 1.0
        }
        
        for mode, outputs in by_mode.items():
            # Process this mode
            
            # Average therapeutics of same type
            # Get device from first therapeutic
            device = outputs[0]['therapeutic'].device
            
            # Ensure all tensors are on same device
            therapeutic_tensors = [o['therapeutic'].to(device) for o in outputs]
            mode_therapeutic = torch.stack(therapeutic_tensors).mean(0)
            
            dose_tensors = [o['dose'].to(device) if torch.is_tensor(o['dose']) else torch.tensor(o['dose'], device=device) for o in outputs]
            mode_dose = torch.stack(dose_tensors).mean()
            
            safety_tensors = [o['safety_score'].to(device) if torch.is_tensor(o['safety_score']) else torch.tensor(o['safety_score'], device=device) for o in outputs]
            mode_safety = torch.stack(safety_tensors).min()
            
            # Average computed
            
            # Add to combined with safety limits
            safe_dose = min(mode_dose.item(), THERAPY_CFG.max_therapeutic_strength)
            # Safe dose determined
            
            # Handle different therapeutic sizes by padding or truncating
            if mode_therapeutic.dim() > 1:
                # Flatten if needed
                mode_therapeutic = mode_therapeutic.squeeze()
            
            # Scale by safe dose
            scaled_therapeutic = mode_therapeutic * safe_dose
            
            if mode_therapeutic.shape[0] <= max_size:
                # Pad with zeros if needed
                if mode_therapeutic.shape[0] < max_size:
                    padded = torch.zeros(max_size, device=mode_therapeutic.device)
                    # Pad to max size
                    padded[:mode_therapeutic.shape[0]] = scaled_therapeutic
                    combined['therapeutic'] += padded
                else:
                    # Exact size match
                    # Size matches
                    combined['therapeutic'] += scaled_therapeutic
            else:
                # Therapeutic is larger than max_size - we need to truncate
                # Truncate to max size
                combined['therapeutic'] += scaled_therapeutic[:max_size]
            
            combined['dose'] += safe_dose / len(by_mode)  # Normalize by number of modes
            combined['safety_score'] *= mode_safety
        
        return combined
    
    def _simulate_patient_response(self, treatment: Dict) -> Dict:
        """Simulate how patient responds to treatment"""
        # Simplified patient response model
        therapeutic = treatment['therapeutic']['therapeutic']
        dose = treatment['therapeutic']['dose']
        
        # Calculate therapeutic effect
        base_effect = (therapeutic.sum() * dose).item()
        
        # Add variability based on patient
        patient_variability = random.gauss(1.0, 0.2)
        actual_effect = base_effect * patient_variability
        
        # Determine response
        response = {
            'efficacy_score': min(1.0, actual_effect / 10.0),
            'safety_score': treatment['therapeutic']['safety_score'].item(),
            'response_speed': random.uniform(0.3, 0.9),
            'efficiency_score': dose.item(),  # Lower dose = more efficient
            'biomarker_change': torch.randn(THERAPY_CFG.num_biomarkers, device=treatment['therapeutic']['therapeutic'].device) * actual_effect
        }
        
        # Check for adverse events
        if random.random() < (1 - response['safety_score']) * 0.1:
            response['adverse_event'] = True
            response['efficacy_score'] *= 0.5
        
        return response
    
    def _evolve_therapeutic_population(self, patient_response: Dict):
        """Evolve population based on treatment success"""
        # Calculate fitness for each cell based on contribution
        for cell in self.population.values():
            # Base fitness on treatment outcome
            cell_fitness = patient_response['efficacy_score'] * patient_response['safety_score']
            
            # Bonus for cells with therapeutic genes
            therapeutic_bonus = sum(
                0.1 for gene in cell.genes 
                if isinstance(gene, (BiosensorGene, TherapeuticEffectorGene, AdaptiveControllerGene))
            )
            
            cell_fitness += therapeutic_bonus
            cell.fitness_history.append(cell_fitness)
        
        # Trigger evolution if response is poor
        if patient_response['efficacy_score'] < 0.5:
            print("   Poor treatment response - triggering evolution!")
            stress_level = 1.0 - patient_response['efficacy_score']
            
            # Increase transposition
            for cell in list(self.population.values())[:50]:
                for gene in cell.genes:
                    if gene.is_active:
                        child = gene.transpose(stress_level, 0.5)
                        if child:
                            # Add to population if successful
                            pass
        
        # Learn from response
        for cell in self.population.values():
            for gene in cell.genes:
                if isinstance(gene, AdaptiveControllerGene):
                    gene.learn_from_response(
                        treatment['therapeutic'],
                        patient_response
                    )

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def run_living_therapeutic_simulation():
    """Example of living therapeutic in action"""
    
    # Patient profile
    patient = {
        'id': 'PT001',
        'disease': 'autoimmune_inflammatory',
        'severity': 0.7,
        'age': 45,
        'weight': 70,
        'comorbidities': ['diabetes']
    }
    
    # Initialize therapeutic system
    therapeutic_system = LivingTherapeuticSystem(patient)
    
    # Simulate treatment over time
    for hour in range(168):  # One week
        # Generate patient biomarkers (would come from real sensors)
        # Generate biomarkers on correct device
        device = next(therapeutic_system.parameters()).device
        biomarkers = torch.randn(THERAPY_CFG.num_biomarkers, device=device)
        
        # Inflammatory markers elevated
        if hour < 48:
            biomarkers[0] = 5.0  # IL-6
            biomarkers[1] = 3.0  # TNF-α
        
        # Run therapeutic cycle
        result = therapeutic_system.therapeutic_cycle(biomarkers)
        
        if hour % 24 == 0:
            print(f"\nDay {hour // 24}:")
            print(f"  Patient severity: {result['patient_state'].get('disease_severity', 0):.3f}")
            print(f"  Treatment efficacy: {result['response']['efficacy_score']:.3f}")
            print(f"  Population size: {result['population_size']}")
            print(f"  Average fitness: {result['avg_fitness']:.3f}")
        
        # Check for critical conditions
        if result['patient_state'].get('critical_conditions'):
            print(f"  ⚠️ Critical condition detected: {list(result['patient_state']['critical_conditions'].keys())}")
            
            # Trigger emergency response in stem genes
            for cell in therapeutic_system.population.values():
                for gene in cell.genes:
                    if isinstance(gene, TherapeuticStemGene):
                        condition = list(result['patient_state']['critical_conditions'].keys())[0]
                        gene.emergency_response(condition)

if __name__ == "__main__":
    print("🧬 Living Therapeutic TE-AI System")
    print("=" * 50)
    run_living_therapeutic_simulation()