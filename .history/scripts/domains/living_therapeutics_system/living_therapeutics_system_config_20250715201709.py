# File: living_therapeutics_system_config.py
"""
Configuration for Living Therapeutic System
==========================================
All configuration constants and settings for the therapeutic system
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional

# ============================================================================
# THERAPEUTIC DOMAIN CONFIGURATION
# ============================================================================

class TherapeuticConfig:
    """Configuration for living therapeutic system"""
    # Biomarkers and patient state
    num_biomarkers = 50  # cytokines, metabolites, etc.
    critical_biomarkers = ['IL-6', 'TNF-alpha', 'CRP', 'glucose', 'pH']
    
    # Therapeutic targets
    therapeutic_modes = ['anti-inflammatory', 'immunomodulation', 
                        'metabolic_regulation', 'tissue_repair', 'targeted_killing']
    
    # Safety thresholds
    toxicity_threshold = 0.3
    max_therapeutic_strength = 0.9
    
    # Patient response dynamics
    response_time_constant = 6.0  # hours
    circadian_period = 24.0  # hours

# Global configuration instance
THERAPY_CFG = TherapeuticConfig()

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device():
    """Get the appropriate device for tensor operations"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# THERAPEUTIC NETWORK ARCHITECTURES
# ============================================================================

@dataclass
class NetworkConfig:
    """Neural network architecture configurations"""
    hidden_dim: int = 128
    num_layers: int = 3
    dropout_rate: float = 0.1
    activation: str = 'relu'

# Network configurations for different therapeutic modes
NETWORK_CONFIGS = {
    'biosensor': NetworkConfig(hidden_dim=64, num_layers=2),
    'effector': NetworkConfig(hidden_dim=128, num_layers=3),
    'controller': NetworkConfig(hidden_dim=256, num_layers=4),
    'stem': NetworkConfig(hidden_dim=512, num_layers=5)
}