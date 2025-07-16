# File: __init__.py
"""
Living Therapeutic System Domain
===============================
Consolidated therapeutic system with clean, organized architecture

This domain provides a complete living therapeutic system that can:
- Sense patient state through biosensor genes
- Generate therapeutic responses through effector genes  
- Coordinate treatment through adaptive controller genes
- Evolve and adapt through stem genes

Architecture:
- config: Configuration and constants
- genes: All therapeutic gene classes
- cells: Cell management and population functions
- main: Main system orchestration (LivingTherapeuticSystem)
- run: Runners, examples, and CLI interface
"""

# Import main components for easy access
from .living_therapeutics_system_config import (
    THERAPY_CFG, 
    TherapeuticConfig,
    get_device,
    NETWORK_CONFIGS
)

from .living_therapeutics_system_genes import (
    # Base class
    TherapeuticSeedGene,
    
    # Specialized genes
    BiosensorGene,
    TherapeuticEffectorGene, 
    AdaptiveControllerGene,
    TherapeuticStemGene
)

# Cells module functions will be defined in main module for now
from .living_therapeutics_system_main import (
    LivingTherapeuticSystem as TherapeuticCellManager
)

from .living_therapeutics_system_main import (
    LivingTherapeuticSystem
)

from .living_therapeutics_system_run import (
    run_basic_therapeutic_simulation,
    run_enhanced_therapeutic_demo,
    ProductionTherapeuticRunner,
    run_validation_suite
)

# Package metadata
__version__ = "1.0.0"
__author__ = "TE-AI System"
__description__ = "Living Therapeutic System - Consolidated Domain Architecture"

# Define what gets imported with "from living_therapeutics_system import *"
__all__ = [
    # Configuration
    'THERAPY_CFG',
    'TherapeuticConfig', 
    'get_device',
    
    # Genes
    'TherapeuticSeedGene',
    'BiosensorGene',
    'TherapeuticEffectorGene',
    'AdaptiveControllerGene', 
    'TherapeuticStemGene',
    
    # Main System
    'TherapeuticCellManager',
    
    # Main System
    'LivingTherapeuticSystem',
    
    # Runners
    'run_basic_therapeutic_simulation',
    'run_therapeutic_with_visualization',
    'run_enhanced_therapeutic_demo',
    'ProductionTherapeuticRunner',
    'run_validation_suite'
]

# Convenience functions
def create_patient_profile(disease_type: str, severity: float = 0.7, **kwargs):
    """Create a standardized patient profile"""
    profiles = {
        'autoimmune': {
            'disease': 'autoimmune_inflammatory',
            'age': kwargs.get('age', 45),
            'comorbidities': kwargs.get('comorbidities', ['diabetes'])
        },
        'cancer': {
            'disease': 'cancer', 
            'age': kwargs.get('age', 60),
            'comorbidities': kwargs.get('comorbidities', [])
        },
        'metabolic': {
            'disease': 'metabolic_syndrome',
            'age': kwargs.get('age', 50), 
            'comorbidities': kwargs.get('comorbidities', ['hypertension'])
        }
    }
    
    base_profile = profiles.get(disease_type, profiles['autoimmune'])
    base_profile.update({
        'id': kwargs.get('id', f'PT_{disease_type.upper()[:3]}'),
        'severity': severity
    })
    base_profile.update(kwargs)
    
    return base_profile

def quick_start(disease_type: str = 'autoimmune', hours: int = 24):
    """Quick start function for rapid testing"""
    print("🚀 Living Therapeutic System - Quick Start")
    
    # Create patient
    patient = create_patient_profile(disease_type)
    
    # Run basic simulation
    return run_basic_therapeutic_simulation(patient, hours)

# Domain status
def get_domain_info():
    """Get information about this domain"""
    return {
        'name': 'Living Therapeutic System',
        'version': __version__,
        'components': {
            'config': 'Configuration and constants',
            'genes': 'Therapeutic gene classes (BS, TE, AC, TS)',
            'cells': 'Cell management and population functions', 
            'main': 'Main system orchestration',
            'run': 'Runners and execution interfaces'
        },
        'capabilities': [
            'Patient state sensing and monitoring',
            'Adaptive therapeutic response generation',
            'Multi-objective treatment optimization',
            'Population evolution and learning',
            'Emergency response and intervention',
            'Treatment resistance adaptation'
        ]
    }