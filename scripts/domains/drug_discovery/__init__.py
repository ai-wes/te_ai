"""
Drug Discovery Domain for TE-AI
================================

This module implements drug target narrowing and prioritization using the
Transposable Element AI architecture. It evolves populations of cells to
identify and rank potential drug targets based on druggability, specificity,
and mutation resistance.
"""

from .drug_target_antigen import DrugTargetAntigen
from .drug_discovery_genes import BindingPocketGene, PharmacophoreGene, AllostericGene
from .drug_discovery_germinal_center import DrugDiscoveryGerminalCenter
from .omics_to_antigen_converter import OmicsToAntigenConverter
from .drug_target_evaluator import DrugTargetEvaluator

__all__ = [
    'DrugTargetAntigen',
    'BindingPocketGene',
    'PharmacophoreGene',
    'AllostericGene',
    'DrugDiscoveryGerminalCenter',
    'OmicsToAntigenConverter',
    'DrugTargetEvaluator'
]