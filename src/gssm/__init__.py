"""
GSSM for Insurance Forecasting
===============================

Core GSSM modules adapted for insurance policy pricing and claims forecasting.
"""

from .gssm_model import GSSMModel
from .flow_selectivity import FlowSelectivityLayer
from .state_space_layer import StateSpaceLayer
from .gssm_trainer import GSSMTrainer
from .insurance_gssm import InsuranceGSSM

__all__ = [
    'GSSMModel',
    'FlowSelectivityLayer',
    'StateSpaceLayer',
    'GSSMTrainer',
    'InsuranceGSSM'
]

__version__ = '1.0.0'
