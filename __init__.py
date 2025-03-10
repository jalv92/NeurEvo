# Actualizar neurevo/__init__.py para exponer la API principal

"""
NeurEvo - Framework de aprendizaje por refuerzo con elementos cognitivos y evolutivos.
"""

from neurevo.brain import BrainInterface, create_brain
from neurevo.core.agent import NeurEvoAgent
from neurevo.config.config import NeurEvoConfig

__all__ = [
    # API principal
    'BrainInterface', 
    'create_brain',
    
    # Componentes principales
    'NeurEvoAgent',
    'NeurEvoConfig',
]

__version__ = '0.1.0'