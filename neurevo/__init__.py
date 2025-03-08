"""
NeurEvo - Framework de aprendizaje por refuerzo con elementos cognitivos y evolutivos.

Este paquete implementa redes neuronales dinámicas que pueden crecer y reducirse
automáticamente, junto con mecanismos de curiosidad intrínseca, memoria episódica
y transferencia de habilidades.
"""

# Importación de API principal
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