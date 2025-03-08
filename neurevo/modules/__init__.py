"""
Modules - Componentes modulares de la arquitectura neuronal de NeurEvo.

Este módulo contiene los componentes que conforman la arquitectura neuronal
de NeurEvo, incluyendo capas dinámicas y módulos especializados.
"""

from neurevo.modules.base_module import BaseModule
from neurevo.modules.dynamic_layer import DynamicLayer
from neurevo.modules.perception import PerceptionModule
from neurevo.modules.prediction import PredictionModule
from neurevo.modules.executive import ExecutiveModule

__all__ = [
    'BaseModule',
    'DynamicLayer',
    'PerceptionModule',
    'PredictionModule',
    'ExecutiveModule',
] 