"""
Utils - Utilidades para el framework NeurEvo.

Este módulo proporciona utilidades generales para el framework NeurEvo,
incluyendo gestión de tensores, optimización GPU y más.
"""

from neurevo.utils.tensor_utils import *
from neurevo.utils.memory_manager import MemoryManager
from neurevo.utils.gpu_optimization import optimize_gpu_memory, set_gpu_optimization
from neurevo.utils.registry import ComponentRegistry, register_component

__all__ = [
    # Gestión de memoria
    'MemoryManager',
    
    # Optimización GPU
    'optimize_gpu_memory',
    'set_gpu_optimization',
    
    # Registro de componentes
    'ComponentRegistry',
    'register_component',
    
    # Re-exportar utils de tensores (utilizando importación *)
    'to_tensor',
    'to_numpy',
    'ensure_tensor',
    'ensure_numpy',
    'clone_tensors',
    'detach_tensors',
] 