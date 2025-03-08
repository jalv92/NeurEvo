"""
Base Environment - Clase base para adaptadores de entornos en NeurEvo.

Este módulo define la interfaz que deben implementar todos los adaptadores
de entorno para ser compatibles con el framework NeurEvo.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Union, Dict, List, Optional, TypeVar, Generic

import numpy as np

Observation = TypeVar('Observation')
Action = TypeVar('Action')
Info = Dict[str, Any]

class EnvironmentAdapter(Generic[Observation, Action], ABC):
    """
    Clase base abstracta para adaptadores de entorno.
    
    Define la interfaz común que todos los adaptadores de entorno deben implementar
    para ser compatibles con NeurEvo, independientemente del backend utilizado.
    """
    
    @abstractmethod
    def reset(self) -> Observation:
        """
        Reinicia el entorno y devuelve la observación inicial.
        
        Returns:
            Observación inicial
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Info]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar
            
        Returns:
            Tuple con (nueva_observación, recompensa, terminado, info)
        """
        pass
    
    @abstractmethod
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Obtiene la forma del espacio de observación.
        
        Returns:
            Tupla con las dimensiones del espacio de observación
        """
        pass
    
    @abstractmethod
    def get_action_size(self) -> int:
        """
        Obtiene el tamaño del espacio de acciones.
        
        Returns:
            Número de acciones posibles o dimensiones del espacio de acciones
        """
        pass
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Renderiza el estado actual del entorno.
        
        Args:
            mode: Modo de renderizado ('human', 'rgb_array', etc.)
            
        Returns:
            Datos de renderizado según el modo, o None
        """
        pass
    
    def close(self) -> None:
        """
        Cierra el entorno y libera recursos.
        """
        pass
    
    def seed(self, seed: int = None) -> List[int]:
        """
        Establece la semilla para la generación de números aleatorios.
        
        Args:
            seed: Semilla para el generador de números aleatorios
            
        Returns:
            Lista de semillas utilizadas
        """
        return [seed]
    
    def get_action_space_type(self) -> str:
        """
        Obtiene el tipo de espacio de acciones.
        
        Returns:
            'discrete' para acciones discretas, 'continuous' para continuas
        """
        return 'discrete'
    
    def normalize_observation(self, observation: Observation) -> np.ndarray:
        """
        Normaliza una observación para ser procesada por la red neuronal.
        
        Args:
            observation: Observación a normalizar
            
        Returns:
            Observación normalizada como array numpy
        """
        if isinstance(observation, np.ndarray):
            return observation
        return np.array(observation)
    
    def denormalize_action(self, action: np.ndarray) -> Action:
        """
        Convierte una acción de la red neuronal al formato requerido por el entorno.
        
        Args:
            action: Acción predecida por la red neuronal
            
        Returns:
            Acción en el formato requerido por el entorno
        """
        return action 