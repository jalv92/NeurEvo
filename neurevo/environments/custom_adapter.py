"""
Custom Adapter - Adaptador para entornos personalizados.

Este módulo proporciona un adaptador para integrar entornos personalizados
con el framework NeurEvo.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union, Callable

from neurevo.environments.base import EnvironmentAdapter

class CustomEnvironmentAdapter(EnvironmentAdapter):
    """
    Adaptador para entornos personalizados.
    
    Permite integrar fácilmente entornos personalizados con NeurEvo,
    proporcionando funciones para las operaciones necesarias.
    """
    
    def __init__(
        self,
        reset_fn: Callable[[], Any],
        step_fn: Callable[[Any], Tuple[Any, float, bool, Dict[str, Any]]],
        observation_shape: Tuple[int, ...],
        action_size: int,
        render_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
        close_fn: Optional[Callable[[], None]] = None,
        seed_fn: Optional[Callable[[Optional[int]], List[int]]] = None,
        is_action_discrete: bool = True,
        observation_normalizer: Optional[Callable[[Any], np.ndarray]] = None,
        action_denormalizer: Optional[Callable[[np.ndarray], Any]] = None
    ):
        """
        Inicializa un adaptador de entorno personalizado.
        
        Args:
            reset_fn: Función para reiniciar el entorno
            step_fn: Función para ejecutar un paso en el entorno
            observation_shape: Forma del espacio de observación
            action_size: Tamaño del espacio de acciones
            render_fn: Función opcional para renderizar el entorno
            close_fn: Función opcional para cerrar el entorno
            seed_fn: Función opcional para establecer la semilla
            is_action_discrete: Si las acciones son discretas o continuas
            observation_normalizer: Función para normalizar observaciones
            action_denormalizer: Función para denormalizar acciones
        """
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._render_fn = render_fn
        self._close_fn = close_fn
        self._seed_fn = seed_fn
        self._is_action_discrete = is_action_discrete
        self._observation_normalizer = observation_normalizer
        self._action_denormalizer = action_denormalizer
    
    def reset(self):
        """
        Reinicia el entorno y devuelve la observación inicial.
        
        Returns:
            Observación inicial
        """
        return self._reset_fn()
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar
            
        Returns:
            Tuple con (nueva_observación, recompensa, terminado, info)
        """
        return self._step_fn(action)
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Obtiene la forma del espacio de observación.
        
        Returns:
            Tupla con las dimensiones del espacio de observación
        """
        return self._observation_shape
    
    def get_action_size(self) -> int:
        """
        Obtiene el tamaño del espacio de acciones.
        
        Returns:
            Número de acciones posibles o dimensiones del espacio de acciones
        """
        return self._action_size
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Renderiza el estado actual del entorno.
        
        Args:
            mode: Modo de renderizado (ignorado en el adaptador personalizado)
            
        Returns:
            Datos de renderizado según el modo, o None
        """
        if self._render_fn is not None:
            return self._render_fn()
        return None
    
    def close(self) -> None:
        """
        Cierra el entorno y libera recursos.
        """
        if self._close_fn is not None:
            self._close_fn()
    
    def seed(self, seed: int = None) -> List[int]:
        """
        Establece la semilla para la generación de números aleatorios.
        
        Args:
            seed: Semilla para el generador de números aleatorios
            
        Returns:
            Lista de semillas utilizadas
        """
        if self._seed_fn is not None:
            return self._seed_fn(seed)
        return [seed]
    
    def get_action_space_type(self) -> str:
        """
        Obtiene el tipo de espacio de acciones.
        
        Returns:
            'discrete' para acciones discretas, 'continuous' para continuas
        """
        return 'discrete' if self._is_action_discrete else 'continuous'
    
    def normalize_observation(self, observation: Any) -> np.ndarray:
        """
        Normaliza una observación para ser procesada por la red neuronal.
        
        Args:
            observation: Observación a normalizar
            
        Returns:
            Observación normalizada como array numpy
        """
        if self._observation_normalizer is not None:
            return self._observation_normalizer(observation)
        return super().normalize_observation(observation)
    
    def denormalize_action(self, action: np.ndarray) -> Any:
        """
        Convierte una acción de la red neuronal al formato requerido por el entorno.
        
        Args:
            action: Acción predecida por la red neuronal
            
        Returns:
            Acción en el formato requerido por el entorno
        """
        if self._action_denormalizer is not None:
            return self._action_denormalizer(action)
        
        if self._is_action_discrete:
            # Para espacios discretos, tomar el índice de mayor valor
            if len(action.shape) > 0 and action.shape[0] > 1:
                return int(np.argmax(action))
            else:
                return int(round(float(action)))
        
        return action


def create_custom_environment(
    reset_fn: Callable[[], Any],
    step_fn: Callable[[Any], Tuple[Any, float, bool, Dict[str, Any]]],
    observation_shape: Tuple[int, ...],
    action_size: int,
    **kwargs
) -> CustomEnvironmentAdapter:
    """
    Función de utilidad para crear un entorno personalizado.
    
    Args:
        reset_fn: Función para reiniciar el entorno
        step_fn: Función para ejecutar un paso en el entorno
        observation_shape: Forma del espacio de observación
        action_size: Tamaño del espacio de acciones
        **kwargs: Argumentos adicionales para CustomEnvironmentAdapter
        
    Returns:
        Adaptador de entorno personalizado
    """
    return CustomEnvironmentAdapter(
        reset_fn=reset_fn,
        step_fn=step_fn,
        observation_shape=observation_shape,
        action_size=action_size,
        **kwargs
    ) 