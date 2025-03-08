"""
Clase base para entornos compatibles con el framework NeurEvo.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Union, List

class BaseEnvironment(ABC):
    """
    Clase abstracta que define la interfaz para entornos compatibles con NeurEvo.
    Todos los entornos deben heredar de esta clase e implementar sus métodos.
    """
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reinicia el entorno y devuelve la observación inicial.
        
        Returns:
            Observación inicial del entorno
        """
        pass
    
    @abstractmethod
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Ejecuta una acción en el entorno y devuelve el resultado.
        
        Args:
            action: Acción a ejecutar en el entorno
            
        Returns:
            Tupla con (observación, recompensa, terminado, info)
            - observación: Nueva observación del entorno
            - recompensa: Recompensa obtenida por la acción
            - terminado: Indicador de si el episodio ha terminado
            - info: Información adicional del entorno
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human') -> Union[np.ndarray, None]:
        """
        Renderiza el estado actual del entorno.
        
        Args:
            mode: Modo de renderizado ('human', 'rgb_array', etc.)
            
        Returns:
            Representación visual del entorno o None
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Cierra el entorno y libera recursos.
        """
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Tuple[int, ...]:
        """
        Devuelve la forma del espacio de observación.
        
        Returns:
            Tupla con las dimensiones del espacio de observación
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Union[int, Tuple[int, ...]]:
        """
        Devuelve el tamaño del espacio de acciones.
        
        Returns:
            Entero para acciones discretas o tupla para acciones continuas
        """
        pass
    
    def seed(self, seed: int = None) -> List[int]:
        """
        Establece la semilla para la generación de números aleatorios.
        
        Args:
            seed: Semilla para el generador de números aleatorios
            
        Returns:
            Lista con las semillas utilizadas
        """
        return [seed]
    
    def get_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado interno del entorno para guardarlo.
        
        Returns:
            Diccionario con el estado interno del entorno
        """
        return {}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Establece el estado interno del entorno desde un estado guardado.
        
        Args:
            state: Diccionario con el estado interno del entorno
        """
        pass 