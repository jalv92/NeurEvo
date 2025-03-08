"""
Gym Adapter - Adaptador para entornos Gym/Gymnasium.

Este módulo proporciona un adaptador para usar entornos de Gym/Gymnasium
con el framework NeurEvo.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union, cast

try:
    import gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    import gymnasium
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from neurevo.environments.base import EnvironmentAdapter

class GymAdapter(EnvironmentAdapter):
    """
    Adaptador para entornos Gym/Gymnasium.
    
    Este adaptador permite usar entornos de Gym o Gymnasium con NeurEvo,
    adaptando las interfaces y convirtiendo tipos de datos según sea necesario.
    """
    
    def __init__(self, env_id: str, use_gymnasium: bool = None, **env_kwargs):
        """
        Inicializa un adaptador de entorno Gym.
        
        Args:
            env_id: Identificador del entorno Gym (ej: 'CartPole-v1')
            use_gymnasium: Si True, usa la biblioteca gymnasium. Si None, 
                          intenta usar gymnasium y cae de vuelta a gym
            **env_kwargs: Parámetros adicionales para la creación del entorno
            
        Raises:
            ImportError: Si ni gym ni gymnasium están instalados
            ValueError: Si el entorno solicitado no existe
        """
        if use_gymnasium is None:
            # Auto-detectar qué biblioteca usar
            if HAS_GYMNASIUM:
                use_gymnasium = True
            elif HAS_GYM:
                use_gymnasium = False
            else:
                raise ImportError(
                    "Ni 'gym' ni 'gymnasium' están instalados. "
                    "Instala al menos uno con: pip install gymnasium"
                )
        elif use_gymnasium and not HAS_GYMNASIUM:
            raise ImportError(
                "Gymnasium solicitado pero no está instalado. "
                "Instala con: pip install gymnasium"
            )
        elif not use_gymnasium and not HAS_GYM:
            raise ImportError(
                "Gym solicitado pero no está instalado. "
                "Instala con: pip install gym"
            )
        
        self.use_gymnasium = use_gymnasium
        self.env_id = env_id
        
        # Crear entorno
        if self.use_gymnasium:
            self.env = gymnasium.make(env_id, **env_kwargs)
        else:
            self.env = gym.make(env_id, **env_kwargs)
        
        # Determinar tipo de espacio de acción
        if self.use_gymnasium:
            from gymnasium.spaces import Discrete, Box
            self.is_discrete = isinstance(self.env.action_space, Discrete)
        else:
            from gym.spaces import Discrete, Box
            self.is_discrete = isinstance(self.env.action_space, Discrete)
        
        self.action_size = self._get_action_size()
    
    def reset(self):
        """
        Reinicia el entorno y devuelve la observación inicial.
        
        Returns:
            Observación inicial
        """
        if self.use_gymnasium:
            observation, _ = self.env.reset()
            return observation
        else:
            return self.env.reset()
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar
            
        Returns:
            Tuple con (nueva_observación, recompensa, terminado, info)
        """
        if self.use_gymnasium:
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            return obs, reward, done, info
        else:
            return self.env.step(action)
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Obtiene la forma del espacio de observación.
        
        Returns:
            Tupla con las dimensiones del espacio de observación
        """
        space = self.env.observation_space
        
        if hasattr(space, 'shape'):
            return space.shape
        else:
            # Espacio discreto, convertir a one-hot
            return (space.n,)
    
    def get_action_size(self) -> int:
        """
        Obtiene el tamaño del espacio de acciones.
        
        Returns:
            Número de acciones posibles o dimensiones del espacio de acciones
        """
        return self.action_size
    
    def _get_action_size(self) -> int:
        """
        Método auxiliar para determinar el tamaño del espacio de acciones.
        
        Returns:
            Tamaño del espacio de acciones
        """
        if self.use_gymnasium:
            from gymnasium.spaces import Discrete, Box
        else:
            from gym.spaces import Discrete, Box
        
        space = self.env.action_space
        
        if isinstance(space, Discrete):
            return space.n
        elif isinstance(space, Box):
            return space.shape[0]
        else:
            # Para otros tipos de espacios, proporcionar un valor por defecto razonable
            return 1
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Renderiza el estado actual del entorno.
        
        Args:
            mode: Modo de renderizado ('human', 'rgb_array', etc.)
            
        Returns:
            Datos de renderizado según el modo, o None
        """
        try:
            if self.use_gymnasium:
                return self.env.render()
            else:
                return self.env.render(mode=mode)
        except Exception as e:
            print(f"Error al renderizar: {e}")
            return None
    
    def close(self) -> None:
        """
        Cierra el entorno y libera recursos.
        """
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def seed(self, seed: int = None) -> List[int]:
        """
        Establece la semilla para la generación de números aleatorios.
        
        Args:
            seed: Semilla para el generador de números aleatorios
            
        Returns:
            Lista de semillas utilizadas
        """
        if self.use_gymnasium:
            if hasattr(self.env, 'reset'):
                self.env.reset(seed=seed)
            return [seed]
        else:
            if hasattr(self.env, 'seed'):
                return self.env.seed(seed)
        return [seed]
    
    def get_action_space_type(self) -> str:
        """
        Obtiene el tipo de espacio de acciones.
        
        Returns:
            'discrete' para acciones discretas, 'continuous' para continuas
        """
        return 'discrete' if self.is_discrete else 'continuous'
    
    def denormalize_action(self, action: np.ndarray) -> Any:
        """
        Convierte una acción de la red neuronal al formato requerido por el entorno.
        
        Args:
            action: Acción predecida por la red neuronal
            
        Returns:
            Acción en el formato requerido por el entorno
        """
        if self.is_discrete:
            # Para espacios discretos, tomar el índice de mayor valor
            if len(action.shape) > 0 and action.shape[0] > 1:
                return int(np.argmax(action))
            else:
                # Si es un solo valor, redondear al entero más cercano
                return int(round(float(action)))
        else:
            # Para espacios continuos, usar directamente
            return action 