"""
Environments - Adaptadores de entorno para NeurEvo.

Este módulo contiene adaptadores para diferentes tipos de entornos,
permitiendo que NeurEvo interactúe con ellos de manera uniforme.
"""

from neurevo.environments.base import EnvironmentAdapter
from neurevo.environments.gym_adapter import GymAdapter
from neurevo.environments.custom_adapter import (
    CustomEnvironmentAdapter, 
    create_custom_environment
)
from neurevo.utils.registry import ComponentRegistry

__all__ = [
    'EnvironmentAdapter',
    'GymAdapter',
    'CustomEnvironmentAdapter',
    'create_custom_environment',
    'register_builtin_environments',
]

def register_builtin_environments(registry: ComponentRegistry) -> None:
    """
    Registra entornos integrados en el registro de componentes.
    
    Args:
        registry: Registro donde se añadirán los entornos
    """
    # Entornos de Gym
    try:
        import gym
        # Registrar entornos de Gym clásicos
        classic_control = [
            'CartPole-v1',
            'MountainCar-v0',
            'MountainCarContinuous-v0',
            'Acrobot-v1',
            'Pendulum-v1'
        ]
        
        for env_id in classic_control:
            registry.register(
                "environment", 
                env_id, 
                (GymAdapter, {"env_id": env_id, "use_gymnasium": False})
            )
    except ImportError:
        pass
    
    # Entornos de Gymnasium
    try:
        import gymnasium
        # Registrar entornos de Gymnasium clásicos
        classic_control = [
            'CartPole-v1',
            'MountainCar-v0',
            'MountainCarContinuous-v0',
            'Acrobot-v1',
            'Pendulum-v1'
        ]
        
        for env_id in classic_control:
            registry.register(
                "environment", 
                f"gymnasium_{env_id}", 
                (GymAdapter, {"env_id": env_id, "use_gymnasium": True})
            )
    except ImportError:
        pass 