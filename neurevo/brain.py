"""
Brain Interface - Interfaz unificada para el framework NeurEvo.

Este módulo proporciona una API simplificada para usar NeurEvo, permitiendo crear
y configurar agentes para diferentes entornos.
"""

import torch
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

from neurevo.core.agent import NeurEvoAgent
from neurevo.core.base_environment import BaseEnvironment
from neurevo.config.config import NeurEvoConfig
from neurevo.utils.registry import ComponentRegistry
from neurevo.environments import register_builtin_environments

class BrainInterface:
    """
    Interfaz principal para el framework NeurEvo.
    
    Esta clase proporciona métodos para crear y gestionar agentes NeurEvo,
    entrenarlos en diferentes entornos, y evaluar su rendimiento.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la interfaz del cerebro NeurEvo.
        
        Args:
            config: Diccionario de configuración opcional o instancia de NeurEvoConfig
        """
        if isinstance(config, NeurEvoConfig):
            self.config = config
        else:
            self.config = NeurEvoConfig(config) if config else NeurEvoConfig()
        
        # Inicializar registros para agentes y entornos
        self.environment_registry = ComponentRegistry()
        self.agents = {}
        
        # Registrar entornos incorporados
        register_builtin_environments(self.environment_registry)
        
        # Configurar dispositivo automáticamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NeurEvo iniciado en dispositivo: {self.device}")
    
    def register_environment(self, env_id: str, environment_class: Any, **env_kwargs) -> None:
        """
        Registra un entorno personalizado para su uso con NeurEvo.
        
        Args:
            env_id: Identificador único para el entorno
            environment_class: Clase del entorno o función fábrica
            **env_kwargs: Argumentos para la inicialización del entorno
        """
        self.environment_registry.register("environment", env_id, 
                                         (environment_class, env_kwargs))
    
    def create_for_environment(self, env_id: str, agent_id: Optional[str] = None,
                              **agent_kwargs) -> NeurEvoAgent:
        """
        Crea un agente configurado para un entorno específico.
        
        Args:
            env_id: ID del entorno registrado
            agent_id: ID opcional para el agente (se generará uno si no se proporciona)
            **agent_kwargs: Parámetros específicos para la configuración del agente
            
        Returns:
            Un agente NeurEvo configurado para el entorno especificado
            
        Raises:
            ValueError: Si el entorno no está registrado
        """
        if not self.environment_registry.contains("environment", env_id):
            raise ValueError(f"Entorno '{env_id}' no registrado")
        
        # Crear instancia de entorno
        env_class, env_kwargs = self.environment_registry.get("environment", env_id)
        environment = env_class(**env_kwargs)
        
        # Obtener información del entorno
        observation_shape = environment.get_observation_shape()
        action_size = environment.get_action_size()
        
        # Generar ID de agente si no se proporciona
        if agent_id is None:
            agent_id = f"agent_{len(self.agents) + 1}"
        
        # Crear agente
        agent = NeurEvoAgent(
            observation_shape=observation_shape,
            action_size=action_size,
            device=self.device,
            config=self.config,
            **agent_kwargs
        )
        
        self.agents[agent_id] = (agent, env_id)
        return agent
    
    def train(self, agent_id: Optional[str] = None, episodes: int = 1000, 
             render: bool = False, eval_interval: int = 100) -> Dict[str, List[float]]:
        """
        Entrena un agente específico o el último agente creado.
        
        Args:
            agent_id: ID del agente a entrenar (usa el último creado si es None)
            episodes: Número de episodios a entrenar
            render: Si se debe visualizar el entorno durante el entrenamiento
            eval_interval: Cada cuántos episodios evaluar el rendimiento
            
        Returns:
            Diccionario con métricas de entrenamiento
            
        Raises:
            ValueError: Si no hay agentes o el agente especificado no existe
        """
        if not self.agents:
            raise ValueError("No hay agentes creados. Usa create_for_environment primero.")
        
        # Usar el último agente si no se especifica uno
        if agent_id is None:
            agent_id = list(self.agents.keys())[-1]
        
        if agent_id not in self.agents:
            raise ValueError(f"Agente '{agent_id}' no encontrado")
        
        agent, env_id = self.agents[agent_id]
        
        # Crear instancia de entorno
        env_class, env_kwargs = self.environment_registry.get("environment", env_id)
        environment = env_class(**env_kwargs)
        
        # Métricas a recolectar
        metrics = {
            'rewards': [],
            'losses': [],
            'curiosity': [],
            'network_size': []
        }
        
        # Bucle de entrenamiento
        for episode in range(episodes):
            # Entrenar un episodio
            episode_metrics = agent.train_episode(environment, render=render)
            
            # Recolectar métricas
            for key in metrics:
                if key in episode_metrics:
                    metrics[key].append(episode_metrics[key])
            
            # Evaluación periódica
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate(agent_id, episodes=5)
                print(f"Episodio {episode+1}/{episodes} - Recompensa: {eval_reward:.2f}")
        
        return metrics
    
    def evaluate(self, agent_id: Optional[str] = None, 
                episodes: int = 10) -> float:
        """
        Evalúa el rendimiento de un agente.
        
        Args:
            agent_id: ID del agente a evaluar (usa el último creado si es None)
            episodes: Número de episodios para la evaluación
            
        Returns:
            Recompensa promedio durante la evaluación
        """
        if not self.agents:
            raise ValueError("No hay agentes creados. Usa create_for_environment primero.")
        
        # Usar el último agente si no se especifica uno
        if agent_id is None:
            agent_id = list(self.agents.keys())[-1]
        
        if agent_id not in self.agents:
            raise ValueError(f"Agente '{agent_id}' no encontrado")
        
        agent, env_id = self.agents[agent_id]
        
        # Crear instancia de entorno
        env_class, env_kwargs = self.environment_registry.get("environment", env_id)
        environment = env_class(**env_kwargs)
        
        # Evaluar
        total_reward = 0.0
        for _ in range(episodes):
            total_reward += agent.evaluate_episode(environment)
        
        return total_reward / episodes
    
    def save(self, agent_id: Optional[str] = None, path: Optional[str] = None) -> str:
        """
        Guarda un agente en disco.
        
        Args:
            agent_id: ID del agente a guardar (usa el último creado si es None)
            path: Ruta donde guardar el agente (genera una por defecto si es None)
            
        Returns:
            Ruta donde se guardó el agente
        """
        if not self.agents:
            raise ValueError("No hay agentes creados. Usa create_for_environment primero.")
        
        # Usar el último agente si no se especifica uno
        if agent_id is None:
            agent_id = list(self.agents.keys())[-1]
        
        if agent_id not in self.agents:
            raise ValueError(f"Agente '{agent_id}' no encontrado")
        
        agent, _ = self.agents[agent_id]
        
        # Generar ruta por defecto si no se proporciona
        if path is None:
            import os
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("models", f"{agent_id}_{timestamp}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Guardar agente
        agent.save(path)
        return path
    
    def load(self, path: str, agent_id: Optional[str] = None) -> str:
        """
        Carga un agente desde disco.
        
        Args:
            path: Ruta desde donde cargar el agente
            agent_id: ID opcional para el agente cargado (genera uno si es None)
            
        Returns:
            ID del agente cargado
        """
        # Generar ID de agente si no se proporciona
        if agent_id is None:
            agent_id = f"agent_loaded_{len(self.agents) + 1}"
        
        # Crear un agente temporal para cargar (se actualizará con load)
        # Nota: Esto es una simplificación, en la implementación real
        # necesitaríamos determinar observation_shape y action_size
        agent = NeurEvoAgent(
            observation_shape=(1,), 
            action_size=1,
            device=self.device
        )
        
        # Cargar agente
        agent.load(path)
        
        # Almacenar el agente (sin entorno asociado por ahora)
        self.agents[agent_id] = (agent, None)
        
        return agent_id


def create_brain(config: Optional[Dict[str, Any]] = None) -> BrainInterface:
    """
    Función fábrica para crear una instancia de BrainInterface.
    
    Args:
        config: Configuración opcional para el cerebro
        
    Returns:
        Nueva instancia de BrainInterface
    """
    return BrainInterface(config) 