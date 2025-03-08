"""
NeurEvo - Framework de aprendizaje por refuerzo con elementos cognitivos y evolutivos.
Este módulo actúa como el orquestador principal del framework.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple

from neurevo.core.agent import NeurEvoAgent
from neurevo.core.base_environment import BaseEnvironment
from neurevo.config.config import NeurEvoConfig

class NeurEvo:
    """
    Clase principal que orquesta la interacción entre agentes y entornos.
    Sirve como punto de entrada principal para los usuarios del framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el framework NeurEvo.
        
        Args:
            config: Configuración opcional para personalizar el comportamiento del framework
        """
        self.config = NeurEvoConfig(config) if config else NeurEvoConfig()
        self.agents = {}
        self.environments = {}
        
        # Configurar dispositivo automáticamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NeurEvo iniciado en dispositivo: {self.device}")
    
    def create_agent(self, agent_id: str, observation_shape: Union[Tuple, List[int]], 
                    action_size: int, **kwargs) -> NeurEvoAgent:
        """
        Crea un nuevo agente.
        
        Args:
            agent_id: Identificador único para el agente
            observation_shape: Forma del espacio de observación
            action_size: Tamaño del espacio de acciones
            **kwargs: Parámetros adicionales para la configuración del agente
            
        Returns:
            El agente creado
        """
        agent = NeurEvoAgent(
            observation_shape=observation_shape,
            action_size=action_size,
            device=self.device,
            **kwargs
        )
        self.agents[agent_id] = agent
        return agent
    
    def register_environment(self, env_id: str, environment: BaseEnvironment) -> None:
        """
        Registra un entorno para ser utilizado con agentes.
        
        Args:
            env_id: Identificador único para el entorno
            environment: Instancia de un entorno compatible
        """
        self.environments[env_id] = environment
    
    def train(self, agent_id: str, env_id: str, episodes: int, 
             render: bool = False, eval_interval: int = 10) -> Dict[str, List[float]]:
        """
        Entrena un agente en un entorno específico.
        
        Args:
            agent_id: Identificador del agente a entrenar
            env_id: Identificador del entorno para el entrenamiento
            episodes: Número de episodios de entrenamiento
            render: Si se debe renderizar el entorno durante el entrenamiento
            eval_interval: Intervalo para evaluar el rendimiento del agente
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        agent = self.agents.get(agent_id)
        env = self.environments.get(env_id)
        
        if not agent:
            raise ValueError(f"Agente con ID {agent_id} no encontrado")
        if not env:
            raise ValueError(f"Entorno con ID {env_id} no encontrado")
        
        # Ejecutar bucle de entrenamiento
        metrics = {
            'rewards': [],
            'losses': [],
            'q_values': []
        }
        
        for episode in range(episodes):
            # Entrenamiento
            episode_metrics = agent.train_episode(env, render=render)
            
            # Registrar métricas
            for key in metrics:
                if key in episode_metrics:
                    metrics[key].append(episode_metrics[key])
            
            # Evaluación periódica
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate(agent_id, env_id, episodes=3)
                print(f"Episodio {episode+1}/{episodes} - Recompensa de evaluación: {eval_reward:.2f}")
        
        return metrics
    
    def evaluate(self, agent_id: str, env_id: str, episodes: int = 5) -> float:
        """
        Evalúa un agente en un entorno específico.
        
        Args:
            agent_id: Identificador del agente a evaluar
            env_id: Identificador del entorno para la evaluación
            episodes: Número de episodios de evaluación
            
        Returns:
            Recompensa promedio obtenida
        """
        agent = self.agents.get(agent_id)
        env = self.environments.get(env_id)
        
        if not agent:
            raise ValueError(f"Agente con ID {agent_id} no encontrado")
        if not env:
            raise ValueError(f"Entorno con ID {env_id} no encontrado")
        
        total_reward = 0.0
        
        for _ in range(episodes):
            episode_reward = agent.evaluate_episode(env)
            total_reward += episode_reward
        
        return total_reward / episodes
    
    def save_agent(self, agent_id: str, path: str) -> None:
        """
        Guarda un agente en disco.
        
        Args:
            agent_id: Identificador del agente a guardar
            path: Ruta donde guardar el agente
        """
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agente con ID {agent_id} no encontrado")
        
        agent.save(path)
    
    def load_agent(self, agent_id: str, path: str) -> NeurEvoAgent:
        """
        Carga un agente desde disco.
        
        Args:
            agent_id: Identificador para el agente cargado
            path: Ruta desde donde cargar el agente
            
        Returns:
            El agente cargado
        """
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agente con ID {agent_id} no encontrado")
        
        agent.load(path)
        return agent 