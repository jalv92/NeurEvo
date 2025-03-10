"""
Brain Interface - Interfaz unificada para el framework NeurEvo.

Proporciona una API completa para usar NeurEvo en cualquier proyecto,
con soporte específico para aplicaciones de trading.
"""

import torch
import threading
import pickle
import os
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple

from neurevo.core.agent import NeurEvoAgent
from neurevo.config.config import NeurEvoConfig
from neurevo.utils.registry import ComponentRegistry
from neurevo.environments import register_builtin_environments
from neurevo.environments.custom_adapter import create_custom_environment
from neurevo.environments.base import EnvironmentAdapter

class BrainInterface:
    """
    Interfaz principal para el framework NeurEvo.
    
    Proporciona métodos para crear y gestionar agentes, entrenarlos
    en diferentes entornos, y evaluar su rendimiento.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la interfaz del cerebro NeurEvo.
        
        Args:
            config: Diccionario de configuración opcional
        """
        # Configuración
        self.config = NeurEvoConfig(config if config else {})
        
        # Registros para entornos y agentes
        self.environment_registry = ComponentRegistry()
        self.agents = {}
        self.agent_counter = 0
        
        # Registro de entornos-agentes
        self.agent_to_env = {}
        
        # Registrar entornos incorporados
        register_builtin_environments(self.environment_registry)
        
        # Configurar dispositivo automáticamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lock para thread-safety
        self._predict_lock = threading.RLock()
    
    def register_environment(self, env, env_id: Optional[str] = None) -> str:
        """
        Registra un entorno para su uso con NeurEvo.
        
        Args:
            env: Objeto de entorno tipo gym o adaptador
            env_id: Identificador opcional para el entorno
            
        Returns:
            El ID del entorno registrado
        """
        # Generar ID si no se proporciona
        if env_id is None:
            env_id = f"env_{len(self.environment_registry.list_components('environment')) + 1}"
        
        # Si es un entorno Gym, adaptarlo
        try:
            if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
                from neurevo.environments.gym_adapter import GymAdapter
                env_adapter = GymAdapter(env)
            elif isinstance(env, EnvironmentAdapter):
                env_adapter = env
            else:
                # Intentar crear adaptador personalizado
                reset_fn = getattr(env, 'reset', None)
                step_fn = getattr(env, 'step', None)
                
                if reset_fn and step_fn:
                    # Determinar observation_shape y action_size
                    try:
                        # Intentar inferir del entorno
                        state = env.reset()
                        observation_shape = state.shape if hasattr(state, 'shape') else (len(state),)
                        
                        # Inferir action_size
                        if hasattr(env, 'action_space'):
                            if hasattr(env.action_space, 'n'):
                                action_size = env.action_space.n
                            else:
                                action_size = env.action_space.shape[0]
                        else:
                            # Valor por defecto
                            action_size = 1
                        
                        env_adapter = create_custom_environment(
                            reset_fn=reset_fn,
                            step_fn=step_fn,
                            observation_shape=observation_shape,
                            action_size=action_size
                        )
                    except Exception as e:
                        raise ValueError(f"No se pudo crear adaptador para entorno: {e}")
                else:
                    raise ValueError("El entorno debe implementar los métodos reset() y step()")
        except Exception as e:
            raise ValueError(f"Error al adaptar entorno: {e}")
        
        # Registrar el adaptador
        self.environment_registry.register("environment", env_id, env_adapter)
        
        return env_id
    
    def create_for_environment(self, env, agent_id: Optional[str] = None, **agent_kwargs) -> str:
        """
        Crea un agente para un entorno específico.
        
        Args:
            env: Entorno o ID de entorno registrado
            agent_id: ID opcional para el agente
            **agent_kwargs: Parámetros específicos para el agente
            
        Returns:
            ID del agente creado
        """
        # Determinar el entorno
        env_id = None
        env_adapter = None
        
        if isinstance(env, str):
            # Es un ID de entorno
            env_id = env
            if self.environment_registry.contains("environment", env_id):
                env_adapter = self.environment_registry.get("environment", env_id)
            else:
                raise ValueError(f"Entorno con ID '{env_id}' no registrado")
        else:
            # Es un objeto entorno, registrarlo
            env_id = self.register_environment(env)
            env_adapter = self.environment_registry.get("environment", env_id)
        
        # Generar ID de agente si no se proporciona
        if agent_id is None:
            self.agent_counter += 1
            agent_id = f"agent_{self.agent_counter}"
        
        # Obtener información del entorno
        observation_shape = env_adapter.get_observation_shape()
        action_size = env_adapter.get_action_size()
        
        # Actualizar configuración con kwargs
        agent_config = self.config.copy()
        for key, value in agent_kwargs.items():
            agent_config[key] = value
        
        # Crear agente
        agent = NeurEvoAgent(
            observation_shape=observation_shape,
            action_size=action_size,
            device=self.device,
            config=agent_config
        )
        
        # Almacenar agente y su asociación con el entorno
        self.agents[agent_id] = agent
        self.agent_to_env[agent_id] = env_id
        
        return agent_id
    
    def predict(self, agent, state) -> Any:
        """
        Predice una acción basada en el estado actual.
        
        Args:
            agent: ID del agente o objeto agente
            state: Estado de observación del entorno
            
        Returns:
            Acción predicha
        """
        try:
            # Thread-safety
            with self._predict_lock:
                # Determinar el agente
                if isinstance(agent, str):
                    if agent not in self.agents:
                        raise ValueError(f"Agente con ID '{agent}' no encontrado")
                    agent_obj = self.agents[agent]
                else:
                    agent_obj = agent
                
                # Normalizar estado
                if isinstance(state, list):
                    state = np.array(state)
                elif isinstance(state, torch.Tensor):
                    state = state.detach().cpu().numpy()
                
                # Asegurar que el estado tiene la forma correcta
                if state.shape != agent_obj.observation_shape and len(state.shape) == 1:
                    state = state.reshape(agent_obj.observation_shape)
                
                # Predecir acción
                action = agent_obj.select_action(state, training=False)
                
                return action
        except Exception as e:
            # En producción, no lanzar excepción sino devolver acción por defecto
            print(f"Error en predict(): {e}")
            return 0
    
    def train(self, agent_id: str, episodes: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        """
        Entrena un agente durante un número específico de episodios.
        
        Args:
            agent_id: ID del agente a entrenar
            episodes: Número de episodios de entrenamiento
            verbose: Si se debe mostrar información del progreso
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        # Verificar agente
        if agent_id not in self.agents:
            raise ValueError(f"Agente con ID '{agent_id}' no encontrado")
        
        agent = self.agents[agent_id]
        
        # Obtener entorno
        env_id = self.agent_to_env.get(agent_id)
        if not env_id:
            raise ValueError(f"No hay entorno asociado al agente '{agent_id}'")
        
        env_adapter = self.environment_registry.get("environment", env_id)
        
        # Métricas a recolectar
        metrics = {
            "rewards": [],
            "avg_drawdowns": [],
            "losses": [],
            "exploration_rates": []
        }
        
        # Bucle de entrenamiento
        for episode in range(episodes):
            # Entrenar un episodio
            episode_metrics = agent.train_episode(env_adapter)
            
            # Recolectar métricas
            rewards = metrics["rewards"]
            rewards.append(episode_metrics.get("reward", 0.0))
            
            if "loss" in episode_metrics:
                metrics["losses"].append(episode_metrics["loss"])
            
            if "exploration_rate" in episode_metrics:
                metrics["exploration_rates"].append(episode_metrics["exploration_rate"])
            
            # Calcular drawdown si hay suficientes episodios
            if len(rewards) >= 5:
                # Drawdown: diferencia entre máximo previo y valor actual
                max_reward = max(rewards[:-5])
                current_reward = rewards[-1]
                drawdown = max(0, max_reward - current_reward)
                
                # Normalizar por el máximo
                if max_reward > 0:
                    relative_drawdown = drawdown / max_reward
                else:
                    relative_drawdown = 0.0
                
                metrics["avg_drawdowns"].append(relative_drawdown)
            
            # Mostrar progreso
            if verbose and (episode + 1) % max(1, episodes // 10) == 0:
                current_reward = rewards[-1]
                avg_reward = sum(rewards[-10:]) / min(10, len(rewards[-10:]))
                print(f"Episodio {episode+1}/{episodes} - Recompensa: {current_reward:.2f}, Media últimos 10: {avg_reward:.2f}")
        
        # Añadir recompensa final
        metrics["final_reward"] = metrics["rewards"][-1] if metrics["rewards"] else 0.0
        
        return metrics
    
    def save(self, filepath: str) -> bool:
        """
        Guarda el estado completo del cerebro en disco.
        
        Args:
            filepath: Ruta donde guardar el estado
            
        Returns:
            True si la operación tuvo éxito
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Estado a guardar
            state = {
                "config": self.config,
                "agents": {},
                "agent_to_env": self.agent_to_env,
                "agent_counter": self.agent_counter
            }
            
            # Guardar cada agente
            for agent_id, agent in self.agents.items():
                agent_state = agent.get_state_dict()
                state["agents"][agent_id] = agent_state
            
            # Determinar formato basado en extensión
            if filepath.endswith('.pt'):
                # Formato PyTorch
                torch.save(state, filepath)
            else:
                # Formato pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)
            
            return True
        except Exception as e:
            print(f"Error al guardar cerebro: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Carga el estado completo del cerebro desde disco.
        
        Args:
            filepath: Ruta desde donde cargar el estado
            
        Returns:
            True si la operación tuvo éxito
        """
        try:
            # Determinar formato basado en extensión
            if filepath.endswith('.pt'):
                # Formato PyTorch
                state = torch.load(filepath, map_location=self.device)
            else:
                # Formato pickle
                with open(filepath, 'rb') as f:
                    state = pickle.load(f)
            
            # Cargar configuración
            self.config = state.get("config", NeurEvoConfig())
            self.agent_to_env = state.get("agent_to_env", {})
            self.agent_counter = state.get("agent_counter", 0)
            
            # Cargar agentes
            for agent_id, agent_state in state.get("agents", {}).items():
                # Recrear agente
                if "observation_shape" in agent_state and "action_size" in agent_state:
                    agent = NeurEvoAgent(
                        observation_shape=agent_state["observation_shape"],
                        action_size=agent_state["action_size"],
                        device=self.device,
                        config=self.config
                    )
                    
                    # Cargar estado
                    agent.load_state_dict(agent_state)
                    
                    # Almacenar agente
                    self.agents[agent_id] = agent
            
            return True
        except Exception as e:
            print(f"Error al cargar cerebro: {e}")
            return False
    
    def run_episode(self, agent_id: str, render: bool = False) -> Dict[str, Any]:
        """
        Ejecuta un episodio completo con un agente en su entorno asociado.
        
        Args:
            agent_id: ID del agente
            render: Si se debe renderizar el entorno
            
        Returns:
            Diccionario con métricas del episodio
        """
        # Verificar agente
        if agent_id not in self.agents:
            raise ValueError(f"Agente con ID '{agent_id}' no encontrado")
        
        agent = self.agents[agent_id]
        
        # Obtener entorno
        env_id = self.agent_to_env.get(agent_id)
        if not env_id:
            raise ValueError(f"No hay entorno asociado al agente '{agent_id}'")
        
        env_adapter = self.environment_registry.get("environment", env_id)
        
        # Ejecutar episodio
        reward = agent.evaluate_episode(env_adapter, render=render)
        
        return {
            "reward": reward,
            "success": reward > 0  # Criterio simple de éxito
        }


def create_brain(config: Optional[Dict[str, Any]] = None) -> BrainInterface:
    """
    Crea una nueva instancia de BrainInterface.
    
    Args:
        config: Configuración opcional para el cerebro
        
    Returns:
        Instancia de BrainInterface
    """
    # Verificar y sanitizar configuración
    if config is None:
        config = {}
    
    return BrainInterface(config)