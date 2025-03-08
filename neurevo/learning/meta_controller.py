"""
Controlador para meta-aprendizaje en el framework NeurEvo.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Union, Optional, Callable
import random
import time

class MetaController:
    """
    Controlador para meta-aprendizaje que adapta dinámicamente las estrategias
    de aprendizaje, hiperparámetros y arquitecturas según el rendimiento.
    """
    
    def __init__(self, 
                 enable_curriculum: bool = True,
                 enable_hyperparameter_tuning: bool = True,
                 enable_architecture_search: bool = False,
                 evaluation_interval: int = 10):
        """
        Inicializa el meta-controlador.
        
        Args:
            enable_curriculum: Si se debe habilitar el aprendizaje curricular
            enable_hyperparameter_tuning: Si se debe habilitar el ajuste de hiperparámetros
            enable_architecture_search: Si se debe habilitar la búsqueda de arquitecturas
            evaluation_interval: Intervalo de episodios para evaluación y adaptación
        """
        # Configuración
        self.enable_curriculum = enable_curriculum
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
        self.enable_architecture_search = enable_architecture_search
        self.evaluation_interval = evaluation_interval
        
        # Estado actual
        self.curriculum_level = 0
        self.current_strategy = "exploration"  # exploration, exploitation, balanced
        self.last_adaptation_time = time.time()
        
        # Historial de rendimiento
        self.performance_history = {
            'rewards': [],
            'success_rates': [],
            'episode_lengths': [],
            'learning_progress': []
        }
        
        # Estrategias disponibles
        self.strategies = {
            "exploration": {
                "epsilon": 0.3,
                "learning_rate": 0.001,
                "curiosity_weight": 0.2,
                "entropy_weight": 0.01
            },
            "exploitation": {
                "epsilon": 0.05,
                "learning_rate": 0.0005,
                "curiosity_weight": 0.05,
                "entropy_weight": 0.001
            },
            "balanced": {
                "epsilon": 0.1,
                "learning_rate": 0.0007,
                "curiosity_weight": 0.1,
                "entropy_weight": 0.005
            }
        }
        
        # Niveles curriculares (de más fácil a más difícil)
        self.curriculum_levels = []
        
        # Configuraciones de arquitectura
        self.architecture_configs = []
    
    def update_performance(self, metrics: Dict[str, Any]) -> None:
        """
        Actualiza el historial de rendimiento con nuevas métricas.
        
        Args:
            metrics: Diccionario con métricas de rendimiento
        """
        # Extraer métricas relevantes
        if 'reward' in metrics:
            self.performance_history['rewards'].append(metrics['reward'])
        
        if 'success_rate' in metrics:
            self.performance_history['success_rates'].append(metrics['success_rate'])
        
        if 'episode_length' in metrics:
            self.performance_history['episode_lengths'].append(metrics['episode_length'])
        
        # Calcular progreso de aprendizaje (mejora relativa)
        if len(self.performance_history['rewards']) > 1:
            recent_rewards = self.performance_history['rewards'][-10:]
            if len(recent_rewards) > 1:
                progress = (np.mean(recent_rewards[-3:]) - np.mean(recent_rewards[:-3])) / (np.std(recent_rewards) + 1e-8)
                self.performance_history['learning_progress'].append(progress)
    
    def should_adapt(self, episode: int) -> bool:
        """
        Determina si es momento de adaptar la estrategia.
        
        Args:
            episode: Episodio actual
            
        Returns:
            True si se debe adaptar la estrategia
        """
        # Adaptar en intervalos regulares
        if episode % self.evaluation_interval == 0:
            return True
        
        # Adaptar si ha pasado suficiente tiempo
        current_time = time.time()
        if current_time - self.last_adaptation_time > 300:  # 5 minutos
            self.last_adaptation_time = current_time
            return True
        
        # Adaptar si el rendimiento se ha estancado
        if len(self.performance_history['learning_progress']) >= 5:
            recent_progress = self.performance_history['learning_progress'][-5:]
            if np.mean(recent_progress) < 0.01:
                return True
        
        return False
    
    def adapt_strategy(self, agent: Any = None) -> Dict[str, Any]:
        """
        Adapta la estrategia de aprendizaje según el rendimiento actual.
        
        Args:
            agent: Agente a adaptar (opcional)
            
        Returns:
            Diccionario con los parámetros adaptados
        """
        # Actualizar tiempo de adaptación
        self.last_adaptation_time = time.time()
        
        # Determinar la mejor estrategia según el progreso de aprendizaje
        if len(self.performance_history['learning_progress']) >= 3:
            progress = np.mean(self.performance_history['learning_progress'][-3:])
            
            if progress < -0.1:
                # Rendimiento empeorando: aumentar exploración
                self.current_strategy = "exploration"
            elif progress > 0.1:
                # Buen progreso: equilibrar
                self.current_strategy = "balanced"
            elif len(self.performance_history['rewards']) > 10:
                recent_rewards = self.performance_history['rewards'][-10:]
                if np.mean(recent_rewards) > 0.7 * max(self.performance_history['rewards']):
                    # Rendimiento cercano al máximo: explotar
                    self.current_strategy = "exploitation"
        
        # Obtener parámetros de la estrategia actual
        params = self.strategies[self.current_strategy].copy()
        
        # Aplicar parámetros al agente si se proporciona
        if agent is not None:
            if hasattr(agent, 'epsilon'):
                agent.epsilon = params["epsilon"]
            
            if hasattr(agent, 'optimizer') and hasattr(agent.optimizer, 'param_groups'):
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = params["learning_rate"]
            
            if hasattr(agent, 'curiosity') and hasattr(agent.curiosity, 'weight'):
                agent.curiosity.weight = params["curiosity_weight"]
        
        return params
    
    def advance_curriculum(self, success_rate: float) -> bool:
        """
        Avanza al siguiente nivel curricular si el rendimiento es suficiente.
        
        Args:
            success_rate: Tasa de éxito actual
            
        Returns:
            True si se avanzó de nivel
        """
        if not self.enable_curriculum or not self.curriculum_levels:
            return False
        
        # Avanzar si la tasa de éxito es suficiente
        if success_rate > 0.7 and self.curriculum_level < len(self.curriculum_levels) - 1:
            self.curriculum_level += 1
            return True
        
        return False
    
    def get_current_curriculum(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del nivel curricular actual.
        
        Returns:
            Configuración del nivel curricular actual
        """
        if not self.curriculum_levels or self.curriculum_level >= len(self.curriculum_levels):
            return {}
        
        return self.curriculum_levels[self.curriculum_level]
    
    def tune_hyperparameters(self, agent: Any = None) -> Dict[str, Any]:
        """
        Ajusta los hiperparámetros según el rendimiento.
        
        Args:
            agent: Agente a ajustar (opcional)
            
        Returns:
            Diccionario con los hiperparámetros ajustados
        """
        if not self.enable_hyperparameter_tuning:
            return {}
        
        # Análisis simple del rendimiento
        if len(self.performance_history['rewards']) < 10:
            return {}
        
        recent_rewards = self.performance_history['rewards'][-10:]
        reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        
        # Ajustes basados en tendencias
        params = {}
        
        if reward_trend < 0:
            # Rendimiento empeorando: reducir tasa de aprendizaje
            params["learning_rate"] = max(0.0001, self.strategies[self.current_strategy]["learning_rate"] * 0.8)
            
            # Aumentar exploración
            params["epsilon"] = min(0.5, self.strategies[self.current_strategy]["epsilon"] * 1.5)
            
            # Aumentar peso de curiosidad
            params["curiosity_weight"] = min(0.3, self.strategies[self.current_strategy]["curiosity_weight"] * 1.5)
        
        elif reward_trend > 0:
            # Rendimiento mejorando: mantener o aumentar ligeramente tasa de aprendizaje
            params["learning_rate"] = min(0.01, self.strategies[self.current_strategy]["learning_rate"] * 1.1)
            
            # Reducir exploración gradualmente
            params["epsilon"] = max(0.05, self.strategies[self.current_strategy]["epsilon"] * 0.9)
        
        # Actualizar estrategia actual
        if params:
            for key, value in params.items():
                self.strategies[self.current_strategy][key] = value
            
            # Aplicar al agente si se proporciona
            if agent is not None:
                self.apply_params_to_agent(agent, params)
        
        return params
    
    def apply_params_to_agent(self, agent: Any, params: Dict[str, Any]) -> None:
        """
        Aplica los parámetros al agente.
        
        Args:
            agent: Agente a ajustar
            params: Parámetros a aplicar
        """
        if "learning_rate" in params and hasattr(agent, 'optimizer'):
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = params["learning_rate"]
        
        if "epsilon" in params and hasattr(agent, 'epsilon'):
            agent.epsilon = params["epsilon"]
        
        if "curiosity_weight" in params and hasattr(agent, 'curiosity'):
            agent.curiosity.weight = params["curiosity_weight"]
    
    def suggest_architecture(self, observation_shape: Tuple[int, ...], 
                            action_size: int) -> Dict[str, Any]:
        """
        Sugiere una arquitectura de red basada en el rendimiento.
        
        Args:
            observation_shape: Forma del espacio de observación
            action_size: Tamaño del espacio de acciones
            
        Returns:
            Configuración de arquitectura sugerida
        """
        if not self.enable_architecture_search:
            return {}
        
        # Arquitectura base
        architecture = {
            "hidden_layers": [128, 128],
            "use_batch_norm": True,
            "activation": "relu",
            "use_dueling": True
        }
        
        # Ajustar según complejidad del problema
        input_size = np.prod(observation_shape)
        
        if input_size > 1000:
            # Entrada grande (imágenes): usar arquitectura más profunda
            architecture["hidden_layers"] = [256, 256, 128]
        elif input_size < 10:
            # Entrada pequeña: arquitectura más simple
            architecture["hidden_layers"] = [64, 64]
        
        if action_size > 20:
            # Muchas acciones: aumentar última capa
            architecture["hidden_layers"][-1] = max(128, architecture["hidden_layers"][-1])
        
        # Ajustar según rendimiento
        if len(self.performance_history['rewards']) > 20:
            recent_progress = np.mean(self.performance_history['learning_progress'][-5:])
            
            if recent_progress < 0.01:
                # Estancamiento: probar arquitectura diferente
                if random.random() < 0.5:
                    # Más profunda
                    architecture["hidden_layers"].append(64)
                else:
                    # Más ancha
                    architecture["hidden_layers"] = [size * 2 for size in architecture["hidden_layers"]]
        
        return architecture
    
    def register_curriculum_level(self, config: Dict[str, Any]) -> None:
        """
        Registra un nivel curricular.
        
        Args:
            config: Configuración del nivel curricular
        """
        self.curriculum_levels.append(config)
    
    def reset(self) -> None:
        """
        Reinicia el meta-controlador.
        """
        self.curriculum_level = 0
        self.current_strategy = "exploration"
        self.last_adaptation_time = time.time()
        self.performance_history = {
            'rewards': [],
            'success_rates': [],
            'episode_lengths': [],
            'learning_progress': []
        } 