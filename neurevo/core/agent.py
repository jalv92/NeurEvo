"""
Implementación del agente principal del framework NeurEvo.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from typing import Dict, Any, List, Tuple, Union, Optional

from neurevo.modules.perception import PerceptionModule
from neurevo.modules.prediction import PredictionModule
from neurevo.modules.executive import ExecutiveModule
from neurevo.memory.episodic_memory import EpisodicMemory
from neurevo.learning.curiosity import CuriosityModule
from neurevo.learning.meta_controller import MetaController
from neurevo.learning.skill_library import SkillLibrary
from neurevo.utils.tensor_utils import validate_tensor_shape
from neurevo.core.base_environment import BaseEnvironment
from neurevo.config.config import NeurEvoConfig

class NeurEvoAgent:
    """
    Agente principal del framework NeurEvo.
    
    Implementa un agente con capacidades cognitivas y evolutivas,
    incluyendo percepción, predicción, toma de decisiones, curiosidad
    intrínseca y adaptación dinámica de su arquitectura.
    """
    
    def __init__(self, observation_shape: Union[Tuple[int, ...], List[int]], 
                 action_size: int, device: torch.device = None, config: Optional[NeurEvoConfig] = None, **kwargs):
        """
        Inicializa un nuevo agente NeurEvo.
        
        Args:
            observation_shape: Forma del espacio de observación
            action_size: Tamaño del espacio de acciones
            device: Dispositivo para cálculos (CPU/GPU)
            config: Configuración del agente
            **kwargs: Parámetros adicionales de configuración
        """
        # Configuración
        self.config = config if config is not None else NeurEvoConfig()
        
        # Actualizar configuración con kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Parámetros básicos
        self.observation_shape = tuple(observation_shape)
        self.action_size = action_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar módulos cognitivos
        self.perception = PerceptionModule(
            input_shape=self.observation_shape,
            hidden_size=self.config.perception_hidden_size,
            output_size=self.config.perception_output_size,
            device=self.device
        )
        
        self.prediction = PredictionModule(
            input_size=self.config.perception_output_size,
            hidden_size=self.config.prediction_hidden_size,
            output_size=self.config.prediction_output_size,
            device=self.device
        )
        
        self.executive = ExecutiveModule(
            input_size=self.config.perception_output_size + self.config.prediction_output_size,
            hidden_size=self.config.executive_hidden_size,
            output_size=action_size,
            device=self.device
        )
        
        # Inicializar módulos de aprendizaje
        self.memory = EpisodicMemory(
            capacity=self.config.memory_capacity,
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            prioritized=self.config.prioritized_replay
        )
        
        self.curiosity = CuriosityModule(
            state_size=self.config.perception_output_size,
            action_size=self.action_size,
            hidden_size=self.config.curiosity_hidden_size,
            device=self.device,
            learning_rate=self.config.curiosity_learning_rate
        )
        
        self.meta_controller = MetaController(
            state_size=self.config.perception_output_size,
            action_size=self.action_size,
            hidden_size=self.config.meta_hidden_size,
            device=self.device
        )
        
        self.skill_library = SkillLibrary(
            state_size=self.config.perception_output_size,
            action_size=self.action_size,
            hidden_size=self.config.skill_hidden_size,
            device=self.device
        )
        
        # Optimizador
        self.optimizer = optim.Adam(
            list(self.perception.parameters()) +
            list(self.prediction.parameters()) +
            list(self.executive.parameters()),
            lr=self.config.learning_rate
        )
        
        # Contadores y estadísticas
        self.total_steps = 0
        self.episodes_completed = 0
        self.exploration_rate = self.config.initial_exploration
        
        # Iniciar episodio en memoria
        self.memory.start_episode()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción basada en el estado actual.
        
        Args:
            state: Estado actual del entorno
            training: Si el agente está en modo entrenamiento
            
        Returns:
            Acción seleccionada
        """
        # Convertir estado a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Exploración aleatoria durante entrenamiento
        if training and random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        
        # Procesamiento a través de módulos cognitivos
        with torch.no_grad():
            # Percepción
            perception_output = self.perception(state_tensor)
            
            # Predicción
            prediction_output = self.prediction(perception_output)
            
            # Fusionar características
            combined_features = torch.cat([perception_output, prediction_output], dim=1)
            
            # Ejecutivo (selección de acción)
            action_probs = self.executive(combined_features)
            
            # Seleccionar acción con mayor probabilidad
            action = torch.argmax(action_probs, dim=1).item()
        
        # Actualizar tasa de exploración
        if training:
            self.exploration_rate = max(
                self.config.min_exploration,
                self.exploration_rate * self.config.exploration_decay
            )
        
        return action
    
    def train_episode(self, env: BaseEnvironment, render: bool = False) -> Dict[str, float]:
        """
        Entrena al agente durante un episodio completo.
        
        Args:
            env: Entorno de entrenamiento
            render: Si se debe renderizar el entorno
            
        Returns:
            Diccionario con métricas del episodio
        """
        # Reiniciar entorno
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        episode_loss = 0.0
        episode_curiosity = 0.0
        
        # Iniciar nuevo episodio en memoria
        self.memory.start_episode()
        
        # Bucle de episodio
        while not done:
            # Seleccionar acción
            action = self.select_action(state, training=True)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, info = env.step(action)
            
            # Calcular recompensa de curiosidad
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            
            curiosity_reward = self.curiosity.compute_reward(
                state_tensor, action_tensor, next_state_tensor
            )
            
            # Combinar recompensas
            combined_reward = reward + self.config.curiosity_weight * curiosity_reward
            
            # Almacenar experiencia en memoria
            self.memory.add(state, action, combined_reward, next_state, done, info)
            
            # Actualizar estado
            state = next_state
            
            # Actualizar contadores
            episode_reward += reward
            episode_curiosity += curiosity_reward.item()
            episode_steps += 1
            self.total_steps += 1
            
            # Entrenar si hay suficientes muestras
            if self.total_steps % self.config.train_frequency == 0 and len(self.memory) >= self.config.batch_size:
                loss = self.train_step()
                episode_loss += loss
            
            # Renderizar si es necesario
            if render:
                env.render()
            
            # Adaptar capas dinámicas periódicamente
            if self.total_steps % self.config.adaptation_frequency == 0:
                self.adapt_dynamic_layers()
        
        # Finalizar episodio en memoria
        self.memory.end_episode(success=info.get('success', False) if info else False)
        
        # Actualizar contador de episodios
        self.episodes_completed += 1
        
        # Calcular métricas promedio
        avg_loss = episode_loss / max(1, episode_steps // self.config.train_frequency)
        avg_curiosity = episode_curiosity / max(1, episode_steps)
        
        # Devolver métricas
        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'loss': avg_loss,
            'curiosity': avg_curiosity,
            'exploration_rate': self.exploration_rate
        }
    
    def train_step(self) -> float:
        """
        Realiza un paso de entrenamiento usando un lote de experiencias.
        
        Returns:
            Pérdida del paso de entrenamiento
        """
        # Muestrear lote de experiencias
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.config.batch_size)
        
        # Convertir a tensores
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Procesar estados actuales
        perception_output = self.perception(states_tensor)
        prediction_output = self.prediction(perception_output)
        combined_features = torch.cat([perception_output, prediction_output], dim=1)
        action_values = self.executive(combined_features)
        
        # Obtener valores Q para acciones tomadas
        q_values = action_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Calcular valores Q objetivo (Double DQN)
        with torch.no_grad():
            # Procesar estados siguientes
            next_perception_output = self.perception(next_states_tensor)
            next_prediction_output = self.prediction(next_perception_output)
            next_combined_features = torch.cat([next_perception_output, next_prediction_output], dim=1)
            next_action_values = self.executive(next_combined_features)
            
            # Seleccionar mejores acciones
            best_actions = torch.argmax(next_action_values, dim=1)
            
            # Calcular valores Q para mejores acciones
            next_q_values = next_action_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Calcular valores objetivo
            targets = rewards_tensor + (1 - dones_tensor) * self.config.gamma * next_q_values
        
        # Calcular pérdida
        loss = nn.functional.smooth_l1_loss(q_values, targets, reduction='none')
        
        # Aplicar pesos de prioridad
        weighted_loss = (loss * weights_tensor).mean()
        
        # Actualizar prioridades en memoria
        if self.memory.prioritized:
            td_errors = torch.abs(targets - q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)
        
        # Optimización
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Recortar gradientes para estabilidad
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        
        # Entrenar módulo de curiosidad
        curiosity_loss = self.curiosity.train(states_tensor, actions_tensor, next_states_tensor)
        
        # Entrenar meta-controlador periódicamente
        if self.total_steps % self.config.meta_train_frequency == 0:
            self.meta_controller.train(self.memory)
        
        # Entrenar biblioteca de habilidades periódicamente
        if self.total_steps % self.config.skill_train_frequency == 0:
            self.skill_library.train(self.memory)
        
        return weighted_loss.item()
    
    def adapt_dynamic_layers(self):
        """
        Adapta las capas dinámicas basándose en el rendimiento y la complejidad.
        """
        # Adaptar módulo de percepción
        if hasattr(self.perception, 'adapt'):
            self.perception.adapt()
        
        # Adaptar módulo de predicción
        if hasattr(self.prediction, 'adapt'):
            self.prediction.adapt()
        
        # Adaptar módulo ejecutivo
        if hasattr(self.executive, 'adapt'):
            self.executive.adapt()
        
        # Verificar dimensiones después de adaptación
        self.check_module_dimensions()
    
    def evaluate_episode(self, env: BaseEnvironment, render: bool = False) -> float:
        """
        Evalúa al agente durante un episodio completo.
        
        Args:
            env: Entorno de evaluación
            render: Si se debe renderizar el entorno
            
        Returns:
            Recompensa total del episodio
        """
        # Reiniciar entorno
        state = env.reset()
        done = False
        total_reward = 0.0
        
        # Bucle de episodio
        while not done:
            # Seleccionar acción (sin exploración)
            action = self.select_action(state, training=False)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, _ = env.step(action)
            
            # Actualizar estado y recompensa
            state = next_state
            total_reward += reward
            
            # Renderizar si es necesario
            if render:
                env.render()
        
        return total_reward
    
    def parameters(self):
        """
        Devuelve los parámetros del agente para optimización.
        
        Returns:
            Iterador de parámetros
        """
        return list(self.perception.parameters()) + \
               list(self.prediction.parameters()) + \
               list(self.executive.parameters())
    
    def check_module_dimensions(self):
        """
        Verifica y corrige inconsistencias dimensionales entre módulos.
        """
        try:
            # Verificar dimensiones de salida de percepción
            perception_output_size = self.perception.get_output_size()
            
            # Verificar dimensiones de entrada de predicción
            if self.prediction.get_input_size() != perception_output_size:
                print(f"Corrigiendo dimensión de entrada de predicción: {self.prediction.get_input_size()} -> {perception_output_size}")
                self.prediction.update_input_size(perception_output_size)
            
            # Verificar dimensiones de salida de predicción
            prediction_output_size = self.prediction.get_output_size()
            
            # Verificar dimensiones de entrada de ejecutivo
            expected_executive_input = perception_output_size + prediction_output_size
            if self.executive.get_input_size() != expected_executive_input:
                print(f"Corrigiendo dimensión de entrada de ejecutivo: {self.executive.get_input_size()} -> {expected_executive_input}")
                self.executive.update_input_size(expected_executive_input)
            
        except Exception as e:
            print(f"Error al verificar dimensiones: {e}")
    
    def save(self, filepath: str) -> None:
        """
        Guarda el estado del agente en disco.
        
        Args:
            filepath: Ruta donde guardar el agente
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar estado
        state = {
            'perception_state': self.perception.state_dict(),
            'prediction_state': self.prediction.state_dict(),
            'executive_state': self.executive.state_dict(),
            'curiosity_state': self.curiosity.state_dict(),
            'meta_controller_state': self.meta_controller.state_dict(),
            'skill_library_state': self.skill_library.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'observation_shape': self.observation_shape,
            'action_size': self.action_size,
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed,
            'exploration_rate': self.exploration_rate
        }
        
        torch.save(state, filepath)
        print(f"Agente guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el estado del agente desde disco.
        
        Args:
            filepath: Ruta desde donde cargar el agente
        """
        # Verificar que el archivo existe
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo {filepath}")
        
        # Cargar estado
        state = torch.load(filepath, map_location=self.device)
        
        # Verificar compatibilidad
        if state['observation_shape'] != self.observation_shape or state['action_size'] != self.action_size:
            print("Advertencia: Las dimensiones del modelo cargado no coinciden con las actuales.")
            print(f"Modelo: {state['observation_shape']}, {state['action_size']}")
            print(f"Actual: {self.observation_shape}, {self.action_size}")
        
        # Cargar estados de módulos
        self.perception.load_state_dict(state['perception_state'])
        self.prediction.load_state_dict(state['prediction_state'])
        self.executive.load_state_dict(state['executive_state'])
        self.curiosity.load_state_dict(state['curiosity_state'])
        self.meta_controller.load_state_dict(state['meta_controller_state'])
        self.skill_library.load_state_dict(state['skill_library_state'])
        
        # Cargar estado del optimizador
        self.optimizer.load_state_dict(state['optimizer_state'])
        
        # Cargar contadores y estadísticas
        self.total_steps = state['total_steps']
        self.episodes_completed = state['episodes_completed']
        self.exploration_rate = state['exploration_rate']
        
        # Cargar configuración
        self.config = state['config']
        
        print(f"Agente cargado desde {filepath}")
        print(f"Pasos totales: {self.total_steps}, Episodios completados: {self.episodes_completed}") 