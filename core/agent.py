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
from neurevo.utils.memory_manager import MemoryManager
from neurevo.core.base_environment import BaseEnvironment

class NeurEvoAgent:
    """
    Agente principal del framework NeurEvo.
    Integra módulos cognitivos, memoria episódica y aprendizaje por refuerzo.
    """
    
    def __init__(self, observation_shape: Union[Tuple[int, ...], List[int]], 
                 action_size: int, device: torch.device = None, **kwargs):
        """
        Inicializa un agente NeurEvo.
        
        Args:
            observation_shape: Forma del espacio de observación
            action_size: Tamaño del espacio de acciones
            device: Dispositivo para cálculos (CPU/GPU)
            **kwargs: Parámetros adicionales de configuración
        """
        # Configuración básica
        self.observation_shape = tuple(observation_shape) if isinstance(observation_shape, list) else observation_shape
        self.action_size = action_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Parámetros de aprendizaje
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon_start", 1.0)
        self.epsilon_end = kwargs.get("epsilon_end", 0.05)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.batch_size = kwargs.get("batch_size", 64)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        
        # Inicializar módulos cognitivos
        self.perception = PerceptionModule(
            input_shape=self.observation_shape,
            hidden_layers=kwargs.get("hidden_layers", [128, 128]),
            device=self.device
        )
        
        self.prediction = PredictionModule(
            input_shape=self.perception.output_shape,
            horizon=kwargs.get("prediction_horizon", 5),
            device=self.device
        )
        
        self.executive = ExecutiveModule(
            input_shape=self.perception.output_shape,
            action_size=self.action_size,
            device=self.device
        )
        
        # Inicializar memoria y aprendizaje
        self.memory = EpisodicMemory(
            capacity=kwargs.get("memory_size", 10000),
            observation_shape=self.observation_shape,
            action_size=self.action_size
        )
        
        self.curiosity = CuriosityModule(
            input_shape=self.perception.output_shape,
            action_size=self.action_size,
            device=self.device,
            weight=kwargs.get("curiosity_weight", 0.1)
        )
        
        self.meta_controller = MetaController()
        self.skill_library = SkillLibrary()
        
        # Inicializar optimizador
        self.optimizer = optim.Adam(
            list(self.perception.parameters()) +
            list(self.prediction.parameters()) +
            list(self.executive.parameters()) +
            list(self.curiosity.parameters()),
            lr=self.learning_rate,
            weight_decay=kwargs.get("weight_decay", 0.0001)
        )
        
        # Estadísticas de entrenamiento
        self.episodes_trained = 0
        self.losses = []
        self.rewards = []
        self.q_values = []
        
        # Administrador de memoria
        self.memory_manager = MemoryManager(self)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción basada en el estado actual.
        
        Args:
            state: Estado actual del entorno
            training: Si está en modo entrenamiento (para exploración)
            
        Returns:
            Acción seleccionada
        """
        # Convertir estado a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Exploración aleatoria en entrenamiento
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Procesamiento cognitivo
        with torch.no_grad():
            # Percepción
            features = self.perception(state_tensor)
            
            # Predicción (no afecta la selección de acción directamente)
            self.prediction(features)
            
            # Toma de decisiones
            q_values = self.executive(features)
            
            # Registrar valor Q promedio para seguimiento
            self.q_values.append(q_values.mean().item())
            
            # Seleccionar acción con mayor valor Q
            return q_values.argmax(dim=1).item()
    
    def train_episode(self, env: BaseEnvironment, render: bool = False) -> Dict[str, float]:
        """
        Entrena al agente durante un episodio completo.
        
        Args:
            env: Entorno de entrenamiento
            render: Si se debe renderizar el entorno
            
        Returns:
            Métricas del episodio
        """
        # Reiniciar entorno
        state = env.reset()
        done = False
        total_reward = 0
        episode_loss = 0
        steps = 0
        
        # Bucle de episodio
        while not done:
            # Seleccionar y ejecutar acción
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Calcular recompensa intrínseca por curiosidad
            intrinsic_reward = self.curiosity.compute_reward(
                torch.FloatTensor(state).unsqueeze(0).to(self.device),
                torch.tensor([action]).to(self.device),
                torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            )
            
            # Combinar recompensas
            combined_reward = reward + intrinsic_reward.item()
            
            # Almacenar experiencia
            self.memory.add(state, action, combined_reward, next_state, done)
            
            # Actualizar estado y recompensa
            state = next_state
            total_reward += reward  # Solo la recompensa externa para estadísticas
            steps += 1
            
            # Renderizar si es necesario
            if render:
                env.render()
            
            # Entrenar si hay suficientes muestras
            if len(self.memory) > self.batch_size:
                loss = self.train_step()
                episode_loss += loss
        
        # Actualizar epsilon para exploración
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Actualizar estadísticas
        self.episodes_trained += 1
        self.rewards.append(total_reward)
        
        # Limpieza de memoria después del episodio
        self.memory_manager.after_episode_cleanup()
        
        # Devolver métricas
        return {
            "reward": total_reward,
            "loss": episode_loss / max(1, steps),
            "epsilon": self.epsilon,
            "steps": steps,
            "q_value": np.mean(self.q_values[-steps:]) if steps > 0 else 0
        }
    
    def train_step(self) -> float:
        """
        Realiza un paso de entrenamiento con un lote de experiencias.
        
        Returns:
            Pérdida del paso de entrenamiento
        """
        # Muestrear lote de experiencias
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calcular valores Q actuales
        features = self.perception(states)
        q_values = self.executive(features)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calcular valores Q objetivo
        with torch.no_grad():
            next_features = self.perception(next_states)
            next_q_values = self.executive(next_features)
            max_next_q = next_q_values.max(1)[0]
            q_targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Calcular pérdida de valor Q
        q_loss = nn.MSELoss()(q_values, q_targets)
        
        # Calcular pérdida de predicción
        pred_loss = self.prediction.compute_loss(features, next_features)
        
        # Calcular pérdida de curiosidad
        curiosity_loss = self.curiosity.compute_loss(states, actions, next_states)
        
        # Pérdida total
        total_loss = q_loss + 0.5 * pred_loss + 0.1 * curiosity_loss
        
        # Optimización
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Recorte de gradiente
        self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate_episode(self, env: BaseEnvironment, render: bool = False) -> float:
        """
        Evalúa al agente durante un episodio completo sin entrenamiento.
        
        Args:
            env: Entorno de evaluación
            render: Si se debe renderizar el entorno
            
        Returns:
            Recompensa total del episodio
        """
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Seleccionar acción sin exploración
            action = self.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            # Actualizar estado y recompensa
            state = next_state
            total_reward += reward
            
            # Renderizar si es necesario
            if render:
                env.render()
        
        return total_reward
    
    def parameters(self):
        """
        Devuelve los parámetros de todos los módulos para optimización.
        
        Returns:
            Iterador de parámetros
        """
        for module in [self.perception, self.prediction, self.executive, self.curiosity]:
            for param in module.parameters():
                yield param
    
    def save(self, filename: str) -> None:
        """
        Guarda el estado del agente en disco.
        
        Args:
            filename: Ruta del archivo donde guardar el agente
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Guardar estado
        state = {
            # Módulos
            'perception': self.perception.state_dict(),
            'prediction': self.prediction.state_dict(),
            'executive': self.executive.state_dict(),
            'curiosity': self.curiosity.state_dict(),
            
            # Optimizador
            'optimizer': self.optimizer.state_dict(),
            
            # Otros atributos
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'losses': self.losses,
            'rewards': self.rewards,
            'q_values': self.q_values,
            
            # Componentes adicionales
            'skill_library': self.skill_library.skills,
            'meta_controller': {
                'curriculum_level': self.meta_controller.curriculum_level,
                'strategy': self.meta_controller.current_strategy
            }
        }
        
        torch.save(state, filename)
        print(f"Agente guardado en {filename}")
    
    def load(self, filename: str) -> int:
        """
        Carga el estado del agente desde disco.
        
        Args:
            filename: Ruta del archivo desde donde cargar el agente
            
        Returns:
            Número de episodios entrenados
        """
        # Verificar que el archivo existe
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No se encontró el archivo {filename}")
        
        # Cargar estado
        state = torch.load(filename, map_location=self.device)
        
        # Cargar módulos
        self.perception.load_state_dict(state['perception'])
        self.prediction.load_state_dict(state['prediction'])
        self.executive.load_state_dict(state['executive'])
        self.curiosity.load_state_dict(state['curiosity'])
        
        # Cargar optimizador
        self.optimizer.load_state_dict(state['optimizer'])
        
        # Cargar otros atributos
        self.epsilon = state['epsilon']
        self.episodes_trained = state['episodes_trained']
        self.losses = state['losses']
        self.rewards = state['rewards']
        self.q_values = state['q_values']
        
        # Cargar biblioteca de habilidades
        if 'skill_library' in state:
            self.skill_library.skills = state['skill_library']
        
        # Cargar meta-controlador
        if 'meta_controller' in state:
            self.meta_controller.curriculum_level = state['meta_controller']['curriculum_level']
            self.meta_controller.current_strategy = state['meta_controller']['strategy']
        
        print(f"Agente cargado desde {filename}")
        return self.episodes_trained 