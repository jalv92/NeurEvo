"""
Sistema de memoria episódica para el framework NeurEvo.
"""

import numpy as np
import torch
import random
from collections import deque
from typing import List, Tuple, Dict, Any, Union, Optional
import time
import os
import pickle

from neurevo.memory.episode import Episode, Experience

class EpisodicMemory:
    """
    Implementación de memoria episódica con soporte para priorización de experiencias.
    
    Almacena episodios completos y permite muestrear experiencias individuales
    para entrenamiento, con énfasis en aquellas con mayor error de predicción.
    """
    
    def __init__(self, capacity: int = 10000, observation_shape: Tuple[int, ...] = None, 
                action_size: int = None, prioritized: bool = True, alpha: float = 0.6, 
                beta: float = 0.4, beta_increment: float = 0.001):
        """
        Inicializa la memoria episódica.
        
        Args:
            capacity: Capacidad máxima de experiencias
            observation_shape: Forma de las observaciones
            action_size: Tamaño del espacio de acciones
            prioritized: Si se debe usar muestreo priorizado
            alpha: Factor de priorización (0 = uniforme, 1 = completamente priorizado)
            beta: Factor de corrección de sesgo (0 = sin corrección, 1 = corrección completa)
            beta_increment: Incremento de beta por cada muestreo
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Memoria para experiencias individuales
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Memoria para episodios completos
        self.episodes = []
        self.max_episodes = 1000
        self.current_episode = None
        self.episode_counter = 0
        
        # Estadísticas
        self.total_experiences = 0
        self.total_episodes = 0
        
        # Tiempo para calcular tasas de adición
        self.last_add_time = time.time()
        self.add_rates = deque(maxlen=100)
    
    def start_episode(self) -> None:
        """
        Inicia un nuevo episodio.
        """
        self.current_episode = Episode(episode_id=self.episode_counter)
        self.episode_counter += 1
    
    def end_episode(self, success: bool = False) -> Episode:
        """
        Finaliza el episodio actual y lo almacena.
        
        Args:
            success: Si el episodio fue exitoso
            
        Returns:
            El episodio finalizado
        """
        if self.current_episode is None:
            return None
        
        # Marcar éxito y calcular estadísticas
        self.current_episode.success = success
        self.current_episode.calculate_statistics()
        
        # Almacenar episodio
        self.episodes.append(self.current_episode)
        self.total_episodes += 1
        
        # Limitar número de episodios almacenados
        if len(self.episodes) > self.max_episodes:
            # Eliminar episodio más antiguo que no sea exitoso
            # Si todos son exitosos, eliminar el más antiguo
            for i, episode in enumerate(self.episodes):
                if not episode.success:
                    self.episodes.pop(i)
                    break
            else:
                self.episodes.pop(0)
        
        # Guardar referencia al episodio finalizado
        completed_episode = self.current_episode
        
        # Reiniciar episodio actual
        self.current_episode = None
        
        return completed_episode
    
    def add(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: bool, info: Dict[str, Any] = None) -> None:
        """
        Añade una experiencia a la memoria.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Estado siguiente
            done: Si el episodio ha terminado
            info: Información adicional
        """
        # Crear experiencia
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {}
        )
        
        # Añadir a episodio actual si existe
        if self.current_episode is not None:
            self.current_episode.add_experience(experience)
        
        # Añadir a memoria de experiencias
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.memory.append(experience)
            self.size += 1
        else:
            self.memory[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
        self.total_experiences += 1
        
        # Calcular tasa de adición
        current_time = time.time()
        self.add_rates.append(1.0 / (current_time - self.last_add_time))
        self.last_add_time = current_time
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un lote de experiencias de la memoria.
        
        Args:
            batch_size: Tamaño del lote a muestrear
            
        Returns:
            Tupla con (estados, acciones, recompensas, siguientes_estados, terminados)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        if self.prioritized:
            # Muestreo priorizado
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
            
            # Corrección de sesgo
            weights = (self.size * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()
            
            # Incrementar beta
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # Muestreo uniforme
            indices = np.random.choice(self.size, batch_size, replace=False)
            weights = np.ones_like(indices, dtype=np.float32)
        
        # Extraer experiencias
        batch = [self.memory[i] for i in indices]
        
        # Convertir a arrays
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], errors: np.ndarray) -> None:
        """
        Actualiza las prioridades de las experiencias.
        
        Args:
            indices: Índices de las experiencias a actualizar
            errors: Errores de predicción
        """
        if not self.prioritized:
            return
        
        # Añadir pequeño valor para evitar prioridad cero
        priorities = np.abs(errors) + 1e-6
        
        for i, priority in zip(indices, priorities):
            if i < self.size:
                self.priorities[i] = priority
    
    def get_recent_episodes(self, n: int = 10) -> List[Episode]:
        """
        Obtiene los episodios más recientes.
        
        Args:
            n: Número de episodios a obtener
            
        Returns:
            Lista de episodios recientes
        """
        return self.episodes[-n:]
    
    def get_best_episodes(self, n: int = 10) -> List[Episode]:
        """
        Obtiene los episodios con mayor recompensa acumulada.
        
        Args:
            n: Número de episodios a obtener
            
        Returns:
            Lista de mejores episodios
        """
        sorted_episodes = sorted(
            self.episodes, 
            key=lambda e: e.total_reward, 
            reverse=True
        )
        return sorted_episodes[:n]
    
    def get_episode_by_id(self, episode_id: int) -> Optional[Episode]:
        """
        Obtiene un episodio por su ID.
        
        Args:
            episode_id: ID del episodio a buscar
            
        Returns:
            Episodio encontrado o None
        """
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return episode
        return None
    
    def search_episodes_by_tag(self, tag: str) -> List[Episode]:
        """
        Busca episodios por etiqueta.
        
        Args:
            tag: Etiqueta a buscar
            
        Returns:
            Lista de episodios con la etiqueta
        """
        return [
            episode for episode in self.episodes
            if tag in episode.tags
        ]
    
    def clear(self) -> None:
        """
        Limpia la memoria.
        """
        self.memory.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        self.episodes.clear()
        self.current_episode = None
        
        # Mantener contadores para IDs únicos
        # self.episode_counter = 0
        
        # Reiniciar estadísticas
        self.total_experiences = 0
        self.total_episodes = 0
    
    def save(self, path: str) -> None:
        """
        Guarda la memoria en disco.
        
        Args:
            path: Ruta donde guardar la memoria
        """
        data = {
            'capacity': self.capacity,
            'observation_shape': self.observation_shape,
            'action_size': self.action_size,
            'prioritized': self.prioritized,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'position': self.position,
            'size': self.size,
            'episode_counter': self.episode_counter,
            'total_experiences': self.total_experiences,
            'total_episodes': self.total_episodes,
            'priorities': self.priorities,
            'episodes': self.episodes,
        }
        
        try:
            torch.save(data, path)
            print(f"Memoria episódica guardada en {path}")
        except Exception as e:
            print(f"Error al guardar memoria episódica: {e}")
    
    def load(self, path: str) -> None:
        """
        Carga la memoria desde disco.
        
        Args:
            path: Ruta desde donde cargar la memoria
        """
        try:
            data = torch.load(path)
            
            self.capacity = data['capacity']
            self.observation_shape = data['observation_shape']
            self.action_size = data['action_size']
            self.prioritized = data['prioritized']
            self.alpha = data['alpha']
            self.beta = data['beta']
            self.beta_increment = data['beta_increment']
            self.position = data['position']
            self.size = data['size']
            self.episode_counter = data['episode_counter']
            self.total_experiences = data['total_experiences']
            self.total_episodes = data['total_episodes']
            self.priorities = data['priorities']
            self.episodes = data['episodes']
            
            # Reconstruir memoria de experiencias
            self.memory = deque(maxlen=self.capacity)
            for episode in self.episodes:
                for exp in episode.experiences:
                    if len(self.memory) < self.capacity:
                        self.memory.append(exp)
            
            print(f"Memoria episódica cargada desde {path}")
        except Exception as e:
            print(f"Error al cargar memoria episódica: {e}")
    
    def __len__(self) -> int:
        """
        Obtiene el número de experiencias en la memoria.
        
        Returns:
            Número de experiencias
        """
        return self.size 