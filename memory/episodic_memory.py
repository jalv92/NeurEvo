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

from neurevo.memory.episode import Episode, Transition

class EpisodicMemory:
    """
    Sistema de memoria episódica que almacena experiencias para aprendizaje por refuerzo
    y proporciona funcionalidades para recuperación y análisis de experiencias pasadas.
    """
    
    def __init__(self, capacity: int = 10000, observation_shape: Tuple[int, ...] = None, 
                action_size: int = None, prioritized: bool = True, alpha: float = 0.6, 
                beta: float = 0.4, beta_increment: float = 0.001):
        """
        Inicializa el sistema de memoria episódica.
        
        Args:
            capacity: Capacidad máxima del buffer de experiencia
            observation_shape: Forma de las observaciones
            action_size: Tamaño del espacio de acciones
            prioritized: Si se debe usar muestreo prioritizado
            alpha: Exponente de prioridad (0 = uniforme, 1 = completamente prioritizado)
            beta: Exponente para corrección de sesgo (0 = sin corrección, 1 = corrección completa)
            beta_increment: Incremento de beta por cada muestreo
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Buffer de experiencia para entrenamiento rápido
        self.buffer = deque(maxlen=capacity)
        
        # Prioridades para muestreo prioritizado
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        # Episodios completos para análisis y meta-aprendizaje
        self.episodes = []
        self.current_episode = None
        
        # Índices para búsqueda rápida
        self.state_index = {}  # Mapeo de estados a índices
        self.action_index = {}  # Mapeo de acciones a índices
        self.reward_index = {}  # Mapeo de recompensas a índices
        
        # Estadísticas
        self.stats = {
            'total_transitions': 0,
            'total_episodes': 0,
            'avg_reward': 0.0,
            'avg_episode_length': 0.0,
            'success_rate': 0.0
        }
    
    def start_episode(self) -> None:
        """
        Inicia un nuevo episodio.
        """
        episode_id = self.stats['total_episodes'] + 1
        self.current_episode = Episode(episode_id=episode_id)
    
    def end_episode(self, success: bool = False) -> Episode:
        """
        Finaliza el episodio actual y lo añade a la memoria.
        
        Args:
            success: Si el episodio fue exitoso
            
        Returns:
            El episodio finalizado
        """
        if self.current_episode is None:
            return None
        
        # Actualizar metadatos del episodio
        if self.current_episode.metadata['end_time'] is None:
            self.current_episode.metadata['end_time'] = time.time()
            self.current_episode.metadata['duration'] = (
                self.current_episode.metadata['end_time'] - self.current_episode.metadata['start_time']
            )
        self.current_episode.metadata['success'] = success
        
        # Añadir episodio a la memoria
        self.episodes.append(self.current_episode)
        
        # Actualizar estadísticas
        self.stats['total_episodes'] += 1
        self.stats['avg_reward'] = (
            (self.stats['avg_reward'] * (self.stats['total_episodes'] - 1) + 
             self.current_episode.get_total_reward()) / self.stats['total_episodes']
        )
        self.stats['avg_episode_length'] = (
            (self.stats['avg_episode_length'] * (self.stats['total_episodes'] - 1) + 
             len(self.current_episode)) / self.stats['total_episodes']
        )
        self.stats['success_rate'] = (
            (self.stats['success_rate'] * (self.stats['total_episodes'] - 1) + 
             (1 if success else 0)) / self.stats['total_episodes']
        )
        
        # Guardar referencia al episodio finalizado
        finished_episode = self.current_episode
        
        # Iniciar nuevo episodio
        self.current_episode = None
        
        return finished_episode
    
    def add(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: bool, info: Dict[str, Any] = None) -> None:
        """
        Añade una transición a la memoria.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Estado siguiente
            done: Indicador de fin de episodio
            info: Información adicional
        """
        # Añadir a buffer de experiencia
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        
        # Actualizar prioridades para muestreo prioritizado
        if self.prioritized:
            idx = len(self.buffer) - 1
            if idx < len(self.priorities):
                self.priorities[idx] = self.max_priority
        
        # Añadir a episodio actual si existe
        if self.current_episode is not None:
            self.current_episode.add_transition(state, action, reward, next_state, done, info)
        
        # Si es el final del episodio, finalizarlo
        if done and self.current_episode is not None:
            self.end_episode(success=info.get('success', False) if info else False)
        
        # Actualizar estadísticas
        self.stats['total_transitions'] += 1
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un lote de transiciones de la memoria.
        
        Args:
            batch_size: Tamaño del lote
            
        Returns:
            Tupla de arrays (estados, acciones, recompensas, siguientes estados, terminados)
        """
        # Verificar que hay suficientes muestras
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            batch_size = buffer_size
        
        # Muestreo prioritizado
        if self.prioritized and buffer_size > 0:
            # Calcular probabilidades de muestreo
            priorities = self.priorities[:buffer_size]
            probs = priorities ** self.alpha
            probs = probs / np.sum(probs)
            
            # Muestrear índices
            indices = np.random.choice(buffer_size, batch_size, replace=False, p=probs)
            
            # Calcular pesos para corrección de sesgo
            weights = (buffer_size * probs[indices]) ** (-self.beta)
            weights = weights / np.max(weights)
            
            # Incrementar beta para futuros muestreos
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # Muestreo uniforme
            indices = np.random.choice(buffer_size, batch_size, replace=False)
            weights = np.ones(batch_size)
        
        # Extraer transiciones
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertir a arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones
    
    def update_priorities(self, indices: List[int], errors: np.ndarray) -> None:
        """
        Actualiza las prioridades de las transiciones basándose en los errores TD.
        
        Args:
            indices: Índices de las transiciones
            errors: Errores TD correspondientes
        """
        if not self.prioritized:
            return
        
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                # Añadir pequeño valor para evitar prioridad cero
                priority = (abs(error) + 1e-5) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def get_recent_episodes(self, n: int = 10) -> List[Episode]:
        """
        Devuelve los episodios más recientes.
        
        Args:
            n: Número de episodios a devolver
            
        Returns:
            Lista de episodios
        """
        return self.episodes[-n:]
    
    def get_best_episodes(self, n: int = 10) -> List[Episode]:
        """
        Devuelve los mejores episodios según la recompensa total.
        
        Args:
            n: Número de episodios a devolver
            
        Returns:
            Lista de episodios
        """
        sorted_episodes = sorted(self.episodes, 
                                key=lambda e: e.get_total_reward(), 
                                reverse=True)
        return sorted_episodes[:n]
    
    def get_episode_by_id(self, episode_id: int) -> Optional[Episode]:
        """
        Busca un episodio por su ID.
        
        Args:
            episode_id: ID del episodio
            
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
        return [episode for episode in self.episodes if episode.has_tag(tag)]
    
    def clear(self) -> None:
        """
        Limpia la memoria.
        """
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.episodes = []
        self.current_episode = None
        self.state_index = {}
        self.action_index = {}
        self.reward_index = {}
        
        # Reiniciar estadísticas
        self.stats = {
            'total_transitions': 0,
            'total_episodes': 0,
            'avg_reward': 0.0,
            'avg_episode_length': 0.0,
            'success_rate': 0.0
        }
    
    def save(self, path: str) -> None:
        """
        Guarda la memoria en disco.
        
        Args:
            path: Ruta donde guardar la memoria
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Guardar solo los episodios (el buffer es temporal)
        data = {
            'episodes': self.episodes,
            'stats': self.stats,
            'capacity': self.capacity,
            'observation_shape': self.observation_shape,
            'action_size': self.action_size,
            'prioritized': self.prioritized,
            'alpha': self.alpha,
            'beta': self.beta
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """
        Carga la memoria desde disco.
        
        Args:
            path: Ruta desde donde cargar la memoria
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.episodes = data['episodes']
        self.stats = data['stats']
        self.capacity = data['capacity']
        self.observation_shape = data['observation_shape']
        self.action_size = data['action_size']
        self.prioritized = data['prioritized']
        self.alpha = data['alpha']
        self.beta = data['beta']
        
        # Reconstruir buffer a partir de episodios recientes
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        
        # Añadir transiciones de episodios recientes al buffer
        recent_episodes = self.get_recent_episodes(5)
        for episode in recent_episodes:
            for transition in episode.get_transitions():
                self.buffer.append((
                    transition.state,
                    transition.action,
                    transition.reward,
                    transition.next_state,
                    transition.done
                ))
                
                # Actualizar prioridades
                if self.prioritized:
                    idx = len(self.buffer) - 1
                    if idx < len(self.priorities):
                        self.priorities[idx] = self.max_priority
    
    def __len__(self) -> int:
        """
        Devuelve el número de transiciones en el buffer.
        
        Returns:
            Número de transiciones
        """
        return len(self.buffer) 