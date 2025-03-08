"""
Clase de episodio individual para el sistema de memoria episódica.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
import time

@dataclass
class Transition:
    """
    Representa una transición individual en un episodio.
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class Episode:
    """
    Representa un episodio completo de interacción con el entorno.
    Almacena una secuencia de transiciones y proporciona métodos para
    acceder y analizar los datos del episodio.
    """
    
    def __init__(self, episode_id: int = None, max_size: int = 1000):
        """
        Inicializa un nuevo episodio.
        
        Args:
            episode_id: Identificador único del episodio
            max_size: Tamaño máximo del episodio (para evitar consumo excesivo de memoria)
        """
        self.episode_id = episode_id if episode_id is not None else int(time.time())
        self.transitions = []
        self.max_size = max_size
        self.metadata = {
            'start_time': time.time(),
            'end_time': None,
            'total_reward': 0.0,
            'success': False,
            'length': 0
        }
        self.tags = set()
    
    def add_transition(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool, info: Dict[str, Any] = None) -> None:
        """
        Añade una nueva transición al episodio.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Estado siguiente
            done: Indicador de fin de episodio
            info: Información adicional
        """
        # Verificar si el episodio ya está lleno
        if len(self.transitions) >= self.max_size:
            return
        
        # Crear y añadir la transición
        transition = Transition(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            info=info or {}
        )
        self.transitions.append(transition)
        
        # Actualizar metadatos
        self.metadata['total_reward'] += reward
        self.metadata['length'] += 1
        
        # Si es el final del episodio, actualizar metadatos finales
        if done:
            self.metadata['end_time'] = time.time()
            self.metadata['duration'] = self.metadata['end_time'] - self.metadata['start_time']
            
            # Determinar si el episodio fue exitoso (basado en info si está disponible)
            if info and 'success' in info:
                self.metadata['success'] = info['success']
    
    def get_transitions(self) -> List[Transition]:
        """
        Devuelve todas las transiciones del episodio.
        
        Returns:
            Lista de transiciones
        """
        return self.transitions
    
    def get_states(self) -> np.ndarray:
        """
        Devuelve todos los estados del episodio.
        
        Returns:
            Array con los estados
        """
        if not self.transitions:
            return np.array([])
        return np.array([t.state for t in self.transitions])
    
    def get_actions(self) -> np.ndarray:
        """
        Devuelve todas las acciones del episodio.
        
        Returns:
            Array con las acciones
        """
        if not self.transitions:
            return np.array([])
        return np.array([t.action for t in self.transitions])
    
    def get_rewards(self) -> np.ndarray:
        """
        Devuelve todas las recompensas del episodio.
        
        Returns:
            Array con las recompensas
        """
        if not self.transitions:
            return np.array([])
        return np.array([t.reward for t in self.transitions])
    
    def get_total_reward(self) -> float:
        """
        Devuelve la recompensa total del episodio.
        
        Returns:
            Recompensa total
        """
        return self.metadata['total_reward']
    
    def get_length(self) -> int:
        """
        Devuelve la longitud del episodio.
        
        Returns:
            Número de transiciones
        """
        return len(self.transitions)
    
    def get_discounted_returns(self, gamma: float = 0.99) -> np.ndarray:
        """
        Calcula los retornos descontados para cada paso del episodio.
        
        Args:
            gamma: Factor de descuento
            
        Returns:
            Array con los retornos descontados
        """
        rewards = self.get_rewards()
        returns = np.zeros_like(rewards)
        
        # Calcular retornos desde el final hacia el principio
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def add_tag(self, tag: str) -> None:
        """
        Añade una etiqueta al episodio para facilitar la búsqueda.
        
        Args:
            tag: Etiqueta a añadir
        """
        self.tags.add(tag)
    
    def has_tag(self, tag: str) -> bool:
        """
        Verifica si el episodio tiene una etiqueta específica.
        
        Args:
            tag: Etiqueta a verificar
            
        Returns:
            True si el episodio tiene la etiqueta
        """
        return tag in self.tags
    
    def to_tensor_batch(self, device: torch.device = None) -> Tuple[torch.Tensor, ...]:
        """
        Convierte el episodio en un lote de tensores para entrenamiento.
        
        Args:
            device: Dispositivo para los tensores
            
        Returns:
            Tupla de tensores (estados, acciones, recompensas, siguientes estados, terminados)
        """
        if not self.transitions:
            return None
        
        # Extraer componentes
        states = np.array([t.state for t in self.transitions])
        actions = np.array([t.action for t in self.transitions])
        rewards = np.array([t.reward for t in self.transitions])
        next_states = np.array([t.next_state for t in self.transitions])
        dones = np.array([t.done for t in self.transitions])
        
        # Convertir a tensores
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Mover a dispositivo si es necesario
        if device:
            states_tensor = states_tensor.to(device)
            actions_tensor = actions_tensor.to(device)
            rewards_tensor = rewards_tensor.to(device)
            next_states_tensor = next_states_tensor.to(device)
            dones_tensor = dones_tensor.to(device)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def __len__(self) -> int:
        """
        Devuelve la longitud del episodio.
        
        Returns:
            Número de transiciones
        """
        return len(self.transitions)
    
    def __str__(self) -> str:
        """
        Representación en cadena del episodio.
        
        Returns:
            Cadena con información del episodio
        """
        return f"Episode {self.episode_id}: {len(self.transitions)} steps, " \
               f"reward: {self.metadata['total_reward']:.2f}, " \
               f"success: {self.metadata['success']}" 