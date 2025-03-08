"""
Implementación de episodios y experiencias para el sistema de memoria episódica.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Experience:
    """
    Representa una experiencia individual (transición) en un episodio.
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
    
    Un episodio es una secuencia de experiencias desde un estado inicial
    hasta un estado terminal o hasta que se interrumpe externamente.
    """
    
    def __init__(self, episode_id: int):
        """
        Inicializa un nuevo episodio.
        
        Args:
            episode_id: Identificador único del episodio
        """
        self.episode_id = episode_id
        self.experiences = []
        self.start_time = time.time()
        self.end_time = None
        self.success = False
        self.tags = set()
        
        # Métricas calculadas
        self.total_reward = 0.0
        self.avg_reward = 0.0
        self.min_reward = float('inf')
        self.max_reward = float('-inf')
        self.duration = 0.0
    
    def add_experience(self, experience: Experience) -> None:
        """
        Añade una experiencia al episodio.
        
        Args:
            experience: Experiencia a añadir
        """
        self.experiences.append(experience)
        
        # Actualizar métricas en tiempo real
        self.total_reward += experience.reward
        self.min_reward = min(self.min_reward, experience.reward)
        self.max_reward = max(self.max_reward, experience.reward)
        
        if len(self.experiences) > 0:
            self.avg_reward = self.total_reward / len(self.experiences)
    
    def calculate_statistics(self) -> None:
        """
        Calcula estadísticas del episodio.
        """
        # Actualizar tiempo de finalización
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        # Calcular métricas finales
        if self.experiences:
            self.total_reward = sum(exp.reward for exp in self.experiences)
            self.avg_reward = self.total_reward / len(self.experiences)
            self.min_reward = min(exp.reward for exp in self.experiences)
            self.max_reward = max(exp.reward for exp in self.experiences)
    
    def add_tag(self, tag: str) -> None:
        """
        Añade una etiqueta al episodio.
        
        Args:
            tag: Etiqueta a añadir
        """
        self.tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """
        Elimina una etiqueta del episodio.
        
        Args:
            tag: Etiqueta a eliminar
        """
        if tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """
        Verifica si el episodio tiene una etiqueta.
        
        Args:
            tag: Etiqueta a verificar
            
        Returns:
            True si el episodio tiene la etiqueta, False en caso contrario
        """
        return tag in self.tags
    
    def get_experience_at(self, index: int) -> Optional[Experience]:
        """
        Obtiene la experiencia en un índice específico.
        
        Args:
            index: Índice de la experiencia
            
        Returns:
            Experiencia en el índice o None si el índice es inválido
        """
        if 0 <= index < len(self.experiences):
            return self.experiences[index]
        return None
    
    def get_state_at(self, index: int) -> Optional[np.ndarray]:
        """
        Obtiene el estado en un índice específico.
        
        Args:
            index: Índice del estado
            
        Returns:
            Estado en el índice o None si el índice es inválido
        """
        exp = self.get_experience_at(index)
        return exp.state if exp else None
    
    def get_action_at(self, index: int) -> Optional[int]:
        """
        Obtiene la acción en un índice específico.
        
        Args:
            index: Índice de la acción
            
        Returns:
            Acción en el índice o None si el índice es inválido
        """
        exp = self.get_experience_at(index)
        return exp.action if exp else None
    
    def get_trajectory(self) -> List[np.ndarray]:
        """
        Obtiene la trayectoria completa de estados del episodio.
        
        Returns:
            Lista de estados
        """
        return [exp.state for exp in self.experiences]
    
    def get_action_sequence(self) -> List[int]:
        """
        Obtiene la secuencia completa de acciones del episodio.
        
        Returns:
            Lista de acciones
        """
        return [exp.action for exp in self.experiences]
    
    def get_reward_sequence(self) -> List[float]:
        """
        Obtiene la secuencia completa de recompensas del episodio.
        
        Returns:
            Lista de recompensas
        """
        return [exp.reward for exp in self.experiences]
    
    def __len__(self) -> int:
        """
        Obtiene la longitud del episodio (número de experiencias).
        
        Returns:
            Número de experiencias en el episodio
        """
        return len(self.experiences)
    
    def __repr__(self) -> str:
        """
        Representación en cadena del episodio.
        
        Returns:
            Cadena con información del episodio
        """
        return (
            f"Episode(id={self.episode_id}, "
            f"length={len(self)}, "
            f"total_reward={self.total_reward:.2f}, "
            f"success={self.success})"
        ) 