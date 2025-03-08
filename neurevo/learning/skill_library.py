"""
Biblioteca de habilidades transferibles para el framework NeurEvo.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
import os
import pickle
import time
import uuid

class Skill:
    """
    Representa una habilidad individual que puede ser reutilizada.
    Una habilidad es una política especializada para resolver una subtarea específica.
    """
    
    def __init__(self, 
                 name: str,
                 model: nn.Module,
                 observation_shape: Tuple[int, ...],
                 action_size: int,
                 precondition: Optional[Callable] = None,
                 postcondition: Optional[Callable] = None,
                 device: torch.device = None):
        """
        Inicializa una habilidad.
        
        Args:
            name: Nombre descriptivo de la habilidad
            model: Modelo de red neuronal que implementa la habilidad
            observation_shape: Forma del espacio de observación
            action_size: Tamaño del espacio de acciones
            precondition: Función que verifica si la habilidad es aplicable
            postcondition: Función que verifica si la habilidad ha completado su objetivo
            device: Dispositivo para cálculos (CPU/GPU)
        """
        self.name = name
        self.id = str(uuid.uuid4())[:8]
        self.model = model
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.precondition = precondition
        self.postcondition = postcondition
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Estadísticas de uso
        self.usage_count = 0
        self.success_count = 0
        self.creation_time = time.time()
        self.last_used_time = None
        
        # Mover modelo al dispositivo correcto
        self.model.to(self.device)
    
    def is_applicable(self, state: np.ndarray) -> bool:
        """
        Verifica si la habilidad es aplicable al estado actual.
        
        Args:
            state: Estado actual
            
        Returns:
            True si la habilidad es aplicable
        """
        if self.precondition is None:
            return True
        
        return self.precondition(state)
    
    def is_completed(self, state: np.ndarray) -> bool:
        """
        Verifica si la habilidad ha completado su objetivo.
        
        Args:
            state: Estado actual
            
        Returns:
            True si la habilidad ha completado su objetivo
        """
        if self.postcondition is None:
            return False
        
        return self.postcondition(state)
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Selecciona una acción basada en el estado actual.
        
        Args:
            state: Estado actual
            
        Returns:
            Acción seleccionada
        """
        # Actualizar estadísticas
        self.usage_count += 1
        self.last_used_time = time.time()
        
        # Convertir estado a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Seleccionar acción
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def record_success(self) -> None:
        """
        Registra un uso exitoso de la habilidad.
        """
        self.success_count += 1
    
    def get_success_rate(self) -> float:
        """
        Calcula la tasa de éxito de la habilidad.
        
        Returns:
            Tasa de éxito (0.0 - 1.0)
        """
        if self.usage_count == 0:
            return 0.0
        
        return self.success_count / self.usage_count
    
    def save(self, path: str) -> None:
        """
        Guarda la habilidad en disco.
        
        Args:
            path: Ruta donde guardar la habilidad
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Guardar modelo
        torch.save(self.model.state_dict(), f"{path}_model.pt")
        
        # Guardar metadatos
        metadata = {
            "name": self.name,
            "id": self.id,
            "observation_shape": self.observation_shape,
            "action_size": self.action_size,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "creation_time": self.creation_time,
            "last_used_time": self.last_used_time
        }
        
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump(metadata, f)
    
    def load(self, path: str, model_class: nn.Module) -> None:
        """
        Carga la habilidad desde disco.
        
        Args:
            path: Ruta desde donde cargar la habilidad
            model_class: Clase del modelo a cargar
        """
        # Cargar metadatos
        with open(f"{path}_meta.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        # Actualizar atributos
        self.name = metadata["name"]
        self.id = metadata["id"]
        self.observation_shape = metadata["observation_shape"]
        self.action_size = metadata["action_size"]
        self.usage_count = metadata["usage_count"]
        self.success_count = metadata["success_count"]
        self.creation_time = metadata["creation_time"]
        self.last_used_time = metadata["last_used_time"]
        
        # Cargar modelo
        self.model = model_class(self.observation_shape, self.action_size)
        self.model.load_state_dict(torch.load(f"{path}_model.pt", map_location=self.device))
        self.model.to(self.device)
    
    def __str__(self) -> str:
        """
        Representación en cadena de la habilidad.
        
        Returns:
            Cadena con información de la habilidad
        """
        return f"Skill({self.name}, id={self.id}, success_rate={self.get_success_rate():.2f})"


class SkillLibrary:
    """
    Biblioteca de habilidades transferibles que pueden ser reutilizadas
    por el agente para resolver tareas complejas.
    """
    
    def __init__(self, max_skills: int = 50):
        """
        Inicializa la biblioteca de habilidades.
        
        Args:
            max_skills: Número máximo de habilidades a almacenar
        """
        self.skills = {}  # Diccionario de habilidades por ID
        self.max_skills = max_skills
        self.skill_usage_history = []
    
    def add_skill(self, skill: Skill) -> str:
        """
        Añade una habilidad a la biblioteca.
        
        Args:
            skill: Habilidad a añadir
            
        Returns:
            ID de la habilidad añadida
        """
        # Verificar si ya existe una habilidad similar
        for existing_skill in self.skills.values():
            if existing_skill.name == skill.name:
                # Actualizar habilidad existente si la nueva es mejor
                if skill.get_success_rate() > existing_skill.get_success_rate():
                    self.skills[existing_skill.id] = skill
                    return existing_skill.id
                else:
                    return existing_skill.id
        
        # Eliminar habilidades menos útiles si se alcanza el límite
        if len(self.skills) >= self.max_skills:
            self._prune_skills()
        
        # Añadir nueva habilidad
        self.skills[skill.id] = skill
        return skill.id
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """
        Obtiene una habilidad por su ID.
        
        Args:
            skill_id: ID de la habilidad
            
        Returns:
            Habilidad o None si no existe
        """
        return self.skills.get(skill_id)
    
    def get_applicable_skills(self, state: np.ndarray) -> List[Skill]:
        """
        Obtiene todas las habilidades aplicables al estado actual.
        
        Args:
            state: Estado actual
            
        Returns:
            Lista de habilidades aplicables
        """
        return [skill for skill in self.skills.values() if skill.is_applicable(state)]
    
    def select_best_skill(self, state: np.ndarray) -> Optional[Skill]:
        """
        Selecciona la mejor habilidad para el estado actual.
        
        Args:
            state: Estado actual
            
        Returns:
            Mejor habilidad o None si no hay aplicables
        """
        applicable_skills = self.get_applicable_skills(state)
        
        if not applicable_skills:
            return None
        
        # Seleccionar habilidad con mayor tasa de éxito
        best_skill = max(applicable_skills, key=lambda s: s.get_success_rate())
        
        # Registrar uso
        self.skill_usage_history.append((best_skill.id, time.time()))
        
        return best_skill
    
    def _prune_skills(self) -> None:
        """
        Elimina las habilidades menos útiles de la biblioteca.
        """
        if not self.skills:
            return
        
        # Calcular utilidad de cada habilidad
        utilities = {}
        for skill_id, skill in self.skills.items():
            # Combinar tasa de éxito y frecuencia de uso
            success_rate = skill.get_success_rate()
            usage_frequency = skill.usage_count / max(1, (time.time() - skill.creation_time) / 86400)  # Usos por día
            
            # Utilidad = tasa de éxito * frecuencia de uso
            utilities[skill_id] = success_rate * usage_frequency
        
        # Ordenar habilidades por utilidad
        sorted_skills = sorted(utilities.items(), key=lambda x: x[1])
        
        # Eliminar las menos útiles
        skills_to_remove = len(self.skills) - self.max_skills + 1
        for i in range(skills_to_remove):
            if i < len(sorted_skills):
                skill_id = sorted_skills[i][0]
                del self.skills[skill_id]
    
    def save(self, directory: str) -> None:
        """
        Guarda todas las habilidades en disco.
        
        Args:
            directory: Directorio donde guardar las habilidades
        """
        os.makedirs(directory, exist_ok=True)
        
        # Guardar cada habilidad
        for skill_id, skill in self.skills.items():
            skill_path = os.path.join(directory, skill_id)
            skill.save(skill_path)
        
        # Guardar índice de habilidades
        index = {
            "skills": list(self.skills.keys()),
            "max_skills": self.max_skills
        }
        
        with open(os.path.join(directory, "index.pkl"), "wb") as f:
            pickle.dump(index, f)
    
    def load(self, directory: str, model_class: nn.Module) -> None:
        """
        Carga todas las habilidades desde disco.
        
        Args:
            directory: Directorio desde donde cargar las habilidades
            model_class: Clase del modelo a cargar
        """
        # Cargar índice de habilidades
        index_path = os.path.join(directory, "index.pkl")
        if not os.path.exists(index_path):
            return
        
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        
        self.max_skills = index["max_skills"]
        
        # Cargar cada habilidad
        for skill_id in index["skills"]:
            skill_path = os.path.join(directory, skill_id)
            
            # Verificar que existen los archivos
            if os.path.exists(f"{skill_path}_meta.pkl") and os.path.exists(f"{skill_path}_model.pt"):
                skill = Skill("", None, (0,), 0)
                skill.load(skill_path, model_class)
                self.skills[skill_id] = skill
    
    def get_skill_count(self) -> int:
        """
        Obtiene el número de habilidades en la biblioteca.
        
        Returns:
            Número de habilidades
        """
        return len(self.skills)
    
    def get_average_success_rate(self) -> float:
        """
        Calcula la tasa de éxito promedio de todas las habilidades.
        
        Returns:
            Tasa de éxito promedio
        """
        if not self.skills:
            return 0.0
        
        return sum(skill.get_success_rate() for skill in self.skills.values()) / len(self.skills)
    
    def clear(self) -> None:
        """
        Limpia la biblioteca de habilidades.
        """
        self.skills = {}
        self.skill_usage_history = [] 