"""
Registry - Sistema de registro para componentes dinámicos en NeurEvo.

Este módulo proporciona mecanismos para registrar y recuperar diferentes tipos
de componentes, como entornos, algoritmos de aprendizaje, etc.
"""

from typing import Dict, Any, Optional, Callable, Type, List


class ComponentRegistry:
    """
    Registro flexible para componentes del framework.
    
    Permite registrar y recuperar diferentes tipos de componentes
    de forma dinámica, facilitando la extensibilidad del framework.
    """
    
    def __init__(self):
        """
        Inicializa un nuevo registro de componentes.
        """
        self._registry = {}
    
    def register(self, category: str, name: str, component: Any) -> None:
        """
        Registra un componente en una categoría específica.
        
        Args:
            category: Categoría del componente (ej: "environment", "module")
            name: Nombre único para el componente dentro de su categoría
            component: El componente a registrar (clase, función, objeto, etc.)
            
        Raises:
            ValueError: Si el nombre ya está registrado en esa categoría
        """
        if category not in self._registry:
            self._registry[category] = {}
            
        if name in self._registry[category]:
            raise ValueError(
                f"Componente '{name}' ya registrado en categoría '{category}'"
            )
            
        self._registry[category][name] = component
    
    def get(self, category: str, name: str) -> Any:
        """
        Recupera un componente del registro.
        
        Args:
            category: Categoría del componente
            name: Nombre del componente
            
        Returns:
            El componente registrado
            
        Raises:
            ValueError: Si la categoría o nombre no existen
        """
        if category not in self._registry:
            raise ValueError(f"Categoría '{category}' no existe en el registro")
            
        if name not in self._registry[category]:
            raise ValueError(
                f"Componente '{name}' no encontrado en categoría '{category}'"
            )
            
        return self._registry[category][name]
    
    def contains(self, category: str, name: str) -> bool:
        """
        Verifica si un componente está registrado.
        
        Args:
            category: Categoría a verificar
            name: Nombre del componente a verificar
            
        Returns:
            True si el componente existe, False en caso contrario
        """
        return (
            category in self._registry and 
            name in self._registry[category]
        )
    
    def list_components(self, category: str) -> List[str]:
        """
        Lista todos los componentes en una categoría.
        
        Args:
            category: Categoría cuyos componentes se listarán
            
        Returns:
            Lista de nombres de componentes registrados en esa categoría
        """
        if category not in self._registry:
            return []
            
        return list(self._registry[category].keys())
    
    def list_categories(self) -> List[str]:
        """
        Lista todas las categorías registradas.
        
        Returns:
            Lista de nombres de categorías
        """
        return list(self._registry.keys())


# Decoradores para facilitar el registro

def register_component(registry: ComponentRegistry, category: str, name: Optional[str] = None):
    """
    Decorador para registrar un componente.
    
    Args:
        registry: Registro donde se añadirá el componente
        category: Categoría del componente
        name: Nombre opcional para el componente (usa el nombre de la clase por defecto)
        
    Returns:
        Decorador que registra el componente
    """
    def decorator(component_class):
        component_name = name if name is not None else component_class.__name__
        registry.register(category, component_name, component_class)
        return component_class
    return decorator 