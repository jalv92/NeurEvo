"""
Configuraciones y constantes para el framework NeurEvo.
"""

from typing import Dict, Any, Optional

class NeurEvoConfig:
    """
    Clase de configuración para el framework NeurEvo.
    Contiene parámetros por defecto y permite personalización.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la configuración con valores por defecto o personalizados.
        
        Args:
            config: Diccionario opcional con configuraciones personalizadas
        """
        # Valores por defecto
        self.defaults = {
            # Parámetros generales
            "seed": 42,
            "verbose": True,
            
            # Parámetros de red neuronal
            "hidden_layers": [128, 128],
            "learning_rate": 0.001,
            "activation": "relu",
            "use_batch_norm": True,
            
            # Parámetros de aprendizaje por refuerzo
            "gamma": 0.99,  # Factor de descuento
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.995,
            "batch_size": 64,
            "target_update": 10,  # Episodios entre actualizaciones de la red objetivo
            "memory_size": 10000,  # Tamaño del buffer de experiencia
            
            # Parámetros de módulos cognitivos
            "use_prediction": True,
            "prediction_horizon": 5,
            "use_curiosity": True,
            "curiosity_weight": 0.1,
            "use_meta_learning": True,
            
            # Parámetros de optimización
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            
            # Parámetros de hardware
            "use_gpu": True,
            "precision": "float32",
        }
        
        # Actualizar con configuraciones personalizadas
        self.config = self.defaults.copy()
        if config:
            self.config.update(config)
    
    def copy(self) -> 'NeurEvoConfig':
        """
        Crea una copia de la configuración.
        
        Returns:
            Nueva instancia de NeurEvoConfig con los mismos valores
        """
        return NeurEvoConfig(self.config.copy())
    
    def __getattr__(self, name):
        """
        Permite acceder a los parámetros de configuración como atributos.
        
        Args:
            name: Nombre del parámetro de configuración
            
        Returns:
            Valor del parámetro de configuración
            
        Raises:
            AttributeError: Si el parámetro no existe
        """
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'{self.__class__.__name__}' no tiene el atributo '{name}'")
    
    def __setattr__(self, name, value):
        """
        Permite establecer parámetros de configuración como atributos.
        
        Args:
            name: Nombre del parámetro de configuración
            value: Valor a establecer
        """
        if name in ["defaults", "config"]:
            super().__setattr__(name, value)
        else:
            self.config[name] = value
    
    def get(self, name, default=None):
        """
        Obtiene un parámetro de configuración con un valor por defecto opcional.
        
        Args:
            name: Nombre del parámetro de configuración
            default: Valor por defecto si el parámetro no existe
            
        Returns:
            Valor del parámetro de configuración o el valor por defecto
        """
        return self.config.get(name, default)
    
    def update(self, config: Dict[str, Any]):
        """
        Actualiza múltiples parámetros de configuración.
        
        Args:
            config: Diccionario con parámetros de configuración a actualizar
        """
        self.config.update(config)
    
    def __str__(self):
        """
        Representación en cadena de la configuración.
        
        Returns:
            Cadena con los parámetros de configuración
        """
        return f"NeurEvoConfig({self.config})" 