"""
Clase base para todos los módulos del framework NeurEvo.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union, List, Optional, Callable

class BaseModule(nn.Module, ABC):
    """
    Clase base abstracta para todos los módulos neuronales del framework.
    Proporciona funcionalidad común y define la interfaz que deben implementar.
    """
    
    def __init__(self, input_shape: Tuple[int, ...], device: torch.device = None):
        """
        Inicializa el módulo base.
        
        Args:
            input_shape: Forma de la entrada al módulo
            device: Dispositivo para cálculos (CPU/GPU)
        """
        super().__init__()
        self.input_shape = input_shape
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._output_shape = None
        self.connected_modules = {
            "input": [],  # Módulos que alimentan este módulo
            "output": []  # Módulos alimentados por este módulo
        }
    
    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Devuelve la forma de la salida del módulo.
        
        Returns:
            Tupla con las dimensiones de la salida
        """
        if self._output_shape is None:
            # Inferir forma de salida con una pasada hacia adelante
            with torch.no_grad():
                dummy_input = torch.zeros((1,) + self.input_shape).to(self.device)
                output = self.forward(dummy_input)
                self._output_shape = tuple(output.shape[1:])
        return self._output_shape
    
    def connect_to_input(self, module: 'BaseModule') -> None:
        """
        Conecta este módulo a un módulo de entrada.
        
        Args:
            module: Módulo que alimenta a este módulo
        """
        if module not in self.connected_modules["input"]:
            self.connected_modules["input"].append(module)
            # También registrar este módulo como salida del módulo de entrada
            module.connect_to_output(self)
    
    def connect_to_output(self, module: 'BaseModule') -> None:
        """
        Conecta este módulo a un módulo de salida.
        
        Args:
            module: Módulo alimentado por este módulo
        """
        if module not in self.connected_modules["output"]:
            self.connected_modules["output"].append(module)
            # No registramos este módulo como entrada del módulo de salida
            # para evitar recursión infinita, ya que connect_to_input lo hace
    
    def dimension_changed(self, in_features: int, out_features: int, source_module: Optional['BaseModule'] = None) -> None:
        """
        Maneja un cambio de dimensión en un módulo conectado.
        
        Args:
            in_features: Nuevas características de entrada del módulo que cambió
            out_features: Nuevas características de salida del módulo que cambió
            source_module: Módulo que cambió sus dimensiones
        """
        # Si el módulo origen es uno de nuestros módulos de entrada, actualizar nuestro input_shape
        if source_module in self.connected_modules["input"]:
            # El out_features del módulo de entrada es nuestro input_shape
            if len(self.input_shape) == 1:
                # Caso simple: la entrada es un vector
                self.update_input_shape((out_features,))
            else:
                # Caso más complejo: necesita implementación específica en subclases
                pass
        
        # Si el módulo origen es uno de nuestros módulos de salida, pueden necesitar
        # actualización basada en nuestro output_shape
        elif source_module in self.connected_modules["output"]:
            # No hacemos nada por defecto, pero las subclases pueden implementar
            # comportamiento específico si es necesario
            pass
    
    def update_input_shape(self, new_input_shape: Tuple[int, ...]) -> None:
        """
        Actualiza la forma de entrada del módulo y propaga los cambios.
        
        Args:
            new_input_shape: Nueva forma de entrada
        """
        if new_input_shape == self.input_shape:
            return
            
        self.input_shape = new_input_shape
        self._output_shape = None  # Invalidar cache de output_shape
        
        # Implementar lógica específica de adaptación en subclases
        self.adapt_to_input_shape()
        
        # Recalcular output_shape
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + self.input_shape).to(self.device)
            try:
                output = self.forward(dummy_input)
                self._output_shape = tuple(output.shape[1:])
                
                # Propagar cambio a módulos de salida
                for module in self.connected_modules["output"]:
                    if hasattr(module, 'dimension_changed'):
                        module.dimension_changed(self.input_shape[0] if len(self.input_shape) == 1 else None, 
                                              self._output_shape[0] if len(self._output_shape) == 1 else None, 
                                              self)
            except Exception as e:
                print(f"Error al recalcular output_shape en {self.__class__.__name__}: {e}")
    
    def adapt_to_input_shape(self) -> None:
        """
        Adapta este módulo a un cambio en su forma de entrada.
        Esta implementación no hace nada, las subclases deben sobrescribirla.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante del módulo.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        pass
    
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Calcula la pérdida específica del módulo.
        
        Returns:
            Tensor con el valor de la pérdida
        """
        return torch.tensor(0.0, device=self.device)
    
    def reset(self) -> None:
        """
        Reinicia el estado interno del módulo si es necesario.
        """
        pass
    
    def get_intrinsic_reward(self, *args, **kwargs) -> torch.Tensor:
        """
        Calcula una recompensa intrínseca basada en el módulo.
        
        Returns:
            Tensor con el valor de la recompensa intrínseca
        """
        return torch.tensor(0.0, device=self.device)
    
    def summary(self) -> Dict[str, Any]:
        """
        Devuelve un resumen del módulo con información relevante.
        
        Returns:
            Diccionario con información del módulo
        """
        return {
            "name": self.__class__.__name__,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "parameters": sum(p.numel() for p in self.parameters()),
            "device": str(self.device),
            "connected_modules": {
                "input": [m.__class__.__name__ for m in self.connected_modules["input"]],
                "output": [m.__class__.__name__ for m in self.connected_modules["output"]]
            }
        } 