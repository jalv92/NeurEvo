"""
Clase base para todos los módulos del framework NeurEvo.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union, List

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
            "device": str(self.device)
        } 