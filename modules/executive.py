"""
Módulo ejecutivo para la toma de decisiones en el framework NeurEvo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any, Union

from neurevo.modules.base_module import BaseModule
from neurevo.modules.dynamic_layer import DynamicLayer

class ExecutiveModule(BaseModule):
    """
    Módulo ejecutivo responsable de la toma de decisiones basada en
    las características extraídas por el módulo de percepción.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 action_size: int,
                 hidden_layers: List[int] = [128, 64],
                 dueling_network: bool = True,
                 use_dynamic_layers: bool = True,
                 advantage_weight: float = 1.0,
                 device: torch.device = None):
        """
        Inicializa el módulo ejecutivo.
        
        Args:
            input_shape: Forma de la entrada (características)
            action_size: Número de acciones posibles
            hidden_layers: Lista con el tamaño de las capas ocultas
            dueling_network: Si se debe usar arquitectura dueling para Q-learning
            use_dynamic_layers: Si se deben usar capas dinámicas
            advantage_weight: Peso para la ventaja en arquitectura dueling
            device: Dispositivo para cálculos (CPU/GPU)
        """
        super().__init__(input_shape, device)
        
        # Configuración
        self.input_size = input_shape[0] if len(input_shape) == 1 else input_shape
        self.action_size = action_size
        self.dueling_network = dueling_network
        self.advantage_weight = advantage_weight
        
        # Capas compartidas
        self.shared_layers = nn.ModuleList()
        layer_sizes = [self.input_size] + hidden_layers
        
        for i in range(len(layer_sizes) - 1):
            if use_dynamic_layers:
                layer = DynamicLayer(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i+1],
                    activation='relu',
                    use_batch_norm=True,
                    device=self.device
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                    nn.BatchNorm1d(layer_sizes[i+1]),
                    nn.ReLU()
                )
            self.shared_layers.append(layer)
        
        # Arquitectura dueling para Q-learning
        if dueling_network:
            # Rama de valor (estima el valor del estado)
            if use_dynamic_layers:
                self.value_stream = DynamicLayer(
                    in_features=hidden_layers[-1],
                    out_features=1,
                    activation='linear',
                    device=self.device
                )
            else:
                self.value_stream = nn.Linear(hidden_layers[-1], 1)
            
            # Rama de ventaja (estima la ventaja de cada acción)
            if use_dynamic_layers:
                self.advantage_stream = DynamicLayer(
                    in_features=hidden_layers[-1],
                    out_features=action_size,
                    activation='linear',
                    device=self.device
                )
            else:
                self.advantage_stream = nn.Linear(hidden_layers[-1], action_size)
        else:
            # Red Q estándar
            if use_dynamic_layers:
                self.q_layer = DynamicLayer(
                    in_features=hidden_layers[-1],
                    out_features=action_size,
                    activation='linear',
                    device=self.device
                )
            else:
                self.q_layer = nn.Linear(hidden_layers[-1], action_size)
        
        # Historial de decisiones para análisis
        self.action_distribution = np.zeros(action_size)
        self.q_value_history = []
        
        # Mover al dispositivo correcto
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante del módulo.
        
        Args:
            x: Tensor de entrada [batch_size, input_size]
            
        Returns:
            Valores Q para cada acción [batch_size, action_size]
        """
        # Procesar con capas compartidas
        for layer in self.shared_layers:
            x = layer(x)
        
        # Arquitectura dueling
        if self.dueling_network:
            # Calcular valor del estado
            value = self.value_stream(x)
            
            # Calcular ventajas de las acciones
            advantages = self.advantage_stream(x)
            
            # Combinar valor y ventajas para obtener valores Q
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            q_values = value + self.advantage_weight * (
                advantages - advantages.mean(dim=1, keepdim=True)
            )
        else:
            # Red Q estándar
            q_values = self.q_layer(x)
        
        # Registrar valores Q para análisis (solo en modo evaluación)
        if not self.training and x.size(0) == 1:
            self.q_value_history.append(q_values.detach().cpu().numpy())
        
        return q_values
    
    def select_action(self, state_features: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Selecciona una acción basada en la política epsilon-greedy.
        
        Args:
            state_features: Características del estado
            epsilon: Probabilidad de exploración
            
        Returns:
            Índice de la acción seleccionada
        """
        # Exploración aleatoria
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            # Explotación (mejor acción según valores Q)
            with torch.no_grad():
                q_values = self.forward(state_features)
                action = q_values.argmax(dim=1).item()
        
        # Actualizar distribución de acciones
        self.action_distribution[action] += 1
        
        return action
    
    def compute_loss(self, q_values: torch.Tensor, target_q_values: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida para el aprendizaje por refuerzo.
        
        Args:
            q_values: Valores Q predichos
            target_q_values: Valores Q objetivo
            
        Returns:
            Tensor con el valor de la pérdida
        """
        # Pérdida de error cuadrático medio
        loss = F.mse_loss(q_values, target_q_values)
        
        return loss
    
    def get_action_distribution(self) -> np.ndarray:
        """
        Devuelve la distribución de acciones tomadas.
        
        Returns:
            Array con la distribución normalizada
        """
        total = self.action_distribution.sum()
        if total > 0:
            return self.action_distribution / total
        return self.action_distribution
    
    def reset_stats(self):
        """
        Reinicia las estadísticas del módulo.
        """
        self.action_distribution = np.zeros(self.action_size)
        self.q_value_history = [] 