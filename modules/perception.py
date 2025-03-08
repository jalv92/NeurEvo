"""
Módulo de percepción para el framework NeurEvo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any, Union

from neurevo.modules.base_module import BaseModule
from neurevo.modules.dynamic_layer import DynamicLayer

class PerceptionModule(BaseModule):
    """
    Módulo de percepción que procesa las observaciones del entorno
    y extrae características relevantes para la toma de decisiones.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 hidden_layers: List[int] = [128, 128],
                 use_batch_norm: bool = True,
                 dropout_rate: float = 0.0,
                 activation: str = 'relu',
                 use_dynamic_layers: bool = True,
                 device: torch.device = None):
        """
        Inicializa el módulo de percepción.
        
        Args:
            input_shape: Forma de la entrada (observación)
            hidden_layers: Lista con el tamaño de las capas ocultas
            use_batch_norm: Si se debe usar normalización por lotes
            dropout_rate: Tasa de dropout
            activation: Función de activación
            use_dynamic_layers: Si se deben usar capas dinámicas
            device: Dispositivo para cálculos (CPU/GPU)
        """
        super().__init__(input_shape, device)
        
        # Aplanar la entrada si es multidimensional
        self.is_conv = len(input_shape) > 1
        
        # Construir red convolucional para entradas multidimensionales
        if self.is_conv:
            self.conv_layers = nn.ModuleList()
            
            # Determinar si la entrada es una imagen
            if len(input_shape) == 3:  # [C, H, W]
                in_channels = input_shape[0]
                
                # Capas convolucionales para procesamiento de imágenes
                self.conv_layers.append(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.Conv2d(32, 64, kernel_size=4, stride=2))
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1))
                self.conv_layers.append(nn.ReLU())
                
                # Calcular tamaño de salida de las convoluciones
                conv_output = self._get_conv_output_size()
                flattened_size = conv_output
            else:
                # Para otras entradas multidimensionales, aplanar directamente
                flattened_size = np.prod(input_shape)
        else:
            # Para entradas unidimensionales
            flattened_size = input_shape[0]
        
        # Construir capas completamente conectadas
        self.fc_layers = nn.ModuleList()
        layer_sizes = [flattened_size] + hidden_layers
        
        for i in range(len(layer_sizes) - 1):
            if use_dynamic_layers:
                # Usar capas dinámicas
                layer = DynamicLayer(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i+1],
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate,
                    device=self.device
                )
            else:
                # Usar capas estándar
                layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
                self.fc_layers.append(layer)
                
                # Añadir normalización por lotes si está habilitada
                if use_batch_norm:
                    self.fc_layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                
                # Añadir activación
                if activation.lower() == 'relu':
                    self.fc_layers.append(nn.ReLU())
                elif activation.lower() == 'tanh':
                    self.fc_layers.append(nn.Tanh())
                elif activation.lower() == 'sigmoid':
                    self.fc_layers.append(nn.Sigmoid())
                elif activation.lower() == 'leaky_relu':
                    self.fc_layers.append(nn.LeakyReLU())
                
                # Añadir dropout si está habilitado
                if dropout_rate > 0:
                    self.fc_layers.append(nn.Dropout(dropout_rate))
            
            # Si es una capa dinámica, añadirla directamente (ya incluye activación, etc.)
            if use_dynamic_layers:
                self.fc_layers.append(layer)
        
        # Capa de salida
        self.output_layer = nn.Linear(hidden_layers[-1], hidden_layers[-1])
        
        # Mover al dispositivo correcto
        self.to(self.device)
    
    def _get_conv_output_size(self) -> int:
        """
        Calcula el tamaño de salida de las capas convolucionales.
        
        Returns:
            Tamaño de la salida aplanada
        """
        # Crear tensor de entrada de prueba
        x = torch.zeros(1, *self.input_shape).to(self.device)
        
        # Pasar por capas convolucionales
        for layer in self.conv_layers:
            x = layer(x)
        
        # Devolver tamaño aplanado
        return int(np.prod(x.shape[1:]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante del módulo.
        
        Args:
            x: Tensor de entrada [batch_size, *input_shape]
            
        Returns:
            Tensor de características extraídas [batch_size, output_size]
        """
        # Procesar con capas convolucionales si es necesario
        if self.is_conv:
            for layer in self.conv_layers:
                x = layer(x)
            
            # Aplanar para capas completamente conectadas
            x = x.view(x.size(0), -1)
        
        # Procesar con capas completamente conectadas
        for layer in self.fc_layers:
            x = layer(x)
        
        # Capa de salida
        x = self.output_layer(x)
        
        return x
    
    def adapt(self):
        """
        Adapta las capas dinámicas del módulo.
        """
        for layer in self.fc_layers:
            if isinstance(layer, DynamicLayer):
                layer.adapt_connectivity()
    
    def compute_loss(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """
        Calcula una pérdida de reconstrucción para el módulo de percepción.
        
        Args:
            x: Tensor de entrada
            target: Tensor objetivo (opcional)
            
        Returns:
            Tensor con el valor de la pérdida
        """
        # Si no hay objetivo, usar la entrada como objetivo (autoencoder)
        if target is None:
            target = x
        
        # Calcular características
        features = self.forward(x)
        
        # Reconstruir entrada (simplificado)
        if not hasattr(self, 'reconstruction_layer'):
            # Crear capa de reconstrucción bajo demanda
            input_size = np.prod(self.input_shape)
            self.reconstruction_layer = nn.Linear(features.shape[1], input_size).to(self.device)
        
        # Reconstruir
        reconstructed = self.reconstruction_layer(features)
        
        # Aplanar objetivo si es necesario
        if target.dim() > 2:
            target = target.view(target.size(0), -1)
        
        # Calcular pérdida de reconstrucción
        loss = F.mse_loss(reconstructed, target)
        
        return loss 