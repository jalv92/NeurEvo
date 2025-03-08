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
        
        # Construir capas ocultas
        self.hidden_layers = hidden_layers
        layer_sizes = [flattened_size] + hidden_layers
        
        self.layers = nn.ModuleList()
        self.use_dynamic_layers = use_dynamic_layers
        
        # Para cada capa oculta
        for i in range(len(layer_sizes) - 1):
            if use_dynamic_layers:
                # Usar capas dinámicas
                layer = DynamicLayer(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i+1],
                    bias=True,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate,
                    device=device
                )
                
                # Registrar listener para cambios de dimensión
                if i > 0:  # No es necesario para la primera capa
                    prev_layer = self.layers[-1]
                    if isinstance(prev_layer, DynamicLayer):
                        # Cuando la capa anterior cambia, actualizar dimensiones de entrada de esta capa
                        prev_layer.register_dimension_listener(
                            lambda in_f, out_f, layer=layer: layer.update_input_size(out_f)
                        )
                    
                self.layers.append(layer)
            else:
                # Usar capas estándar
                self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                if use_batch_norm:
                    self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif activation == 'leaky_relu':
                    self.layers.append(nn.LeakyReLU())
                
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
    
    def _get_conv_output_size(self) -> int:
        """
        Calcula el tamaño de salida de las capas convolucionales.
        
        Returns:
            Tamaño del tensor aplanado después de las convoluciones
        """
        # Crear un tensor de entrada dummy
        dummy_input = torch.zeros((1,) + self.input_shape)
        
        # Pasarlo por las capas convolucionales
        x = dummy_input
        for layer in self.conv_layers:
            x = layer(x)
        
        # Devolver el tamaño total
        return int(np.prod(x.shape[1:]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.
        
        Args:
            x: Tensor de entrada [batch_size, *input_shape]
            
        Returns:
            Tensor de características [batch_size, last_hidden_size]
        """
        batch_size = x.shape[0]
        
        # Preprocesar entrada si es necesario
        if self.is_conv:
            # Aplicar capas convolucionales
            if len(self.input_shape) == 3:  # [C, H, W]
                for layer in self.conv_layers:
                    x = layer(x)
            
            # Aplanar la salida
            x = x.view(batch_size, -1)
        
        # Aplicar capas ocultas
        if self.use_dynamic_layers:
            for layer in self.layers:
                x = layer(x)
        else:
            for layer in self.layers:
                x = layer(x)
        
        return x
    
    def adapt_to_input_shape(self) -> None:
        """
        Adapta el módulo a cambios en la forma de entrada.
        """
        # Recalcular flattened_size si es necesario
        if self.is_conv:
            if len(self.input_shape) == 3:  # [C, H, W]
                # Es posible que necesitemos reconstruir las capas convolucionales
                in_channels = self.input_shape[0]
                
                # Recrear las capas convolucionales
                self.conv_layers = nn.ModuleList()
                self.conv_layers.append(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.Conv2d(32, 64, kernel_size=4, stride=2))
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1))
                self.conv_layers.append(nn.ReLU())
                
                # Recalcular tamaño de salida
                flattened_size = self._get_conv_output_size()
            else:
                # Simplemente recalcular el tamaño aplanado
                flattened_size = np.prod(self.input_shape)
        else:
            # Para entradas unidimensionales
            flattened_size = self.input_shape[0]
        
        # Actualizar la primera capa
        if self.use_dynamic_layers and len(self.layers) > 0:
            first_layer = self.layers[0]
            if isinstance(first_layer, DynamicLayer) and first_layer.in_features != flattened_size:
                first_layer.update_input_size(flattened_size)
    
    def summary(self) -> Dict[str, Any]:
        """
        Genera un resumen del módulo con sus principales características.
        
        Returns:
            Diccionario con información del módulo
        """
        base_summary = super().summary()
        
        additional_info = {
            "is_conv": self.is_conv,
            "hidden_layers": self.hidden_layers,
            "use_dynamic_layers": self.use_dynamic_layers,
            "num_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        # Añadir información de capas dinámicas si se usan
        if self.use_dynamic_layers:
            dynamic_info = {}
            for i, layer in enumerate(self.layers):
                if isinstance(layer, DynamicLayer):
                    dynamic_info[f"layer_{i}"] = {
                        "in_features": layer.in_features,
                        "out_features": layer.out_features,
                        "active_connections": layer.connectivity_mask.sum().item()
                    }
            additional_info["dynamic_layers"] = dynamic_info
        
        return {**base_summary, **additional_info}

    def adapt(self):
        """
        Adapta las capas dinámicas del módulo.
        """
        for layer in self.layers:
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