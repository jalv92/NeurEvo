"""
Implementación de capa neuronal dinámica para el framework NeurEvo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional, Union, Callable, Any

class DynamicLayer(nn.Module):
    """
    Capa neuronal dinámica que puede adaptar su conectividad y parámetros
    durante el entrenamiento basándose en la actividad neuronal.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 activation: Union[str, Callable] = 'relu',
                 use_batch_norm: bool = False,
                 dropout_rate: float = 0.0,
                 lateral_connections: bool = False,
                 sparsity: float = 0.0,
                 device: torch.device = None):
        """
        Inicializa una capa dinámica.
        
        Args:
            in_features: Número de características de entrada
            out_features: Número de características de salida
            bias: Si se debe incluir un término de sesgo
            activation: Función de activación ('relu', 'tanh', 'sigmoid', etc.)
            use_batch_norm: Si se debe usar normalización por lotes
            dropout_rate: Tasa de dropout (0.0 para desactivar)
            lateral_connections: Si se deben incluir conexiones laterales
            sparsity: Nivel de dispersión de la capa (0.0 para densidad completa)
            device: Dispositivo para cálculos (CPU/GPU)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sistema de notificación de cambios dimensionales
        self.dimension_listeners = []
        
        # Inicializar pesos y sesgo
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.has_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Máscara de conectividad (para poda adaptativa)
        self.register_buffer('connectivity_mask', torch.ones_like(self.weight))
        
        # Normalización por lotes
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        
        # Dropout
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Conexiones laterales (recurrentes)
        self.lateral_connections = lateral_connections
        if lateral_connections:
            self.lateral_weight = nn.Parameter(torch.Tensor(out_features, out_features))
            self.register_buffer('lateral_mask', torch.ones_like(self.lateral_weight))
            nn.init.sparse_(self.lateral_weight, sparsity=0.9)  # Inicialización dispersa
        
        # Activación
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'tanh':
                self.activation = torch.tanh
            elif activation.lower() == 'sigmoid':
                self.activation = torch.sigmoid
            elif activation.lower() == 'leaky_relu':
                self.activation = F.leaky_relu
            else:
                raise ValueError(f"Activación no reconocida: {activation}")
        else:
            self.activation = activation
        
        # Nivel de dispersión
        self.sparsity = sparsity
        if sparsity > 0:
            # Aplicar máscara de dispersión inicial
            mask = torch.rand_like(self.weight) > sparsity
            self.connectivity_mask.data.copy_(mask.float())
        
        # Inicialización de parámetros
        self.reset_parameters()
        
        # Historial de activaciones para adaptación
        self.register_buffer('activation_history', torch.zeros(out_features))
        self.register_buffer('importance_score', torch.ones(out_features))
    
    def register_dimension_listener(self, listener: Callable[[int, int], Any]) -> None:
        """
        Registra una función callback que será llamada cuando cambien las dimensiones.
        
        Args:
            listener: Función que recibirá (in_features, out_features) cuando cambien
        """
        self.dimension_listeners.append(listener)
    
    def notify_dimension_change(self) -> None:
        """
        Notifica a todos los listeners registrados que ha habido un cambio en las dimensiones.
        """
        for listener in self.dimension_listeners:
            listener(self.in_features, self.out_features)
    
    def reset_parameters(self):
        """
        Inicializa los parámetros de la capa con valores adecuados.
        """
        # Inicialización de Kaiming para ReLU
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.has_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Inicializar conexiones laterales si existen
        if self.lateral_connections:
            # Inicialización cercana a cero para estabilidad
            nn.init.normal_(self.lateral_weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante de la capa.
        
        Args:
            x: Tensor de entrada [batch_size, in_features]
            
        Returns:
            Tensor de salida [batch_size, out_features]
        """
        # Aplicar máscara de conectividad
        masked_weight = self.weight * self.connectivity_mask
        
        # Propagación hacia adelante
        output = F.linear(x, masked_weight, self.bias)
        
        # Aplicar conexiones laterales si existen y hay estado previo
        if self.lateral_connections and hasattr(self, 'previous_output'):
            lateral_input = self.previous_output
            masked_lateral = self.lateral_weight * self.lateral_mask
            lateral_contribution = F.linear(lateral_input, masked_lateral)
            output = output + 0.1 * lateral_contribution  # Factor de escala para estabilidad
        
        # Guardar salida para conexiones laterales en la siguiente pasada
        if self.lateral_connections:
            self.previous_output = output.detach()
        
        # Aplicar normalización por lotes si está habilitada
        if self.use_batch_norm:
            output = self.batch_norm(output)
        
        # Aplicar activación
        output = self.activation(output)
        
        # Aplicar dropout si está habilitado
        if self.dropout_rate > 0:
            output = self.dropout(output)
        
        # Actualizar historial de activaciones (para adaptación)
        if self.training:
            with torch.no_grad():
                batch_activations = (output > 0).float().mean(dim=0)
                self.activation_history.mul_(0.9).add_(batch_activations * 0.1)
        
        return output
    
    def adapt_connectivity(self, threshold: float = 0.01):
        """
        Adapta la conectividad de la capa basándose en la importancia de las conexiones.
        
        Args:
            threshold: Umbral para podar conexiones
        """
        old_out_features = self.out_features
        
        with torch.no_grad():
            # Calcular importancia de cada conexión
            importance = torch.abs(self.weight)
            
            # Normalizar importancia
            importance = importance / importance.max()
            
            # Actualizar máscara de conectividad
            new_mask = importance > threshold
            
            # Asegurar que cada neurona tenga al menos una conexión
            for i in range(self.out_features):
                if not new_mask[i].any():
                    # Si todas las conexiones serían podadas, mantener la más importante
                    max_idx = torch.argmax(importance[i])
                    new_mask[i, max_idx] = True
            
            # Actualizar máscara
            self.connectivity_mask.data.copy_(new_mask.float())
            
            # Actualizar también conexiones laterales si existen
            if self.lateral_connections:
                lateral_importance = torch.abs(self.lateral_weight)
                lateral_importance = lateral_importance / lateral_importance.max()
                lateral_mask = lateral_importance > threshold
                
                # Asegurar conectividad mínima
                for i in range(self.out_features):
                    if not lateral_mask[i].any():
                        max_idx = torch.argmax(lateral_importance[i])
                        lateral_mask[i, max_idx] = True
                
                self.lateral_mask.data.copy_(lateral_mask.float())
                
        # Si las dimensiones han cambiado, notificar a los listeners
        if self.out_features != old_out_features:
            self.notify_dimension_change()
    
    def grow_connections(self, rate: float = 0.05):
        """
        Hace crecer nuevas conexiones en la capa.
        
        Args:
            rate: Tasa de crecimiento (proporción de nuevas conexiones)
        """
        with torch.no_grad():
            # Identificar conexiones inactivas
            inactive = self.connectivity_mask == 0
            
            # Seleccionar aleatoriamente algunas para reactivar
            candidates = torch.rand_like(self.weight) < rate
            to_grow = inactive & candidates
            
            # Actualizar máscara y reinicializar pesos
            self.connectivity_mask.data[to_grow] = 1.0
            
            # Inicializar nuevas conexiones con valores pequeños
            self.weight.data[to_grow] = torch.randn_like(self.weight.data[to_grow]) * 0.01
            
        # Notificar a los listeners del cambio
        self.notify_dimension_change()
    
    def update_input_size(self, new_in_features: int) -> None:
        """
        Actualiza el tamaño de entrada de la capa, redimensionando los pesos.
        
        Args:
            new_in_features: Nuevo número de características de entrada
        """
        if new_in_features == self.in_features:
            return
            
        with torch.no_grad():
            old_weights = self.weight.data
            self.in_features = new_in_features
            
            # Crear nuevo tensor de pesos con el tamaño actualizado
            new_weight = torch.Tensor(self.out_features, new_in_features).to(self.device)
            
            # Inicializar nuevos pesos
            nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
            
            # Copiar los pesos antiguos en las posiciones correspondientes
            min_features = min(old_weights.shape[1], new_in_features)
            new_weight[:, :min_features] = old_weights[:, :min_features]
            
            # Actualizar parámetro de peso
            self.weight = nn.Parameter(new_weight)
            
            # Actualizar máscara de conectividad
            self.register_buffer('connectivity_mask', torch.ones_like(self.weight))
            if self.sparsity > 0:
                mask = torch.rand_like(self.weight) > self.sparsity
                self.connectivity_mask.data.copy_(mask.float())
                
        # Notificar a los listeners del cambio
        self.notify_dimension_change()
    
    def update_output_size(self, new_out_features: int) -> None:
        """
        Actualiza el tamaño de salida de la capa, redimensionando los pesos.
        
        Args:
            new_out_features: Nuevo número de características de salida
        """
        if new_out_features == self.out_features:
            return
            
        with torch.no_grad():
            old_weights = self.weight.data
            old_out_features = self.out_features
            self.out_features = new_out_features
            
            # Crear nuevo tensor de pesos con el tamaño actualizado
            new_weight = torch.Tensor(new_out_features, self.in_features).to(self.device)
            
            # Inicializar nuevos pesos
            nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
            
            # Copiar los pesos antiguos en las posiciones correspondientes
            min_features = min(old_weights.shape[0], new_out_features)
            new_weight[:min_features, :] = old_weights[:min_features, :]
            
            # Actualizar parámetro de peso
            self.weight = nn.Parameter(new_weight)
            
            # Actualizar sesgo si existe
            if self.has_bias:
                old_bias = self.bias.data
                new_bias = torch.Tensor(new_out_features).to(self.device)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(new_bias, -bound, bound)
                min_bias = min(old_bias.shape[0], new_out_features)
                new_bias[:min_bias] = old_bias[:min_bias]
                self.bias = nn.Parameter(new_bias)
            
            # Actualizar máscara de conectividad
            self.register_buffer('connectivity_mask', torch.ones_like(self.weight))
            if self.sparsity > 0:
                mask = torch.rand_like(self.weight) > self.sparsity
                self.connectivity_mask.data.copy_(mask.float())
                
            # Actualizar historial de activaciones
            new_activation_history = torch.zeros(new_out_features).to(self.device)
            new_importance_score = torch.ones(new_out_features).to(self.device)
            min_size = min(self.activation_history.shape[0], new_out_features)
            new_activation_history[:min_size] = self.activation_history[:min_size]
            new_importance_score[:min_size] = self.importance_score[:min_size]
            self.register_buffer('activation_history', new_activation_history)
            self.register_buffer('importance_score', new_importance_score)
            
            # Actualizar conexiones laterales si existen
            if self.lateral_connections:
                old_lateral = self.lateral_weight.data
                new_lateral = torch.Tensor(new_out_features, new_out_features).to(self.device)
                nn.init.normal_(new_lateral, mean=0.0, std=0.01)
                min_lateral = min(old_lateral.shape[0], new_out_features)
                new_lateral[:min_lateral, :min_lateral] = old_lateral[:min_lateral, :min_lateral]
                self.lateral_weight = nn.Parameter(new_lateral)
                self.register_buffer('lateral_mask', torch.ones_like(self.lateral_weight))
            
            # Actualizar BatchNorm si se usa
            if self.use_batch_norm:
                self.batch_norm = nn.BatchNorm1d(new_out_features)
                
        # Notificar a los listeners del cambio
        self.notify_dimension_change()
    
    def extra_repr(self) -> str:
        """
        Representación de cadena con información adicional.
        
        Returns:
            Cadena con información de la capa
        """
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias}, ' \
               f'sparsity={self.sparsity:.2f}, active_connections={self.connectivity_mask.sum().item()}' 