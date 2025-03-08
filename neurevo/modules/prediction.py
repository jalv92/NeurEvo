"""
Módulo de predicción para el framework NeurEvo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Union

from neurevo.modules.base_module import BaseModule
from neurevo.modules.dynamic_layer import DynamicLayer

class PredictionModule(BaseModule):
    """
    Módulo de predicción que anticipa estados futuros basándose en
    las características extraídas por el módulo de percepción.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 hidden_size: int = 128,
                 horizon: int = 5,
                 use_dynamic_layers: bool = True,
                 device: torch.device = None):
        """
        Inicializa el módulo de predicción.
        
        Args:
            input_shape: Forma de la entrada (características)
            hidden_size: Tamaño de la capa oculta
            horizon: Horizonte de predicción (cuántos pasos hacia el futuro)
            use_dynamic_layers: Si se deben usar capas dinámicas
            device: Dispositivo para cálculos (CPU/GPU)
        """
        super().__init__(input_shape, device)
        
        # Configuración
        self.input_size = input_shape[0] if len(input_shape) == 1 else input_shape
        self.hidden_size = hidden_size
        self.horizon = horizon
        
        # Modelo de predicción (arquitectura tipo LSTM simplificada)
        if use_dynamic_layers:
            self.encoder = DynamicLayer(
                in_features=self.input_size,
                out_features=hidden_size,
                activation='tanh',
                device=self.device
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, hidden_size),
                nn.Tanh()
            )
        
        # Capa recurrente (GRU para predicciones temporales)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        
        # Decodificador para cada paso del horizonte
        self.decoders = nn.ModuleList()
        for _ in range(horizon):
            if use_dynamic_layers:
                decoder = DynamicLayer(
                    in_features=hidden_size,
                    out_features=self.input_size,
                    activation='tanh',
                    device=self.device
                )
            else:
                decoder = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, self.input_size)
                )
            self.decoders.append(decoder)
        
        # Estado oculto
        self.hidden_state = None
        
        # Historial de predicciones para análisis
        self.predictions = []
        self.prediction_errors = []
        
        # Mover al dispositivo correcto
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante del módulo.
        
        Args:
            x: Tensor de entrada [batch_size, input_size]
            
        Returns:
            Lista de predicciones futuras [batch_size, horizon, input_size]
        """
        batch_size = x.size(0)
        
        # Codificar entrada
        encoded = self.encoder(x)
        
        # Inicializar o actualizar estado oculto
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.hidden_size, device=self.device)
        
        # Actualizar estado oculto con entrada actual
        self.hidden_state = self.rnn(encoded, self.hidden_state)
        
        # Generar predicciones para cada paso del horizonte
        predictions = []
        current_state = self.hidden_state
        
        for i in range(self.horizon):
            # Decodificar estado para obtener predicción
            pred = self.decoders[i](current_state)
            predictions.append(pred)
            
            # Actualizar estado para siguiente predicción
            if i < self.horizon - 1:
                current_state = self.rnn(pred, current_state)
        
        # Apilar predicciones [batch_size, horizon, input_size]
        stacked_predictions = torch.stack(predictions, dim=1)
        
        # Guardar última predicción para análisis (solo en modo entrenamiento)
        if self.training and batch_size == 1:
            self.predictions.append(stacked_predictions[0, 0].detach().cpu())
        
        return stacked_predictions
    
    def compute_loss(self, current_features: torch.Tensor, next_features: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida de predicción comparando con el siguiente estado real.
        
        Args:
            current_features: Características del estado actual
            next_features: Características del siguiente estado real
            
        Returns:
            Tensor con el valor de la pérdida
        """
        # Obtener predicciones
        predictions = self.forward(current_features)
        
        # La primera predicción debe coincidir con el siguiente estado
        first_prediction = predictions[:, 0, :]
        
        # Calcular error de predicción
        prediction_loss = F.mse_loss(first_prediction, next_features)
        
        # Guardar error para análisis (solo en modo entrenamiento)
        if self.training and current_features.size(0) == 1:
            error = torch.norm(first_prediction - next_features).item()
            self.prediction_errors.append(error)
        
        return prediction_loss
    
    def reset(self):
        """
        Reinicia el estado interno del módulo.
        """
        self.hidden_state = None
        
    def get_intrinsic_reward(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """
        Calcula una recompensa intrínseca basada en el error de predicción.
        
        Args:
            prediction_error: Error de predicción
            
        Returns:
            Tensor con el valor de la recompensa intrínseca
        """
        # Normalizar error para obtener recompensa (error alto = recompensa alta)
        # Esto incentiva la exploración de áreas donde el modelo no predice bien
        if not hasattr(self, 'error_normalizer'):
            self.error_normalizer = 1.0
            self.error_history = []
        
        # Actualizar historial de errores
        self.error_history.append(prediction_error.item())
        if len(self.error_history) > 100:
            self.error_history.pop(0)
            self.error_normalizer = max(1.0, sum(self.error_history) / len(self.error_history) * 2)
        
        # Calcular recompensa normalizada
        normalized_error = prediction_error / self.error_normalizer
        
        # Aplicar función de saturación para evitar recompensas extremas
        reward = torch.tanh(normalized_error)
        
        return reward
    
    def get_prediction_accuracy(self) -> float:
        """
        Calcula la precisión de las predicciones recientes.
        
        Returns:
            Precisión de predicción como porcentaje
        """
        if not self.prediction_errors:
            return 0.0
        
        # Calcular error promedio normalizado
        avg_error = sum(self.prediction_errors[-100:]) / max(1, len(self.prediction_errors[-100:]))
        
        # Convertir a precisión (100% - error%)
        accuracy = max(0.0, 100.0 * (1.0 - min(1.0, avg_error)))
        
        return accuracy 