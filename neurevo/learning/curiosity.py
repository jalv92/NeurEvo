"""
Módulo de curiosidad intrínseca para el framework NeurEvo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any, Union

from neurevo.modules.base_module import BaseModule
from neurevo.modules.dynamic_layer import DynamicLayer

class CuriosityModule(BaseModule):
    """
    Módulo de curiosidad intrínseca que genera recompensas adicionales
    basadas en la novedad y la sorpresa para fomentar la exploración.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 action_size: int,
                 hidden_size: int = 128,
                 weight: float = 0.1,
                 curiosity_type: str = 'icm',  # 'icm', 'rnd', 'disagreement'
                 use_dynamic_layers: bool = True,
                 device: torch.device = None):
        """
        Inicializa el módulo de curiosidad.
        
        Args:
            input_shape: Forma de la entrada (características)
            action_size: Tamaño del espacio de acciones
            hidden_size: Tamaño de la capa oculta
            weight: Peso para la recompensa intrínseca
            curiosity_type: Tipo de curiosidad ('icm', 'rnd', 'disagreement')
            use_dynamic_layers: Si se deben usar capas dinámicas
            device: Dispositivo para cálculos (CPU/GPU)
        """
        super().__init__(input_shape, device)
        
        # Configuración
        self.input_size = input_shape[0] if len(input_shape) == 1 else input_shape
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.weight = weight
        self.curiosity_type = curiosity_type.lower()
        
        # Modelo de curiosidad basado en el tipo seleccionado
        if self.curiosity_type == 'icm':
            # Modelo de Curiosidad Intrínseca (ICM)
            # Predice el siguiente estado basado en el estado actual y la acción
            
            # Codificador de características
            if use_dynamic_layers:
                self.feature_encoder = DynamicLayer(
                    in_features=self.input_size,
                    out_features=hidden_size,
                    activation='relu',
                    device=self.device
                )
            else:
                self.feature_encoder = nn.Sequential(
                    nn.Linear(self.input_size, hidden_size),
                    nn.ReLU()
                )
            
            # Modelo directo (predice el siguiente estado)
            if use_dynamic_layers:
                self.forward_model = nn.Sequential(
                    DynamicLayer(
                        in_features=hidden_size + action_size,
                        out_features=hidden_size,
                        activation='relu',
                        device=self.device
                    ),
                    DynamicLayer(
                        in_features=hidden_size,
                        out_features=hidden_size,
                        activation='linear',
                        device=self.device
                    )
                )
            else:
                self.forward_model = nn.Sequential(
                    nn.Linear(hidden_size + action_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
            
            # Modelo inverso (predice la acción tomada)
            if use_dynamic_layers:
                self.inverse_model = nn.Sequential(
                    DynamicLayer(
                        in_features=hidden_size * 2,
                        out_features=hidden_size,
                        activation='relu',
                        device=self.device
                    ),
                    DynamicLayer(
                        in_features=hidden_size,
                        out_features=action_size,
                        activation='linear',
                        device=self.device
                    )
                )
            else:
                self.inverse_model = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, action_size)
                )
        
        elif self.curiosity_type == 'rnd':
            # Random Network Distillation (RND)
            # Usa una red aleatoria fija y una red entrenada para medir la novedad
            
            # Red aleatoria (objetivo)
            self.random_network = nn.Sequential(
                nn.Linear(self.input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            
            # Fijar pesos de la red aleatoria
            for param in self.random_network.parameters():
                param.requires_grad = False
            
            # Red de predicción (entrenada para imitar la red aleatoria)
            if use_dynamic_layers:
                self.predictor_network = nn.Sequential(
                    DynamicLayer(
                        in_features=self.input_size,
                        out_features=hidden_size,
                        activation='relu',
                        device=self.device
                    ),
                    DynamicLayer(
                        in_features=hidden_size,
                        out_features=hidden_size,
                        activation='relu',
                        device=self.device
                    ),
                    DynamicLayer(
                        in_features=hidden_size,
                        out_features=hidden_size,
                        activation='linear',
                        device=self.device
                    )
                )
            else:
                self.predictor_network = nn.Sequential(
                    nn.Linear(self.input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
        
        elif self.curiosity_type == 'disagreement':
            # Ensemble Disagreement
            # Usa un conjunto de redes para medir la incertidumbre
            
            # Número de modelos en el conjunto
            self.ensemble_size = 5
            
            # Conjunto de modelos
            self.ensemble = nn.ModuleList()
            for _ in range(self.ensemble_size):
                if use_dynamic_layers:
                    model = nn.Sequential(
                        DynamicLayer(
                            in_features=self.input_size + action_size,
                            out_features=hidden_size,
                            activation='relu',
                            device=self.device
                        ),
                        DynamicLayer(
                            in_features=hidden_size,
                            out_features=hidden_size,
                            activation='relu',
                            device=self.device
                        ),
                        DynamicLayer(
                            in_features=hidden_size,
                            out_features=self.input_size,
                            activation='linear',
                            device=self.device
                        )
                    )
                else:
                    model = nn.Sequential(
                        nn.Linear(self.input_size + action_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.input_size)
                    )
                self.ensemble.append(model)
        
        else:
            raise ValueError(f"Tipo de curiosidad no reconocido: {curiosity_type}")
        
        # Normalización de recompensas
        self.reward_normalizer = RunningStats()
        
        # Mover al dispositivo correcto
        self.to(self.device)
    
    def compute_reward(self, state: torch.Tensor, action: torch.Tensor, 
                      next_state: torch.Tensor) -> torch.Tensor:
        """
        Calcula la recompensa intrínseca basada en la curiosidad.
        
        Args:
            state: Estado actual [batch_size, input_size]
            action: Acción tomada [batch_size]
            next_state: Estado siguiente [batch_size, input_size]
            
        Returns:
            Tensor con la recompensa intrínseca [batch_size]
        """
        batch_size = state.size(0)
        
        # Convertir acción a one-hot si es discreta
        if action.dim() == 1:
            action_one_hot = torch.zeros(batch_size, self.action_size, device=self.device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
        else:
            action_one_hot = action
        
        # Calcular recompensa según el tipo de curiosidad
        if self.curiosity_type == 'icm':
            # Codificar estados
            state_features = self.feature_encoder(state)
            next_state_features = self.feature_encoder(next_state)
            
            # Predecir siguiente estado
            state_action = torch.cat([state_features, action_one_hot], dim=1)
            predicted_next_features = self.forward_model(state_action)
            
            # Error de predicción como recompensa
            prediction_error = F.mse_loss(predicted_next_features, next_state_features.detach(), 
                                         reduction='none').sum(dim=1)
            
            # Normalizar recompensa
            intrinsic_reward = self.normalize_reward(prediction_error)
        
        elif self.curiosity_type == 'rnd':
            # Calcular características aleatorias y predichas
            with torch.no_grad():
                random_features = self.random_network(next_state)
            predicted_features = self.predictor_network(next_state)
            
            # Error de predicción como recompensa
            prediction_error = F.mse_loss(predicted_features, random_features.detach(), 
                                         reduction='none').sum(dim=1)
            
            # Normalizar recompensa
            intrinsic_reward = self.normalize_reward(prediction_error)
        
        elif self.curiosity_type == 'disagreement':
            # Concatenar estado y acción
            state_action = torch.cat([state, action_one_hot], dim=1)
            
            # Obtener predicciones de todos los modelos
            predictions = []
            for model in self.ensemble:
                predictions.append(model(state_action))
            
            # Calcular desacuerdo entre modelos
            predictions = torch.stack(predictions, dim=0)  # [ensemble_size, batch_size, input_size]
            mean_prediction = predictions.mean(dim=0)  # [batch_size, input_size]
            
            # Varianza de las predicciones como medida de desacuerdo
            disagreement = ((predictions - mean_prediction.unsqueeze(0)) ** 2).mean(dim=0).sum(dim=1)
            
            # Normalizar recompensa
            intrinsic_reward = self.normalize_reward(disagreement)
        
        # Aplicar peso de curiosidad
        intrinsic_reward = intrinsic_reward * self.weight
        
        return intrinsic_reward
    
    def compute_loss(self, state: torch.Tensor, action: torch.Tensor, 
                    next_state: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida para entrenar el módulo de curiosidad.
        
        Args:
            state: Estado actual [batch_size, input_size]
            action: Acción tomada [batch_size]
            next_state: Estado siguiente [batch_size, input_size]
            
        Returns:
            Tensor con el valor de la pérdida
        """
        batch_size = state.size(0)
        
        # Convertir acción a one-hot si es discreta
        if action.dim() == 1:
            action_one_hot = torch.zeros(batch_size, self.action_size, device=self.device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
        else:
            action_one_hot = action
        
        # Calcular pérdida según el tipo de curiosidad
        if self.curiosity_type == 'icm':
            # Codificar estados
            state_features = self.feature_encoder(state)
            next_state_features = self.feature_encoder(next_state)
            
            # Modelo directo: predecir siguiente estado
            state_action = torch.cat([state_features, action_one_hot], dim=1)
            predicted_next_features = self.forward_model(state_action)
            forward_loss = F.mse_loss(predicted_next_features, next_state_features.detach())
            
            # Modelo inverso: predecir acción
            state_next_state = torch.cat([state_features, next_state_features], dim=1)
            predicted_action = self.inverse_model(state_next_state)
            inverse_loss = F.cross_entropy(predicted_action, action.squeeze(-1))
            
            # Pérdida total (combinación ponderada)
            total_loss = forward_loss + 0.5 * inverse_loss
        
        elif self.curiosity_type == 'rnd':
            # Calcular características aleatorias y predichas
            with torch.no_grad():
                random_features = self.random_network(next_state)
            predicted_features = self.predictor_network(next_state)
            
            # Pérdida de predicción
            total_loss = F.mse_loss(predicted_features, random_features.detach())
        
        elif self.curiosity_type == 'disagreement':
            # Concatenar estado y acción
            state_action = torch.cat([state, action_one_hot], dim=1)
            
            # Calcular pérdida para cada modelo en el conjunto
            ensemble_losses = []
            for model in self.ensemble:
                predicted_next_state = model(state_action)
                loss = F.mse_loss(predicted_next_state, next_state)
                ensemble_losses.append(loss)
            
            # Pérdida total (promedio de las pérdidas del conjunto)
            total_loss = torch.stack(ensemble_losses).mean()
        
        return total_loss
    
    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """
        Normaliza la recompensa intrínseca para estabilidad.
        
        Args:
            reward: Recompensa a normalizar
            
        Returns:
            Recompensa normalizada
        """
        # Actualizar estadísticas
        if reward.dim() > 0:
            for r in reward.detach().cpu().numpy():
                self.reward_normalizer.update(r)
        else:
            self.reward_normalizer.update(reward.item())
        
        # Normalizar usando media y desviación estándar
        mean = self.reward_normalizer.mean
        std = max(1e-8, self.reward_normalizer.std)
        
        normalized_reward = (reward - mean) / std
        
        # Recortar valores extremos
        normalized_reward = torch.clamp(normalized_reward, -10.0, 10.0)
        
        return normalized_reward


class RunningStats:
    """
    Clase auxiliar para mantener estadísticas en ejecución.
    """
    
    def __init__(self):
        self.n = 0
        self.old_mean = 0.0
        self.new_mean = 0.0
        self.old_var = 0.0
        self.new_var = 0.0
    
    def update(self, x):
        self.n += 1
        
        if self.n == 1:
            self.old_mean = self.new_mean = x
            self.old_var = 0.0
        else:
            self.new_mean = self.old_mean + (x - self.old_mean) / self.n
            self.new_var = self.old_var + (x - self.old_mean) * (x - self.new_mean)
            
            self.old_mean = self.new_mean
            self.old_var = self.new_var
    
    @property
    def mean(self):
        return self.new_mean if self.n > 0 else 0.0
    
    @property
    def var(self):
        return self.new_var / self.n if self.n > 1 else 0.0
    
    @property
    def std(self):
        return np.sqrt(self.var) 