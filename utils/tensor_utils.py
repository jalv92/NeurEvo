"""
Utilidades para manejo de tensores en el framework NeurEvo.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional

def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[Optional[int], ...], 
                         tensor_name: str = "tensor") -> bool:
    """
    Verifica que un tensor tenga la forma esperada o compatible.
    
    Args:
        tensor: El tensor a verificar
        expected_shape: Forma esperada o mínima (los None son comodines)
        tensor_name: Nombre para mensajes de error
        
    Returns:
        True si el tensor es válido
        
    Raises:
        ValueError: Si el tensor no tiene la forma esperada
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"{tensor_name} debe ser un tensor PyTorch, no {type(tensor)}")
    
    if len(tensor.shape) != len(expected_shape):
        raise ValueError(f"{tensor_name} tiene {len(tensor.shape)} dimensiones, se esperaban {len(expected_shape)}")
    
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(f"Dimensión {i} de {tensor_name} es {actual}, se esperaba {expected}")
    
    return True

def safe_cat(tensors: List[torch.Tensor], dim: int = 0, tensor_name: str = "tensores") -> torch.Tensor:
    """
    Realiza una concatenación segura de tensores, con verificaciones y manejo de errores.
    
    Args:
        tensors: Lista de tensores a concatenar
        dim: Dimensión en la que concatenar
        tensor_name: Nombre para mensajes de error
        
    Returns:
        Tensor concatenado
    """
    if not tensors:
        raise ValueError(f"Lista de {tensor_name} vacía")
    
    try:
        # Verificar que todos los tensores tienen la misma forma excepto en la dimensión de concatenación
        shapes = [t.shape for t in tensors]
        for i, shape in enumerate(shapes[1:], 1):
            for j, (s1, s2) in enumerate(zip(shapes[0], shape)):
                if j != dim and s1 != s2:
                    raise ValueError(f"Formas incompatibles: {shapes[0]} y {shape} en dimensión {j}")
        
        # Intentar concatenar
        return torch.cat(tensors, dim=dim)
    
    except Exception as e:
        print(f"Error al concatenar {tensor_name}: {e}")
        
        # Intentar ajustar tensores si es posible
        try:
            # Encontrar la forma máxima en cada dimensión
            max_shape = list(tensors[0].shape)
            for t in tensors[1:]:
                for i, s in enumerate(t.shape):
                    if i != dim and s > max_shape[i]:
                        max_shape[i] = s
            
            # Ajustar tensores a la forma máxima
            adjusted_tensors = []
            for t in tensors:
                if list(t.shape) == max_shape:
                    adjusted_tensors.append(t)
                else:
                    # Crear tensor de ceros con la forma máxima
                    adjusted = torch.zeros(max_shape, dtype=t.dtype, device=t.device)
                    
                    # Copiar datos del tensor original
                    slices = tuple(slice(0, s) for s in t.shape)
                    adjusted[slices] = t
                    
                    adjusted_tensors.append(adjusted)
            
            tensors = adjusted_tensors
        
        except Exception:
            # Si falla el ajuste, usar la forma del primer tensor
            adjusted_tensors = []
            for t in tensors:
                try:
                    adjusted_tensors.append(t.view(tensors[0].shape))
                except:
                    # Si falla, añadir un tensor de ceros
                    adjusted_tensors.append(torch.zeros_like(tensors[0]))
            
            tensors = adjusted_tensors
        
        # Intentar concatenar
        return torch.cat(tensors, dim=dim)

def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convierte índices a codificación one-hot.
    
    Args:
        indices: Tensor de índices
        num_classes: Número de clases
        
    Returns:
        Tensor one-hot
    """
    if indices.dim() == 1:
        indices = indices.unsqueeze(1)
    
    device = indices.device
    one_hot = torch.zeros(indices.size(0), num_classes, device=device)
    one_hot.scatter_(1, indices, 1)
    
    return one_hot

def normalize(tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
    """
    Normaliza un tensor a media 0 y desviación estándar 1.
    
    Args:
        tensor: Tensor a normalizar
        dim: Dimensión a lo largo de la cual normalizar (None para normalización global)
        
    Returns:
        Tensor normalizado
    """
    if dim is None:
        mean = tensor.mean()
        std = tensor.std()
    else:
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True)
    
    # Evitar división por cero
    std = torch.clamp(std, min=1e-8)
    
    return (tensor - mean) / std

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float = 0.001) -> None:
    """
    Actualiza suavemente los parámetros del modelo objetivo con los del modelo fuente.
    
    Args:
        target: Modelo objetivo
        source: Modelo fuente
        tau: Factor de mezcla (0 < tau < 1)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )

def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """
    Actualiza directamente los parámetros del modelo objetivo con los del modelo fuente.
    
    Args:
        target: Modelo objetivo
        source: Modelo fuente
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convierte un tensor de PyTorch a un array de NumPy.
    
    Args:
        tensor: Tensor de PyTorch
        
    Returns:
        Array de NumPy
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    return tensor.detach().numpy()

def numpy_to_tensor(array: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """
    Convierte un array de NumPy a un tensor de PyTorch.
    
    Args:
        array: Array de NumPy
        device: Dispositivo para el tensor
        
    Returns:
        Tensor de PyTorch
    """
    tensor = torch.from_numpy(array)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor

def batch_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Apila una lista de tensores en un lote.
    
    Args:
        tensors: Lista de tensores
        
    Returns:
        Tensor apilado [batch_size, ...]
    """
    # Verificar que todos los tensores tienen la misma forma
    shapes = [t.shape for t in tensors]
    if len(set(shapes)) > 1:
        # Ajustar a la forma máxima
        max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))
        
        adjusted_tensors = []
        for t in tensors:
            if t.shape == max_shape:
                adjusted_tensors.append(t)
            else:
                # Crear tensor de ceros con la forma máxima
                adjusted = torch.zeros(max_shape, dtype=t.dtype, device=t.device)
                
                # Copiar datos del tensor original
                slices = tuple(slice(0, s) for s in t.shape)
                adjusted[slices] = t
                
                adjusted_tensors.append(adjusted)
        
        tensors = adjusted_tensors
    
    return torch.stack(tensors, dim=0)

def unbatch_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Desapila un tensor en una lista de tensores.
    
    Args:
        tensor: Tensor [batch_size, ...]
        
    Returns:
        Lista de tensores
    """
    return [tensor[i] for i in range(tensor.size(0))] 