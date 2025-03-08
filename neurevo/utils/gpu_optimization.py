"""
Optimizaciones para GPU en el framework NeurEvo.
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import time

def optimize_gpu_settings(benchmark: bool = True, deterministic: bool = False, 
                         allow_tf32: bool = True, allow_fp16: bool = True) -> Dict[str, Any]:
    """
    Configura ajustes óptimos para GPU.
    
    Args:
        benchmark: Si se debe habilitar el modo benchmark de cuDNN
        deterministic: Si se debe forzar operaciones deterministas
        allow_tf32: Si se debe permitir TensorFloat-32 en GPUs Ampere+
        allow_fp16: Si se debe permitir precisión FP16
        
    Returns:
        Diccionario con configuración aplicada
    """
    settings = {}
    
    # Verificar disponibilidad de CUDA
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    settings["cuda_available"] = True
    settings["device_count"] = torch.cuda.device_count()
    settings["devices"] = [torch.cuda.get_device_name(i) for i in range(settings["device_count"])]
    
    # Configurar cuDNN
    if benchmark and not deterministic:
        cudnn.benchmark = True
        settings["cudnn_benchmark"] = True
    else:
        cudnn.benchmark = False
        settings["cudnn_benchmark"] = False
    
    # Configurar determinismo
    if deterministic:
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        settings["deterministic"] = True
    else:
        cudnn.deterministic = False
        settings["deterministic"] = False
    
    # Configurar TensorFloat-32 (solo disponible en GPUs Ampere+)
    if allow_tf32 and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        settings["tf32_enabled"] = True
    else:
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            settings["tf32_enabled"] = False
        else:
            settings["tf32_enabled"] = "not_available"
    
    # Configurar FP16
    settings["fp16_supported"] = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    
    return settings

def get_optimal_device() -> torch.device:
    """
    Determina el dispositivo óptimo para cálculos.
    
    Returns:
        Dispositivo óptimo (CUDA o CPU)
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    # Seleccionar GPU con más memoria disponible
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return torch.device("cuda:0")
    
    # Encontrar GPU con más memoria disponible
    max_free_memory = 0
    best_device = 0
    
    for i in range(device_count):
        # Limpiar caché para medición precisa
        torch.cuda.empty_cache()
        
        # Obtener memoria total y asignada
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        free_memory = total_memory - allocated_memory
        
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i
    
    return torch.device(f"cuda:{best_device}")

def enable_amp(model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
    """
    Habilita entrenamiento con precisión mixta automática.
    
    Args:
        model: Modelo a optimizar
        optimizer: Optimizador a utilizar
        
    Returns:
        Tupla con (modelo, optimizador, scaler)
    """
    if not torch.cuda.is_available():
        return model, optimizer, None
    
    # Verificar soporte para AMP
    if hasattr(torch.cuda, "amp") and torch.cuda.is_available():
        # Crear scaler para precisión mixta
        scaler = torch.cuda.amp.GradScaler()
        
        # Mover modelo a CUDA si no está ya
        if next(model.parameters()).device.type != "cuda":
            model = model.cuda()
        
        return model, optimizer, scaler
    
    return model, optimizer, None

def optimize_memory_allocation(reserve_percent: float = 0.9) -> None:
    """
    Optimiza la asignación de memoria CUDA.
    
    Args:
        reserve_percent: Porcentaje de memoria a reservar (0.0 - 1.0)
    """
    if not torch.cuda.is_available():
        return
    
    # Limitar memoria reservada
    for i in range(torch.cuda.device_count()):
        device_properties = torch.cuda.get_device_properties(i)
        total_memory = device_properties.total_memory
        target_memory = int(total_memory * reserve_percent)
        
        # Reservar memoria
        try:
            # Crear tensor grande para reservar memoria
            torch.cuda.empty_cache()
            reserved = torch.empty(target_memory, dtype=torch.uint8, device=f"cuda:{i}")
            # Liberar inmediatamente
            del reserved
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error al reservar memoria en GPU {i}: {e}")

def benchmark_gpu(model: nn.Module, input_shape: Tuple[int, ...], 
                 iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
    """
    Realiza benchmark de rendimiento en GPU.
    
    Args:
        model: Modelo a evaluar
        input_shape: Forma de la entrada
        iterations: Número de iteraciones para el benchmark
        warmup: Número de iteraciones de calentamiento
        
    Returns:
        Diccionario con resultados del benchmark
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA no disponible"}
    
    # Mover modelo a GPU
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    
    # Crear entrada de prueba
    dummy_input = torch.randn(input_shape, device=device)
    
    # Calentamiento
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Sincronizar para medición precisa
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calcular estadísticas
    total_time = end_time - start_time
    avg_time = total_time / iterations
    fps = iterations / total_time
    
    return {
        "total_time": total_time,
        "avg_inference_time": avg_time,
        "fps": fps,
        "iterations": iterations
    }

def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimiza un modelo para inferencia.
    
    Args:
        model: Modelo a optimizar
        
    Returns:
        Modelo optimizado
    """
    # Cambiar a modo evaluación
    model.eval()
    
    # Fusionar BatchNorm con Conv si es posible
    if hasattr(torch, "fx") and hasattr(torch.fx, "symbolic_trace"):
        try:
            # Intentar trazar el modelo
            traced_model = torch.fx.symbolic_trace(model)
            
            # Fusionar operaciones
            from torch.fx.experimental.optimization import fuse
            fused_model = fuse(traced_model)
            
            return fused_model
        except Exception:
            # Si falla, devolver modelo original
            pass
    
    return model

def get_gpu_info() -> Dict[str, Any]:
    """
    Obtiene información detallada sobre las GPUs disponibles.
    
    Returns:
        Diccionario con información de GPUs
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    info = {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "devices": []
    }
    
    for i in range(info["device_count"]):
        device_properties = torch.cuda.get_device_properties(i)
        
        device_info = {
            "name": device_properties.name,
            "compute_capability": f"{device_properties.major}.{device_properties.minor}",
            "total_memory_gb": device_properties.total_memory / (1024**3),
            "multi_processor_count": device_properties.multi_processor_count,
            "max_threads_per_block": device_properties.max_threads_per_block,
            "max_threads_per_mp": device_properties.max_threads_per_multi_processor,
            "memory_allocated_mb": torch.cuda.memory_allocated(i) / (1024**2),
            "memory_reserved_mb": torch.cuda.memory_reserved(i) / (1024**2)
        }
        
        info["devices"].append(device_info)
    
    return info

def set_gpu_power_mode(mode: str = "high") -> bool:
    """
    Configura el modo de energía de la GPU (solo funciona en algunas GPUs NVIDIA).
    
    Args:
        mode: Modo de energía ('high', 'medium', 'low')
        
    Returns:
        True si se aplicó correctamente
    """
    if not torch.cuda.is_available():
        return False
    
    # Esta función requiere acceso al driver NVIDIA
    # Solo funciona en sistemas con utilidades NVIDIA instaladas
    try:
        if mode == "high":
            os.system("nvidia-smi -pm 1")  # Persistencia activada
            os.system("nvidia-smi -ac 4004,1512")  # Máxima frecuencia
        elif mode == "medium":
            os.system("nvidia-smi -pm 1")
            os.system("nvidia-smi -ac 3004,1127")  # Frecuencia media
        elif mode == "low":
            os.system("nvidia-smi -pm 0")  # Persistencia desactivada
            os.system("nvidia-smi -ac 1500,900")  # Frecuencia baja
        else:
            return False
        
        return True
    except Exception:
        return False 