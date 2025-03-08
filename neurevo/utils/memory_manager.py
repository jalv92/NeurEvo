"""
Administrador de memoria para optimizar el uso de recursos en el framework NeurEvo.
"""

import gc
import torch
import time
import psutil
import os
import numpy as np
from typing import Dict, List, Any, Optional

class MemoryManager:
    """
    Administrador de memoria para garantizar una gestión eficiente de recursos CUDA y RAM.
    Monitorea y optimiza el uso de memoria durante el entrenamiento.
    """
    
    def __init__(self, agent=None, memory_threshold: float = 0.9, 
                enable_auto_cleanup: bool = True, cleanup_interval: int = 10):
        """
        Inicializa el administrador de memoria.
        
        Args:
            agent: Agente a monitorear (opcional)
            memory_threshold: Umbral de uso de memoria para activar limpieza (0.0 - 1.0)
            enable_auto_cleanup: Si se debe habilitar la limpieza automática
            cleanup_interval: Intervalo de episodios entre limpiezas
        """
        self.agent = agent
        self.memory_threshold = memory_threshold
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        # Contadores y temporizadores
        self.episode_counter = 0
        self.last_clear_time = time.time()
        self.clear_interval = 30  # segundos entre limpiezas profundas
        
        # Estadísticas de memoria
        self.memory_stats = []
        self.peak_memory_usage = 0.0
        self.last_memory_usage = 0.0
        
        # Detectar dispositivo
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.device_count = torch.cuda.device_count()
            self.device_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
        else:
            self.device_count = 0
            self.device_names = []
    
    def register_agent(self, agent) -> None:
        """
        Registra un agente para monitorear.
        
        Args:
            agent: Agente a monitorear
        """
        self.agent = agent
    
    def after_episode_cleanup(self) -> None:
        """
        Ejecuta limpieza después de cada episodio.
        """
        self.episode_counter += 1
        
        # Limpiar el recolector de basura de Python
        gc.collect()
        
        # Si hay CUDA disponible, limpiar caché
        if self.has_cuda:
            torch.cuda.empty_cache()
            
            # Cada 5 episodios, mostrar estadísticas de memoria
            if self.episode_counter % 5 == 0:
                self._log_memory_stats()
        
        # Limpieza profunda si es necesario
        current_time = time.time()
        if (current_time - self.last_clear_time > self.clear_interval or 
            self.episode_counter % self.cleanup_interval == 0):
            self._deep_cleanup()
            self.last_clear_time = current_time
    
    def _log_memory_stats(self) -> Dict[str, float]:
        """
        Registra estadísticas de uso de memoria.
        
        Returns:
            Diccionario con estadísticas de memoria
        """
        stats = {}
        
        # Estadísticas de RAM
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
        ram_percent = process.memory_percent()
        
        stats["ram_usage_mb"] = ram_usage
        stats["ram_percent"] = ram_percent
        
        # Estadísticas de CUDA
        if self.has_cuda:
            for i in range(self.device_count):
                cuda_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # MB
                cuda_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)  # MB
                cuda_max_allocated = torch.cuda.max_memory_allocated(i) / (1024 * 1024)  # MB
                
                stats[f"cuda{i}_allocated_mb"] = cuda_allocated
                stats[f"cuda{i}_reserved_mb"] = cuda_reserved
                stats[f"cuda{i}_max_allocated_mb"] = cuda_max_allocated
                
                # Actualizar pico de uso
                if cuda_allocated > self.peak_memory_usage:
                    self.peak_memory_usage = cuda_allocated
                
                self.last_memory_usage = cuda_allocated
        
        # Guardar estadísticas
        self.memory_stats.append(stats)
        
        # Limitar historial de estadísticas
        if len(self.memory_stats) > 100:
            self.memory_stats.pop(0)
        
        return stats
    
    def _deep_cleanup(self) -> None:
        """
        Realiza una limpieza profunda de memoria.
        """
        # Limpiar caché de PyTorch
        if self.has_cuda:
            torch.cuda.empty_cache()
        
        # Forzar recolección de basura
        gc.collect()
        
        # Liberar memoria no utilizada al sistema
        if hasattr(torch.cuda, 'memory_summary'):
            for i in range(self.device_count):
                torch.cuda.memory_summary(device=i, abbreviated=True)
        
        # Si el agente tiene buffer de experiencia, compactar si es necesario
        if self.agent and hasattr(self.agent, 'memory'):
            if hasattr(self.agent.memory, 'buffer') and len(self.agent.memory.buffer) > 0:
                # Verificar si el uso de memoria es alto
                if self.last_memory_usage / self.peak_memory_usage > self.memory_threshold:
                    # Reducir tamaño del buffer temporalmente
                    original_size = len(self.agent.memory.buffer)
                    if original_size > 1000:
                        # Mantener solo los elementos más recientes
                        temp_buffer = list(self.agent.memory.buffer)[-1000:]
                        self.agent.memory.buffer.clear()
                        self.agent.memory.buffer.extend(temp_buffer)
                        
                        # Limpiar memoria nuevamente
                        gc.collect()
                        if self.has_cuda:
                            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Obtiene el uso actual de memoria.
        
        Returns:
            Diccionario con uso de memoria
        """
        return self._log_memory_stats()
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """
        Analiza la tendencia de uso de memoria.
        
        Returns:
            Diccionario con tendencias de memoria
        """
        if not self.memory_stats:
            return {"trend": "unknown", "growth_rate": 0.0}
        
        # Analizar tendencia de CUDA si está disponible
        if self.has_cuda and self.device_count > 0:
            cuda_usage = [stats.get(f"cuda0_allocated_mb", 0) for stats in self.memory_stats]
            
            if len(cuda_usage) >= 5:
                # Calcular tasa de crecimiento
                recent_usage = cuda_usage[-5:]
                if recent_usage[0] > 0:
                    growth_rate = (recent_usage[-1] - recent_usage[0]) / recent_usage[0]
                else:
                    growth_rate = 0.0
                
                # Determinar tendencia
                if growth_rate > 0.1:
                    trend = "increasing"
                elif growth_rate < -0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                return {"trend": trend, "growth_rate": growth_rate}
        
        # Analizar tendencia de RAM si no hay CUDA
        ram_usage = [stats.get("ram_percent", 0) for stats in self.memory_stats]
        
        if len(ram_usage) >= 5:
            # Calcular tasa de crecimiento
            recent_usage = ram_usage[-5:]
            if recent_usage[0] > 0:
                growth_rate = (recent_usage[-1] - recent_usage[0]) / recent_usage[0]
            else:
                growth_rate = 0.0
            
            # Determinar tendencia
            if growth_rate > 0.1:
                trend = "increasing"
            elif growth_rate < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"
            
            return {"trend": trend, "growth_rate": growth_rate}
        
        return {"trend": "unknown", "growth_rate": 0.0}
    
    def optimize_memory_usage(self) -> None:
        """
        Optimiza el uso de memoria basándose en tendencias.
        """
        trend = self.get_memory_trend()
        
        if trend["trend"] == "increasing" and trend["growth_rate"] > 0.2:
            # Crecimiento rápido: realizar limpieza profunda
            self._deep_cleanup()
            
            # Reducir intervalo de limpieza
            self.cleanup_interval = max(1, self.cleanup_interval // 2)
            self.clear_interval = max(10, self.clear_interval // 2)
        
        elif trend["trend"] == "stable" or trend["trend"] == "decreasing":
            # Uso estable o decreciente: aumentar intervalo de limpieza
            self.cleanup_interval = min(20, self.cleanup_interval * 2)
            self.clear_interval = min(60, self.clear_interval * 2)
    
    def get_memory_summary(self) -> str:
        """
        Genera un resumen del uso de memoria.
        
        Returns:
            Cadena con resumen de memoria
        """
        if not self.memory_stats:
            return "No hay estadísticas de memoria disponibles"
        
        latest = self.memory_stats[-1]
        
        summary = "Resumen de Memoria:\n"
        summary += f"RAM: {latest.get('ram_usage_mb', 0):.2f} MB ({latest.get('ram_percent', 0):.2f}%)\n"
        
        if self.has_cuda:
            for i in range(self.device_count):
                summary += f"CUDA {i} ({self.device_names[i]}):\n"
                summary += f"  Asignado: {latest.get(f'cuda{i}_allocated_mb', 0):.2f} MB\n"
                summary += f"  Reservado: {latest.get(f'cuda{i}_reserved_mb', 0):.2f} MB\n"
                summary += f"  Pico: {latest.get(f'cuda{i}_max_allocated_mb', 0):.2f} MB\n"
        
        trend = self.get_memory_trend()
        summary += f"Tendencia: {trend['trend']} (tasa: {trend['growth_rate']:.2f})\n"
        
        return summary 