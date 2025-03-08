"""
Utilidades del framework NeurEvo.
"""

from neurevo.utils.tensor_utils import (
    validate_tensor_shape, safe_cat, to_one_hot, normalize,
    soft_update, hard_update, tensor_to_numpy, numpy_to_tensor,
    batch_tensors, unbatch_tensor
)
from neurevo.utils.memory_manager import MemoryManager
from neurevo.utils.gpu_optimization import (
    optimize_gpu_settings, get_optimal_device, enable_amp,
    optimize_memory_allocation, benchmark_gpu, optimize_model_for_inference,
    get_gpu_info, set_gpu_power_mode
)

__all__ = [
    # Tensor utils
    'validate_tensor_shape', 'safe_cat', 'to_one_hot', 'normalize',
    'soft_update', 'hard_update', 'tensor_to_numpy', 'numpy_to_tensor',
    'batch_tensors', 'unbatch_tensor',
    
    # Memory management
    'MemoryManager',
    
    # GPU optimization
    'optimize_gpu_settings', 'get_optimal_device', 'enable_amp',
    'optimize_memory_allocation', 'benchmark_gpu', 'optimize_model_for_inference',
    'get_gpu_info', 'set_gpu_power_mode'
]
