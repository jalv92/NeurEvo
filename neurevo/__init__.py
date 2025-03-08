"""
NeurEvo - Framework de aprendizaje por refuerzo con elementos cognitivos y evolutivos.

Este paquete implementa redes neuronales dinámicas que pueden crecer y reducirse
automáticamente, junto con mecanismos de curiosidad intrínseca, memoria episódica
y transferencia de habilidades.
"""

# Importación simplificada para evitar problemas
# from neurevo.brain import BrainInterface, create_brain
# from neurevo.core.agent import NeurEvoAgent
# from neurevo.config.config import NeurEvoConfig

__all__ = [
    # API principal
    'BrainInterface', 
    'create_brain',
    
    # Componentes principales
    'NeurEvoAgent',
    'NeurEvoConfig',
]

__version__ = '0.1.0'

# Definir funciones temporales para pruebas
def create_brain(*args, **kwargs):
    """Función temporal para pruebas"""
    print("Función create_brain llamada")
    return BrainInterface()

class BrainInterface:
    """Clase temporal para pruebas"""
    def __init__(self, *args, **kwargs):
        print("BrainInterface inicializada")
    
    def create_for_environment(self, env_id, **kwargs):
        print(f"Creando agente para entorno: {env_id}")
        return None 