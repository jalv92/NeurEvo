"""
Core - Componentes centrales del framework NeurEvo.

Este módulo contiene las clases centrales del framework, incluyendo
la implementación del agente y las definiciones de entorno base.
"""

# Importaciones temporalmente comentadas para evitar problemas
# from neurevo.core.agent import NeurEvoAgent
from neurevo.core.base_environment import BaseEnvironment

# Definir clase temporal para pruebas
class NeurEvoAgent:
    """Clase temporal para pruebas"""
    def __init__(self, *args, **kwargs):
        print("NeurEvoAgent inicializado")

__all__ = ['NeurEvoAgent', 'BaseEnvironment'] 