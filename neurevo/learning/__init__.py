"""
Learning - Algoritmos de aprendizaje para NeurEvo.

Este módulo implementa los diversos algoritmos de aprendizaje utilizados
por NeurEvo, incluyendo curiosidad intrínseca y meta-controladores.
"""

from neurevo.learning.curiosity import CuriosityModule
from neurevo.learning.meta_controller import MetaController
from neurevo.learning.skill_library import SkillLibrary

__all__ = [
    'CuriosityModule',
    'MetaController',
    'SkillLibrary',
] 