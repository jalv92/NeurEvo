#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para reiniciar el entrenamiento de un agente desde cero.

Este script crea un nuevo agente con la misma configuración que un modelo guardado,
pero reiniciando completamente los pesos y el estado de entrenamiento.
"""

import torch
import argparse
import os
import sys
import json
import importlib
import importlib.util
import numpy as np
import time
import traceback
from typing import Dict, Any, Tuple, List, Optional, Union, TypeVar, Type

# Determinación absoluta del directorio raíz del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

# Asegurarnos de que el directorio raíz esté en el path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Definir tipos para anotaciones
# Esto evita los errores de importación en Pylance
T = TypeVar('T')
NeurEvoAgentType = TypeVar('NeurEvoAgentType')
BaseEnvironmentType = TypeVar('BaseEnvironmentType')
NeurEvoType = TypeVar('NeurEvoType')
NeurEvoConfigType = TypeVar('NeurEvoConfigType')

# Intentar múltiples métodos para las importaciones
def import_module_safely(module_path: str) -> Optional[Any]:
    """
    Intenta importar un módulo utilizando diferentes métodos.
    
    Args:
        module_path: Ruta al módulo
        
    Returns:
        El módulo importado o None si falla
    """
    # Método 1: Importación estándar
    try:
        return importlib.import_module(module_path)
    except ImportError:
        pass
    
    # Método 2: Importación relativa
    try:
        if module_path.startswith('neurevo.'):
            return importlib.import_module(module_path[len('neurevo.'):])
        return None
    except ImportError:
        pass
    
    # Método 3: Importación desde ruta absoluta
    try:
        module_file = os.path.join(root_dir, *module_path.split('.'))
        if os.path.exists(module_file + '.py'):
            spec = importlib.util.spec_from_file_location(module_path, module_file + '.py')
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    except Exception:
        pass
    
    return None

# Declarar variables para las clases con valores por defecto (type: ignore)
# Esto solucionará los errores de importación en Pylance
NeurEvoAgent = None  # type: ignore
NeurEvoConfig = None  # type: ignore
NeurEvo = None  # type: ignore
BaseEnvironment = None  # type: ignore

# Intentar importar los módulos necesarios
try:
    # Intentar importaciones absolutas primero
    # Usamos from ... import ... as _ClassName y luego asignamos a ClassName
    # para que Pylance no marque errores en estas líneas
    from neurevo.core.agent import NeurEvoAgent as _NeurEvoAgent
    from neurevo.config.config import NeurEvoConfig as _NeurEvoConfig
    from neurevo.neurevo_main import NeurEvo as _NeurEvo
    from neurevo.core.base_environment import BaseEnvironment as _BaseEnvironment
    
    # Asignar a las variables globales
    NeurEvoAgent = _NeurEvoAgent
    NeurEvoConfig = _NeurEvoConfig
    NeurEvo = _NeurEvo
    BaseEnvironment = _BaseEnvironment
    
    print("Importaciones absolutas exitosas.")
except ImportError:
    print("Importaciones absolutas fallidas, intentando alternativas...")
    
    # Intentar importaciones relativas
    try:
        sys.path.insert(0, os.path.abspath(root_dir))
        
        # Intentar importar cada módulo individualmente
        agent_module = import_module_safely('neurevo.core.agent')
        config_module = import_module_safely('neurevo.config.config')
        main_module = import_module_safely('neurevo.neurevo_main')
        env_module = import_module_safely('neurevo.core.base_environment')
        
        # Alternativas para importaciones desde carpetas directas
        if agent_module is None:
            agent_module = import_module_safely('core.agent')
        if config_module is None:
            config_module = import_module_safely('config.config')
        if main_module is None:
            main_module = import_module_safely('neurevo_main')
        if env_module is None:
            env_module = import_module_safely('core.base_environment')
        
        # Verificar que se hayan importado todos los módulos
        if not all([agent_module, config_module, main_module]):
            raise ImportError("No se pudieron importar todos los módulos necesarios")
        
        # Asignar las clases
        NeurEvoAgent = getattr(agent_module, 'NeurEvoAgent')
        NeurEvoConfig = getattr(config_module, 'NeurEvoConfig')
        NeurEvo = getattr(main_module, 'NeurEvo')
        BaseEnvironment = getattr(env_module, 'BaseEnvironment') if env_module else None
        
        print("Importaciones alternativas exitosas.")
    except Exception as e:
        print(f"Error en importaciones alternativas: {e}")
        print(traceback.format_exc())
        print("\nNo se pudieron importar los módulos necesarios.")
        print("Asegúrese de que el script se ejecute desde el directorio raíz del proyecto.")
        print("También puede intentar ejecutar: python -m utils.ResetAndTrain [argumentos]")
        sys.exit(1)


def extract_config_from_model(model_path: str) -> Dict[str, Any]:
    """
    Extrae la configuración de un modelo guardado.
    
    Args:
        model_path: Ruta al modelo guardado
        
    Returns:
        Diccionario con la configuración del modelo
    """
    try:
        # Convertir a ruta absoluta si es relativa
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
            
        # Verificar existencia del archivo
        if not os.path.exists(model_path):
            print(f"Error: No se encontró el archivo {model_path}")
            return {}
            
        # Cargar el modelo con manejo de errores
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except RuntimeError:
            print(f"Error al cargar el modelo con torch.load. Intentando método alternativo...")
            # Intentar carga con pickle
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
                
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print(traceback.format_exc())
        return {}
    
    # Extraer información relevante con verificaciones
    observation_shape = checkpoint.get('observation_shape', None)
    action_size = checkpoint.get('action_size', None)
    
    # Verificar campos críticos
    if observation_shape is None:
        print("Advertencia: El modelo no contiene información de observation_shape")
        print("Campos disponibles:", list(checkpoint.keys()))
        # Intentar encontrar en otras ubicaciones
        if 'perception_state' in checkpoint:
            print("Intentando extraer dimensiones del perception_state...")
        return {}
        
    if action_size is None:
        print("Advertencia: El modelo no contiene información de action_size")
        # Intentar encontrar en otras ubicaciones
        if 'executive_state' in checkpoint:
            print("Intentando extraer dimensiones del executive_state...")
        return {}
    
    # Convertir observation_shape a tupla si es lista
    if isinstance(observation_shape, list):
        observation_shape = tuple(observation_shape)
    
    # Extraer otros parámetros de configuración
    config = {
        'observation_shape': observation_shape,
        'action_size': action_size,
        'gamma': checkpoint.get('gamma', 0.99),
        'epsilon_start': 1.0,  # Reiniciar exploración
        'epsilon_end': checkpoint.get('epsilon_end', 0.05),
        'epsilon_decay': checkpoint.get('epsilon_decay', 0.995),
        'learning_rate': checkpoint.get('learning_rate', 0.001),
        'batch_size': checkpoint.get('batch_size', 64),
        'hidden_layers': checkpoint.get('hidden_layers', [128, 128]),
    }
    
    print(f"Configuración extraída con éxito: {config}")
    return config


def create_new_agent(config: Dict[str, Any]) -> Optional[Any]:
    """
    Crea un nuevo agente con la configuración especificada.
    
    Args:
        config: Configuración para el nuevo agente
        
    Returns:
        Un nuevo agente con la configuración dada
    """
    if not config:
        return None
    
    # Determinar el dispositivo adecuado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Preparar los parámetros (con valores por defecto para evitar errores)
        observation_shape = config['observation_shape']
        action_size = config['action_size']
        
        # Asegurarse de que observation_shape sea una tupla
        if isinstance(observation_shape, list):
            observation_shape = tuple(observation_shape)
        
        # Si observation_shape es un solo número, convertirlo a tupla
        if isinstance(observation_shape, (int, float)):
            observation_shape = (int(observation_shape),)
            
        # Verificar tipos para evitar errores
        assert isinstance(action_size, int), f"action_size debe ser un entero, no {type(action_size)}"
        
        # Imprimir información de depuración
        print(f"Creando agente con: observation_shape={observation_shape} ({type(observation_shape)}), "
              f"action_size={action_size} ({type(action_size)})")
        
        # Verificar que NeurEvoAgent está disponible
        if NeurEvoAgent is None:
            print("Error: NeurEvoAgent no está disponible")
            return None
            
        # Crear nuevo agente con configuración completa
        agent = NeurEvoAgent(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
            gamma=config.get('gamma', 0.99),
            epsilon_start=config.get('epsilon_start', 1.0),
            epsilon_end=config.get('epsilon_end', 0.05),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            learning_rate=config.get('learning_rate', 0.001),
            batch_size=config.get('batch_size', 64),
            hidden_layers=config.get('hidden_layers', [128, 128])
        )
        
        # Verificar que el agente se creó correctamente
        if agent:
            print(f"Agente creado en dispositivo: {device}")
            print(f"Configuración: gamma={config.get('gamma', 0.99)}, "
                  f"epsilon_start={config.get('epsilon_start', 1.0)}, "
                  f"batch_size={config.get('batch_size', 64)}")
        
        return agent
    except Exception as e:
        print(f"Error al crear nuevo agente: {e}")
        print(traceback.format_exc())
        return None


def import_environment_class(env_class_path: str) -> Optional[type]:
    """
    Importa dinámicamente una clase de entorno.
    
    Args:
        env_class_path: Ruta a la clase (formato: 'modulo.submodulo.Clase')
        
    Returns:
        La clase importada o None si falla
    """
    try:
        print(f"Intentando importar entorno: {env_class_path}")
        
        # Dividir la ruta en módulo y nombre de clase
        if '.' in env_class_path:
            module_path, class_name = env_class_path.rsplit('.', 1)
            
            # Intentar múltiples métodos de importación
            module = import_module_safely(module_path)
            
            if module is None:
                print(f"No se pudo importar el módulo: {module_path}")
                return None
                
            # Obtener la clase del módulo
            env_class = getattr(module, class_name, None)
            
            if env_class is None:
                print(f"La clase {class_name} no existe en el módulo {module_path}")
                # Listar clases disponibles en el módulo
                print(f"Clases disponibles: {[name for name in dir(module) if not name.startswith('_')]}")
                return None
            
            # Verificar que la clase es adecuada (subclase o compatible con BaseEnvironment)
            if hasattr(env_class, 'reset') and hasattr(env_class, 'step'):
                print(f"Clase de entorno {class_name} importada correctamente")
                return env_class
            else:
                print(f"Error: La clase {class_name} no tiene los métodos reset() y step() requeridos")
                return None
        else:
            print(f"Error: Formato incorrecto para env_class. Debe ser 'modulo.submodulo.Clase'")
            return None
    except Exception as e:
        print(f"Error al importar la clase de entorno: {e}")
        print(traceback.format_exc())
        return None


def reset_and_train(model_path: Optional[str] = None, 
                   config: Optional[Dict[str, Any]] = None,
                   env_class: Optional[str] = None, 
                   env_kwargs: Optional[Dict[str, Any]] = None,
                   episodes: int = 500,
                   save_dir: Optional[str] = None,
                   save_interval: int = 50) -> bool:
    """
    Reinicia y entrena un nuevo agente, opcionalmente basado en un modelo existente.
    
    Args:
        model_path: Ruta al modelo guardado (opcional)
        config: Configuración directa para el agente (alternativa a model_path)
        env_class: Nombre de la clase de entorno a utilizar
        env_kwargs: Argumentos para inicializar el entorno
        episodes: Número de episodios a entrenar
        save_dir: Directorio donde guardar los checkpoints
        save_interval: Cada cuántos episodios guardar un checkpoint
        
    Returns:
        True si el proceso fue exitoso, False en caso contrario
    """
    # Determinar configuración
    if model_path:
        print(f"Extrayendo configuración de modelo existente: {model_path}")
        agent_config = extract_config_from_model(model_path)
    elif config:
        print("Usando configuración proporcionada")
        agent_config = config
    else:
        print("Error: Debe proporcionar model_path o config")
        return False
    
    if not agent_config:
        print("No se pudo obtener una configuración válida")
        return False
    
    # Crear nuevo agente
    print("Creando nuevo agente con inicialización aleatoria...")
    agent = create_new_agent(agent_config)
    
    if not agent:
        print("No se pudo crear el agente")
        return False
    
    # Mostrar información del agente
    print(f"Nuevo agente creado:")
    print(f"  - Forma de observación: {agent.observation_shape}")
    print(f"  - Tamaño de acción: {agent.action_size}")
    
    # Configurar directorio de guardado
    if save_dir is None:
        if model_path:
            save_dir = os.path.dirname(model_path)
        else:
            save_dir = os.path.join(os.getcwd(), "models")
    
    # Crear directorio si no existe
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Directorio creado: {save_dir}")
        except Exception as e:
            print(f"Error al crear directorio {save_dir}: {e}")
            # Usar un directorio alternativo
            save_dir = os.path.join(os.getcwd(), "reset_models")
            os.makedirs(save_dir, exist_ok=True)
            print(f"Usando directorio alternativo: {save_dir}")
    
    # Guardar primero el agente reiniciado antes de entrenar
    initial_model_path = os.path.join(save_dir, "agent_reset_initial.pt")
    try:
        agent.save(initial_model_path)
        print(f"Agente inicial guardado en: {initial_model_path}")
    except Exception as e:
        print(f"Error al guardar el agente inicial: {e}")
    
    # Cargar entorno si se proporcionó
    if env_class:
        try:
            # Importar dinámicamente la clase de entorno
            env_class_obj = import_environment_class(env_class)
            if not env_class_obj:
                print("No se pudo importar la clase de entorno")
                
                # Guardar agente sin entrenar
                reset_model_path = os.path.join(save_dir, "agent_reset.pt")
                agent.save(reset_model_path)
                print(f"Agente reiniciado guardado en: {reset_model_path} (sin entrenar)")
                return False
                
            # Inicializar entorno con argumentos
            env_kwargs = env_kwargs or {}
            print(f"Inicializando entorno {env_class} con argumentos: {env_kwargs}")
            
            try:
                env = env_class_obj(**env_kwargs)
            except TypeError as e:
                print(f"Error al inicializar entorno: {e}")
                print("El entorno podría requerir diferentes argumentos. Intentando sin argumentos...")
                env = env_class_obj()
            
            # Verificar compatibilidad del entorno
            if not hasattr(env, 'reset') or not hasattr(env, 'step'):
                print("Error: El entorno no implementa los métodos reset() y step() requeridos")
                return False
            
            # Verificar interfaz del entorno
            try:
                initial_state = env.reset()
                print(f"Estado inicial del entorno: tipo={type(initial_state)}, shape={getattr(initial_state, 'shape', 'desconocido')}")
                
                # Verificar compatibilidad de dimensiones
                if hasattr(initial_state, 'shape'):
                    state_shape = initial_state.shape if isinstance(initial_state, np.ndarray) else (len(initial_state),)
                    if state_shape != agent.observation_shape:
                        print(f"Advertencia: La forma del estado ({state_shape}) no coincide con observation_shape del agente ({agent.observation_shape})")
            except Exception as e:
                print(f"Error al verificar la interfaz del entorno: {e}")
                print(traceback.format_exc())
                # Continuar a pesar del error
                
            # Configurar framework NeurEvo para entrenamiento
            try:
                if NeurEvo is None:
                    print("Error: NeurEvo no está disponible")
                    return False
                    
                neurevo = NeurEvo()
                agent_id = "reset_agent"
                env_id = "training_env"
                
                # Registrar agente y entorno
                neurevo.agents[agent_id] = agent
                neurevo.environments[env_id] = env
                
                # Entrenar
                print(f"Iniciando entrenamiento por {episodes} episodios...")
                start_time = time.time()
                
                try:
                    metrics = neurevo.train(agent_id, env_id, episodes=episodes, eval_interval=save_interval)
                    
                    # Guardar métricas
                    metrics_path = os.path.join(save_dir, f"metrics_reset_{episodes}.json")
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    print(f"Métricas guardadas en: {metrics_path}")
                    
                except Exception as e:
                    print(f"Error durante el entrenamiento: {e}")
                    print(traceback.format_exc())
                    
                    # Intentar guardar el agente parcialmente entrenado
                    partial_model_path = os.path.join(save_dir, f"agent_reset_partial.pt")
                    try:
                        agent.save(partial_model_path)
                        print(f"Agente parcialmente entrenado guardado en: {partial_model_path}")
                    except Exception as save_error:
                        print(f"No se pudo guardar el agente parcialmente entrenado: {save_error}")
                    return False
                
                training_time = time.time() - start_time
                print(f"Entrenamiento completado en {training_time:.2f} segundos")
                
                # Guardar agente final
                final_model_path = os.path.join(save_dir, f"agent_reset_episodes_{episodes}.pt")
                try:
                    agent.save(final_model_path)
                    print(f"Agente final guardado en: {final_model_path}")
                except Exception as e:
                    print(f"Error al guardar el agente final: {e}")
                    
                    # Intentar guardar con neurevo
                    try:
                        neurevo.save_agent(agent_id, final_model_path)
                        print(f"Agente final guardado con NeurEvo en: {final_model_path}")
                    except Exception as save_error:
                        print(f"No se pudo guardar el agente final: {save_error}")
                        return False
                
                return True
                
            except Exception as e:
                print(f"Error al configurar NeurEvo: {e}")
                print(traceback.format_exc())
                return False
                
        except Exception as e:
            print(f"Error general durante el proceso: {e}")
            print(traceback.format_exc())
            return False
    else:
        # Solo guardar el agente reiniciado
        reset_model_path = os.path.join(save_dir, "agent_reset.pt")
        try:
            agent.save(reset_model_path)
            print(f"Agente reiniciado guardado en: {reset_model_path}")
            print("Nota: No se proporcionó entorno, por lo que no se realizó entrenamiento")
            return True
        except Exception as e:
            print(f"Error al guardar el agente: {e}")
            return False


def parse_json_arg(arg_string: str) -> Any:
    """
    Convierte una cadena JSON en un objeto Python.
    
    Args:
        arg_string: Cadena JSON
        
    Returns:
        Objeto Python convertido desde JSON
    """
    if not arg_string:
        return None
        
    try:
        # Intentar analizar como JSON
        return json.loads(arg_string)
    except json.JSONDecodeError:
        # Si falla, verificar si es una ruta a un archivo JSON
        if os.path.exists(arg_string) and arg_string.endswith('.json'):
            try:
                with open(arg_string, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
                
        # Si todo falla, devolver la cadena original
        print(f"Advertencia: No se pudo analizar como JSON: '{arg_string}'. Usando como cadena.")
        return arg_string


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Reinicia un agente y lo entrena desde cero.")
    parser.add_argument("--model", "-m", help="Ruta al modelo existente del que extraer la configuración.")
    parser.add_argument("--env", "-e", help="Clase de entorno a utilizar (formato: 'modulo.submodulo.Clase').")
    parser.add_argument("--env-args", "-a", help="Argumentos para el entorno en formato JSON (ej: '{\"arg1\": valor}') o ruta a un archivo JSON.")
    parser.add_argument("--episodes", "-n", type=int, default=500, help="Número de episodios a entrenar.")
    parser.add_argument("--save-dir", "-d", help="Directorio donde guardar los checkpoints.")
    parser.add_argument("--save-interval", "-i", type=int, default=50, 
                        help="Intervalo de episodios para guardar checkpoints.")
    parser.add_argument("--config", "-c", help="Archivo JSON con configuración personalizada para el agente o cadena JSON.")
    parser.add_argument("--debug", action="store_true", help="Activa el modo de depuración con más información.")
    
    args = parser.parse_args()
    
    # Activar modo de depuración si se solicita
    if args.debug:
        print("Modo de depuración activado")
        print(f"Directorio actual: {os.getcwd()}")
        print(f"Directorio del script: {script_dir}")
        print(f"Directorio raíz: {root_dir}")
        print(f"sys.path: {sys.path}")
    
    # Verificar si se proporcionó un modelo o configuración personalizada
    if not args.model and not args.config:
        print("Error: Debe proporcionar una ruta al modelo (--model) o un archivo de configuración (--config)")
        return
    
    # Cargar configuración personalizada si se proporciona
    config = None
    if args.config:
        try:
            config = parse_json_arg(args.config)
            if isinstance(config, str):
                # Intentar cargar como archivo
                with open(config, 'r') as f:
                    config = json.load(f)
            
            print(f"Configuración cargada: {config}")
        except Exception as e:
            print(f"Error al cargar el archivo de configuración: {e}")
            print(traceback.format_exc())
            return
    
    # Cargar argumentos del entorno si se proporcionan
    env_kwargs = {}
    if args.env_args:
        try:
            env_kwargs = parse_json_arg(args.env_args)
            if not isinstance(env_kwargs, dict):
                print(f"Error: Los argumentos del entorno deben ser un diccionario, no {type(env_kwargs)}")
                return
        except Exception as e:
            print(f"Error al procesar argumentos del entorno: {e}")
            print(traceback.format_exc())
            return
    
    # Ejecutar con los argumentos proporcionados
    success = reset_and_train(
        model_path=args.model,
        config=config,
        env_class=args.env,
        env_kwargs=env_kwargs,
        episodes=args.episodes,
        save_dir=args.save_dir,
        save_interval=args.save_interval
    )
    
    # Reportar resultado
    if success:
        print("\n¡Proceso completado exitosamente!")
    else:
        print("\nEl proceso falló. Revise los mensajes de error anteriores.")
        sys.exit(1)


if __name__ == "__main__":
    main() 