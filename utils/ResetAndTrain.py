"""
Script para reiniciar el entrenamiento de un agente desde cero.

Este script crea un nuevo agente con la misma configuración que un modelo guardado,
pero reiniciando completamente los pesos y el estado de entrenamiento.
"""

import torch
import argparse
import os
import sys
import numpy as np
import time
from typing import Dict, Any, Tuple, List, Optional

# Añadir directorio raíz al path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from neurevo.core.agent import NeurEvoAgent
from neurevo.config.config import NeurEvoConfig
from neurevo.neurevo_main import NeurEvo


def extract_config_from_model(model_path: str) -> Dict[str, Any]:
    """
    Extrae la configuración de un modelo guardado.
    
    Args:
        model_path: Ruta al modelo guardado
        
    Returns:
        Diccionario con la configuración del modelo
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return {}
    
    # Extraer información relevante
    observation_shape = checkpoint.get('observation_shape', None)
    action_size = checkpoint.get('action_size', None)
    
    if observation_shape is None or action_size is None:
        print("El modelo no contiene información de observation_shape o action_size")
        return {}
    
    # Extraer otros parámetros de configuración
    config = {
        'observation_shape': observation_shape,
        'action_size': action_size,
        'gamma': checkpoint.get('gamma', 0.99),
        'epsilon_start': 1.0,  # Reiniciar exploración
        'epsilon_end': checkpoint.get('epsilon_end', 0.05),
        'epsilon_decay': checkpoint.get('epsilon_decay', 0.995),
    }
    
    return config


def create_new_agent(config: Dict[str, Any]) -> Optional[NeurEvoAgent]:
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
        # Crear nuevo agente
        agent = NeurEvoAgent(
            observation_shape=config['observation_shape'],
            action_size=config['action_size'],
            device=device,
            gamma=config.get('gamma', 0.99),
            epsilon_start=config.get('epsilon_start', 1.0),
            epsilon_end=config.get('epsilon_end', 0.05),
            epsilon_decay=config.get('epsilon_decay', 0.995)
        )
        
        return agent
    except Exception as e:
        print(f"Error al crear nuevo agente: {e}")
        return None


def reset_and_train(model_path: str = None, 
                   config: Dict[str, Any] = None,
                   env_class: str = None, 
                   env_kwargs: Dict[str, Any] = None,
                   episodes: int = 500,
                   save_dir: str = None,
                   save_interval: int = 50) -> None:
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
        return
    
    if not agent_config:
        print("No se pudo obtener una configuración válida")
        return
    
    # Crear nuevo agente
    print("Creando nuevo agente con inicialización aleatoria...")
    agent = create_new_agent(agent_config)
    
    if not agent:
        print("No se pudo crear el agente")
        return
    
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
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Cargar entorno si se proporcionó
    if env_class:
        try:
            # Importar dinámicamente la clase de entorno
            module_path, class_name = env_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            env_class = getattr(module, class_name)
            
            # Inicializar entorno
            env_kwargs = env_kwargs or {}
            env = env_class(**env_kwargs)
            
            # Configurar framework NeurEvo para entrenamiento
            neurevo = NeurEvo()
            agent_id = "reset_agent"
            env_id = "training_env"
            
            # Registrar agente y entorno
            neurevo.agents[agent_id] = agent
            neurevo.environments[env_id] = env
            
            # Entrenar
            print(f"Iniciando entrenamiento por {episodes} episodios...")
            start_time = time.time()
            
            metrics = neurevo.train(agent_id, env_id, episodes=episodes, eval_interval=save_interval)
            
            training_time = time.time() - start_time
            print(f"Entrenamiento completado en {training_time:.2f} segundos")
            
            # Guardar agente final
            final_model_path = os.path.join(save_dir, f"agent_reset_episodes_{episodes}.pt")
            neurevo.save_agent(agent_id, final_model_path)
            print(f"Agente final guardado en: {final_model_path}")
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Solo guardar el agente reiniciado
        reset_model_path = os.path.join(save_dir, "agent_reset.pt")
        agent.save(reset_model_path)
        print(f"Agente reiniciado guardado en: {reset_model_path}")
        print("Nota: No se proporcionó entorno, por lo que no se realizó entrenamiento")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Reinicia un agente y lo entrena desde cero.")
    parser.add_argument("--model", "-m", help="Ruta al modelo existente del que extraer la configuración.")
    parser.add_argument("--env", "-e", help="Clase de entorno a utilizar (formato: 'modulo.submodulo.Clase').")
    parser.add_argument("--episodes", "-n", type=int, default=500, help="Número de episodios a entrenar.")
    parser.add_argument("--save-dir", "-d", help="Directorio donde guardar los checkpoints.")
    parser.add_argument("--save-interval", "-i", type=int, default=50, 
                        help="Intervalo de episodios para guardar checkpoints.")
    
    args = parser.parse_args()
    
    if not args.model:
        print("Error: Debe proporcionar la ruta a un modelo existente con --model")
        return
    
    reset_and_train(
        model_path=args.model,
        env_class=args.env,
        episodes=args.episodes,
        save_dir=args.save_dir,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main() 