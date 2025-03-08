"""
Script para corregir inconsistencias dimensionales en modelos guardados.

Este script analiza un modelo guardado, identifica inconsistencias dimensionales
entre los módulos y crea un nuevo modelo con las dimensiones corregidas.
"""

import torch
import argparse
import os
import sys
import numpy as np
from typing import Dict, Any, Tuple, List

# Añadir directorio raíz al path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from neurevo.core.agent import NeurEvoAgent


def analyze_model(model_path: str) -> Dict[str, Any]:
    """
    Analiza un modelo guardado para detectar inconsistencias dimensionales.
    
    Args:
        model_path: Ruta al modelo guardado
        
    Returns:
        Diccionario con información sobre el modelo y sus posibles inconsistencias
    """
    # Cargar modelo
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
    
    # Extraer información de dimensiones
    module_dimensions = checkpoint.get('module_dimensions', {})
    
    # Comprobar si se almacenaron dimensiones
    if not module_dimensions:
        print("Advertencia: El modelo no contiene información explícita de dimensiones entre módulos")
    
    # Analizar posibles inconsistencias
    inconsistencies = []
    
    # Si tenemos información de dimensiones, verificar consistencia
    if module_dimensions:
        perception_output = module_dimensions.get('perception_output', None)
        prediction_input = module_dimensions.get('prediction_input', None)
        executive_input = module_dimensions.get('executive_input', None)
        
        if perception_output and prediction_input and perception_output != prediction_input:
            inconsistencies.append(f"Inconsistencia: perception_output={perception_output} != prediction_input={prediction_input}")
        
        if perception_output and executive_input and perception_output != executive_input:
            inconsistencies.append(f"Inconsistencia: perception_output={perception_output} != executive_input={executive_input}")
    
    return {
        'observation_shape': observation_shape,
        'action_size': action_size,
        'module_dimensions': module_dimensions,
        'inconsistencies': inconsistencies,
        'episodes_trained': checkpoint.get('episodes_trained', 0),
        'raw_checkpoint': checkpoint
    }


def fix_model_dimensions(model_path: str, output_path: str = None) -> str:
    """
    Corrige las inconsistencias dimensionales en un modelo guardado.
    
    Args:
        model_path: Ruta al modelo guardado con inconsistencias
        output_path: Ruta donde guardar el modelo corregido (opcional)
        
    Returns:
        Ruta al modelo corregido
    """
    # Determinar ruta de salida si no se proporciona
    if output_path is None:
        base_dir = os.path.dirname(model_path)
        base_name = os.path.basename(model_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(base_dir, f"{name}_fixed{ext}")
    
    # Analizar modelo
    print(f"Analizando modelo: {model_path}")
    model_info = analyze_model(model_path)
    
    if not model_info:
        print("No se pudo analizar el modelo")
        return None
    
    # Extraer información necesaria
    observation_shape = model_info['observation_shape']
    action_size = model_info['action_size']
    checkpoint = model_info['raw_checkpoint']
    
    print(f"Forma de observación: {observation_shape}")
    print(f"Tamaño de acción: {action_size}")
    
    # Mostrar inconsistencias encontradas
    if model_info['inconsistencies']:
        print("Inconsistencias detectadas:")
        for inconsistency in model_info['inconsistencies']:
            print(f"  - {inconsistency}")
    else:
        print("No se detectaron inconsistencias dimensionales explícitas")
    
    # Crear un nuevo agente con la misma configuración
    print("Creando un nuevo agente con dimensiones consistentes...")
    
    # Determinar el dispositivo adecuado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear nuevo agente
    agent = NeurEvoAgent(
        observation_shape=observation_shape,
        action_size=action_size,
        device=device
    )
    
    # Intentar cargar los estados de los módulos con manejo de errores
    try:
        print("Cargando estados de módulos...")
        
        # Cargar cada módulo individualmente para mejor manejo de errores
        try:
            agent.perception.load_state_dict(checkpoint['perception_state'])
            print("  - Módulo perception cargado correctamente")
        except Exception as e:
            print(f"  - Error al cargar módulo perception: {e}")
            print("    Manteniendo inicialización aleatoria para este módulo")
        
        try:
            agent.prediction.load_state_dict(checkpoint['prediction_state'])
            print("  - Módulo prediction cargado correctamente")
        except Exception as e:
            print(f"  - Error al cargar módulo prediction: {e}")
            print("    Manteniendo inicialización aleatoria para este módulo")
        
        try:
            agent.executive.load_state_dict(checkpoint['executive_state'])
            print("  - Módulo executive cargado correctamente")
        except Exception as e:
            print(f"  - Error al cargar módulo executive: {e}")
            print("    Manteniendo inicialización aleatoria para este módulo")
        
        try:
            agent.curiosity.load_state_dict(checkpoint['curiosity_state'])
            print("  - Módulo curiosity cargado correctamente")
        except Exception as e:
            print(f"  - Error al cargar módulo curiosity: {e}")
            print("    Manteniendo inicialización aleatoria para este módulo")
            
    except Exception as e:
        print(f"Error durante la carga de módulos: {e}")
    
    # Verificar y corregir dimensiones
    print("Verificando y corrigiendo dimensiones entre módulos...")
    agent.check_module_dimensions()
    
    # Mostrar dimensiones actualizadas
    perception_output = agent.perception.output_shape
    prediction_input = agent.prediction.input_shape
    executive_input = agent.executive.input_shape
    
    print("Dimensiones después de la corrección:")
    print(f"  - perception_output: {perception_output}")
    print(f"  - prediction_input: {prediction_input}")
    print(f"  - executive_input: {executive_input}")
    
    # Restaurar otros datos del checkpoint
    agent.episodes_trained = checkpoint.get('episodes_trained', 0)
    agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
    agent.gamma = checkpoint.get('gamma', agent.gamma)
    
    # Guardar modelo corregido
    print(f"Guardando modelo corregido en: {output_path}")
    agent.save(output_path)
    
    print("¡Corrección completada!")
    return output_path


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Corrige inconsistencias dimensionales en modelos guardados.")
    parser.add_argument("model_path", help="Ruta al modelo guardado con inconsistencias.")
    parser.add_argument("--output", "-o", help="Ruta donde guardar el modelo corregido (opcional).")
    parser.add_argument("--analyze-only", "-a", action="store_true", 
                        help="Solo analizar el modelo sin corregir.")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        print(f"Analizando modelo: {args.model_path}")
        model_info = analyze_model(args.model_path)
        
        if not model_info:
            print("No se pudo analizar el modelo")
            return
        
        print(f"Forma de observación: {model_info['observation_shape']}")
        print(f"Tamaño de acción: {model_info['action_size']}")
        
        if model_info['inconsistencies']:
            print("Inconsistencias detectadas:")
            for inconsistency in model_info['inconsistencies']:
                print(f"  - {inconsistency}")
        else:
            print("No se detectaron inconsistencias dimensionales explícitas")
    else:
        fixed_model_path = fix_model_dimensions(args.model_path, args.output)
        if fixed_model_path:
            print(f"Modelo corregido guardado en: {fixed_model_path}")
        else:
            print("No se pudo corregir el modelo")


if __name__ == "__main__":
    main() 