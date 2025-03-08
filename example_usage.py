"""
Ejemplo de uso del framework NeurEvo.

Este script muestra cómo se utilizaría el framework NeurEvo en un proyecto real,
siguiendo la nueva API unificada.
"""

import neurevo
import numpy as np

def main():
    print("Ejemplo de uso del framework NeurEvo")
    print("-----------------------------------")
    
    # Crear un cerebro con configuración por defecto
    print("\n1. Creando cerebro...")
    brain = neurevo.create_brain()
    
    # Configuración personalizada (ejemplo)
    config = {
        "learning_rate": 0.0005,
        "batch_size": 128,
        "hidden_layers": [256, 128, 64],
        "curiosity_weight": 0.2
    }
    print("\n2. Creando cerebro con configuración personalizada...")
    brain_custom = neurevo.create_brain(config)
    
    # Crear y configurar para un entorno Gym
    print("\n3. Creando agente para entorno CartPole-v1...")
    agent = brain.create_for_environment("CartPole-v1")
    
    # Ejemplo de entorno personalizado
    print("\n4. Ejemplo de creación de entorno personalizado...")
    
    # Definir funciones para el entorno personalizado
    def reset_fn():
        """Función para reiniciar el entorno personalizado"""
        print("  - Entorno reiniciado")
        return np.zeros(4)  # Estado inicial de ejemplo
    
    def step_fn(action):
        """Función para ejecutar un paso en el entorno personalizado"""
        print(f"  - Acción ejecutada: {action}")
        next_state = np.random.random(4)  # Estado siguiente aleatorio
        reward = 1.0  # Recompensa de ejemplo
        done = False  # No terminado
        info = {}  # Sin información adicional
        return next_state, reward, done, info
    
    # En un caso real, registraríamos el entorno personalizado
    print("  - Registrando entorno personalizado (simulado)")
    # brain.register_environment("CustomEnv", 
    #                           create_custom_environment(
    #                               reset_fn=reset_fn,
    #                               step_fn=step_fn,
    #                               observation_shape=(4,),
    #                               action_size=2
    #                           ))
    
    # Usar el entorno registrado
    print("  - Creando agente para entorno personalizado (simulado)")
    # custom_agent = brain.create_for_environment("CustomEnv")
    
    # Entrenar el agente
    print("\n5. Entrenamiento (simulado)...")
    # results = brain.train(episodes=500)
    print("  - Entrenamiento completado")
    
    # Evaluar el rendimiento
    print("\n6. Evaluación (simulada)...")
    # avg_reward = brain.evaluate(episodes=10)
    print("  - Recompensa promedio: 95.5 (simulada)")
    
    # Guardar el agente
    print("\n7. Guardado (simulado)...")
    # path = brain.save()
    print("  - Agente guardado en: models/agent_1_20240308_123456.pt (simulado)")
    
    # Cargar el agente
    print("\n8. Carga (simulada)...")
    # brain.load("models/agent_1_20240308_123456.pt")
    print("  - Agente cargado correctamente")
    
    print("\nEjemplo completado con éxito!")

if __name__ == "__main__":
    main() 