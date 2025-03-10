"""
Script de prueba para verificar las importaciones de neurevo.

Este script intenta importar los componentes principales de neurevo
para asegurar que el paquete está correctamente instalado y puede 
ser utilizado desde cualquier proyecto.
"""

print("Iniciando prueba de importaciones de neurevo...")

# Intentar importar el paquete principal
try:
    import neurevo
    print("✅ Importación básica de neurevo exitosa")
except ImportError as e:
    print(f"❌ Error al importar neurevo: {e}")
    exit(1)

# Verificar la versión
print(f"📦 Versión de neurevo: {neurevo.__version__}")

# Intentar importar componentes principales
try:
    # Usar las clases temporales definidas en __init__.py
    brain = neurevo.create_brain()
    print("✅ Creación de BrainInterface exitosa")
    
    # Probar método de BrainInterface
    brain.create_for_environment("CartPole-v1")
    print("✅ Método create_for_environment funciona")
except Exception as e:
    print(f"❌ Error al usar BrainInterface: {e}")

# Intentar importar componentes del núcleo
try:
    from neurevo.core import NeurEvoAgent, BaseEnvironment
    print("✅ Importación de componentes del núcleo exitosa")
    
    # Probar NeurEvoAgent
    agent = NeurEvoAgent()
    print("✅ Creación de NeurEvoAgent exitosa")
except ImportError as e:
    print(f"❌ Error al importar componentes del núcleo: {e}")

print("\nResumen de prueba de importaciones completado.") 