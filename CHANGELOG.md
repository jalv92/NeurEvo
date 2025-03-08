# Changelog

## [1.1.0] - Fecha: 2024-03-08

### Restructuración del Framework

#### Nueva Estructura de Paquetes
- Implementada estructura completa de paquetes con archivos `__init__.py` en todos los directorios
- Creado archivo `setup.py` para permitir instalación en modo desarrollo
- Reorganizados módulos para mejorar la estructura y la organización del código

#### Nueva Interfaz Unificada
- Implementada clase `BrainInterface` en `neurevo/brain.py` como punto de entrada principal
- Creada función `create_brain()` para inicialización simplificada
- Mejorada la API para facilitar la interacción con agentes y entornos

#### Sistema de Registro de Componentes
- Creado sistema de registro en `neurevo/utils/registry.py` para componentes dinámicos
- Implementados mecanismos para registrar y recuperar componentes de forma flexible
- Añadido soporte para registro con decoradores y categorías de componentes

#### Adaptadores de Entorno
- Creado nuevo subsistema de adaptadores de entorno en `neurevo/environments/`
- Implementada clase base abstracta `EnvironmentAdapter` para todos los adaptadores
- Creados adaptadores para Gym/Gymnasium y para entornos personalizados
- Añadido sistema de registro automático de entornos integrados

#### Solución de Problemas de Importación
- Corregidos problemas de importación con Pylance/VS Code
- Implementada estructura de importación robusta para evitar errores circulares
- Añadidos stubs temporales para permitir importaciones parciales durante desarrollo

#### Documentación Interna
- Añadidos docstrings completos para todas las nuevas clases y funciones
- Actualizados comentarios existentes para reflejar la nueva estructura
- Mejorada la documentación de la API pública y ejemplos de uso

### Recomendaciones de Uso
- Usar `pip install -e .` para instalar el paquete en modo desarrollo
- Actualizar importaciones para utilizar la nueva estructura de paquetes
- Migrar gradualmente a la nueva API unificada (`BrainInterface`)
- Utilizar los adaptadores de entorno para integrar nuevos entornos

---

## [1.0.1] - Fecha: 2023-XX-XX

### Correcciones de Errores

#### Corrección de Inconsistencias Dimensionales
- Solucionado problema de inconsistencias en dimensiones entre módulos neuronales cuando crecen dinámicamente
- Implementado sistema robusto de propagación de cambios dimensionales entre capas conectadas
- Mejorado el manejo de errores al desempaquetar lotes durante el entrenamiento

### Nuevas Características

#### Sistema de Notificación de Cambios Dimensionales
- Añadido sistema de callbacks a la clase `DynamicLayer` para notificar cambios de dimensiones
- Implementado registro de oyentes (listeners) para reaccionar a cambios dimensionales
- Creados métodos para actualizar dinámicamente dimensiones de entrada y salida

#### Mecanismo de Adaptación entre Módulos
- Implementada comunicación entre módulos para mantener consistencia dimensional
- Añadido sistema para conectar módulos y propagar cambios automáticamente
- Creado mecanismo para adaptar módulos cuando se detectan inconsistencias

#### Utilitarios para Corrección de Modelos
- Creado script `FixModelDimensions.py` para analizar y corregir modelos guardados
- Implementado script `ResetAndTrain.py` para reiniciar entrenamiento desde cero
- Mejorado sistema de guardado y carga para detectar y corregir inconsistencias

### Mejoras Técnicas

#### Mejoras en `DynamicLayer`
- Implementados métodos `register_dimension_listener` y `notify_dimension_change`
- Añadidos métodos `update_input_size` y `update_output_size` para redimensionar capas
- Modificados métodos `adapt_connectivity` y `grow_connections` para notificar cambios

#### Mejoras en `BaseModule`
- Añadido sistema de conexión entre módulos con `connect_to_input` y `connect_to_output`
- Implementado método `dimension_changed` para reaccionar a cambios en módulos conectados
- Creado método `update_input_shape` para propagar cambios a través de la red

#### Mejoras en `NeurEvoAgent`
- Rediseñado método `train_step` con robusto manejo de errores dimensionales
- Implementado sistema de recuperación ante errores de dimensionalidad
- Añadido método `check_module_dimensions` para verificación periódica
- Mejorado sistema de guardado/carga para incluir información dimensional

#### Mejoras en `PerceptionModule`
- Implementado método `adapt_to_input_shape` para adaptación dinámica
- Añadida lógica para reconectar capas cuando cambian dimensiones
- Mejorado manejo de capas convolucionales para adaptarse a cambios

### Recomendaciones de Uso
- Ejecutar regularmente la verificación de dimensiones con `check_module_dimensions`
- Para modelos con problemas, usar primero `FixModelDimensions.py`
- Si la corrección falla, usar `ResetAndTrain.py` para reiniciar desde cero
- Guardar modelos frecuentemente durante el entrenamiento

### Correcciones Adicionales
- Mejorado manejo de errores en funciones críticas
- Añadido sistema de logs más detallados
- Optimizadas operaciones de tensores para mayor eficiencia

---

## Notas para Desarrolladores

### Elementos Clave Implementados

1. **Propagación de Cambios Dimensionales**:
   - Cuando una capa cambia de tamaño, notifica a las capas conectadas
   - Las capas receptoras adaptan sus dimensiones para mantener consistencia

2. **Corrección Automática**:
   - Durante el entrenamiento, se detectan y corrigen inconsistencias
   - Al cargar modelos, se verifican las dimensiones y se adaptan si es necesario

3. **Recuperación de Modelos**:
   - `FixModelDimensions.py` analiza un modelo y crea uno nuevo con dimensiones correctas
   - `ResetAndTrain.py` permite entrenar desde cero conservando la configuración 