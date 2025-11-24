### Proyecto II: Cruces inteligentes con EDGE AI embebido

Este proyecto desarrolla un nodo embebido capaz de analizar video en tiempo real para detectar y clasificar vehículos y fauna en entornos urbanos. El sistema corre en una Raspberry Pi 4 de 64 bits utilizando una imagen Linux personalizada generada con Yocto Project. El objetivo es llevar la inteligencia al borde: procesar video localmente para reducir latencia, evitar dependencia de la nube y permitir una operación continua con bajo consumo energético.

### Fundamentos del sistema
1. ##### Visión por computadora con OpenCV

La imagen generada con Yocto incluye OpenCV integrado de forma nativa, lo que habilita la captura y el procesamiento de video directamente en el dispositivo. Este módulo maneja tareas iniciales como lectura de frames, corrección de color, normalización y detección preliminar de objetos. Es la base de todo el pipeline de inferencia.

2. ##### Inferencia con TensorFlow Lite y librerías instaladas vía pip

El modelo de IA corre con TensorFlow Lite, instalado mediante pip junto con el resto de las dependencias de Python. Esto mantiene la imagen del sistema ligera y facilita la actualización de librerías sin recompilar todo el sistema con Yocto. El flujo típico es:
video preprocesado → modelo TFLite → clasificación en tiempo real.

3. ##### Eficiencia y sistema operativo minimalista

La imagen de Yocto contiene únicamente lo necesario para operar:

- soporte para video,

- herramientas esenciales de Python,

- control del hardware,

- y conectividad inalámbrica.

La imagen también incluye los firmwares necesarios para habilitar el WiFi de la Raspberry Pi 4, garantizando conectividad desde el primer arranque sin configuraciones adicionales.

### Arquitectura del nodo

El sistema sigue un flujo simple y eficiente:

cámara → OpenCV (preprocesamiento) → modelo TFLite (clasificación) → salida


### Requisitos de hardware

- Raspberry Pi 4 (4 GB recomendado)

- Cámara compatible con CSI o USB

- MicroSD de 32 GB o más

- Fuente de alimentación estable

- Conectividad WiFi (ya habilitada en la imagen)

### Construcción y despliegue

La imagen del sistema se genera con Yocto Project, utilizando una configuración personalizada en local.conf.

OpenCV se incluye como parte del build.
Las librerías avanzadas de IA se instalan directamente en el dispositivo mediante pip, permitiendo ajustes y pruebas rápidas.

La combinación Yocto + pip permite mantener un sistema base estable y un entorno de IA flexible.

### Mejoras futuras



### Resumen

El dispositivo actúa como un nodo inteligente capaz de capturar, procesar y clasificar video en tiempo real sin depender de servidores externos. Este enfoque es ideal para sistemas de tráfico, monitoreo ambiental y aplicaciones que requieren respuesta inmediata y autonomía.

El proyecto demuestra cómo combinar sistemas embebidos, Edge AI y un Linux personalizado con Yocto para construir un observador inteligente capaz de operar de forma independiente en escenarios reales.