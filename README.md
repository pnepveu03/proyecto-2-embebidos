# Proyecto II – Cruces Inteligentes con Edge AI Embebido


---

## Descripción General

Este proyecto implementa un nodo de visión computacional embebido capaz de detectar y clasificar vehículos y fauna en tiempo real directamente en el borde. El sistema se ejecuta sobre una **Raspberry Pi 4** con una imagen Linux personalizada generada mediante **Yocto Project**, optimizada para inferencia local con **TensorFlow Lite** y preprocesamiento de video con **OpenCV** y **Imageio**.

**Objetivo:** realizar análisis continuo de video sin depender de la nube, con baja latencia y utilizando solo los recursos estrictamente necesarios.

---

## Arquitectura del Sistema

```mermaid
graph TB
    subgraph Hardware["Capa de Hardware"]
        CAM[Cámara USB/CSI]
        RPI[Raspberry Pi 4]
        SD[MicroSD 32GB+]
    end
    
    subgraph OS["Sistema Operativo Yocto"]
        KERNEL[Linux Kernel]
        SYSTEMD[systemd]
        NETWORK[Networking/WiFi]
    end
    
    subgraph Libraries["Bibliotecas del Sistema"]
        OPENCV[OpenCV Nativo]
        PYTHON[Python 3 + pip]
        DEPS[Dependencias Base]
    end
    
    subgraph Application["Capa de Aplicación"]
        CAPTURE[Captura de Video]
        PREPROC[Preprocesamiento]
        TFLITE[TensorFlow Lite]
        DETECT[Motor de Detección]
        IOT[Integración IoT]
    end
    
    subgraph Output["Salidas"]
        VIDEO[Video Anotado]
        CLOUD[ThingSpeak/MQTT]
    end
    
    CAM --> CAPTURE
    RPI --> OS
    SD --> OS
    OS --> Libraries
    Libraries --> Application
    
    CAPTURE --> PREPROC
    PREPROC --> TFLITE
    TFLITE --> DETECT
    DETECT --> VIDEO
    DETECT --> IOT
    IOT --> CLOUD
    
    classDef hwClass fill:#e74c3c,stroke:#c0392b,color:#fff
    classDef osClass fill:#3498db,stroke:#2980b9,color:#fff
    classDef libClass fill:#2ecc71,stroke:#27ae60,color:#fff
    classDef appClass fill:#9b59b6,stroke:#8e44ad,color:#fff
    classDef outClass fill:#f39c12,stroke:#e67e22,color:#fff
    
    class CAM,RPI,SD hwClass
    class KERNEL,SYSTEMD,NETWORK osClass
    class OPENCV,PYTHON,DEPS libClass
    class CAPTURE,PREPROC,TFLITE,DETECT,IOT appClass
    class VIDEO,LOGS,STATS,CLOUD outClass
```

---

## Pipeline de Procesamiento

```mermaid
flowchart LR
    A[Cámara, Archivo de video] --> B[Captura OpenCV, Captura Imageio]
    B --> C[Preprocesamiento]
    C --> D{Tipo de entrada}
    D -->|uint8| E[Normalización 0-255]
    D -->|float32| F[Normalización -1 a 1]
    E --> G[Inferencia TFLite]
    F --> G
    G --> H[Post-procesamiento]
    H --> I[Filtrado por confianza]
    I --> J[Filtrado por clase]
    J --> L[Salidas]
    L --> M[Video con overlay]
    L --> P[ThingSpeak/IoT]
    
    style A fill:#ff6b6b
    style G fill:#45b7d1
    style L fill:#96ceb4
    style P fill:#ffd93d
```

---

## Fundamentos del Sistema

### 1. Procesamiento de Video con OpenCV

La imagen generada con Yocto integra OpenCV de forma nativa. Este componente se encarga de:

- Captura eficiente de video desde una cámara USB/CSI
- Preprocesamiento: redimensión, normalización y corrección de color
- Entrega de frames listos para inferencia sin overhead gráfico

OpenCV es la primera etapa del pipeline y determina el rendimiento global del sistema.

### 2. Inferencia Ligera con TensorFlow Lite

La Raspberry Pi ejecuta modelos optimizados (por ejemplo, MobileNet o EfficientDet Lite) usando TensorFlow Lite instalado vía pip, lo que evita recompilar TensorFlow desde Yocto.

**Flujo de inferencia:**

```mermaid
sequenceDiagram
    participant C as Cámara
    participant O as OpenCV
    participant P as Preprocesador
    participant T as TFLite
    participant D as Detector
    participant S as Salida
    
    C->>O: Frame crudo (BGR)
    O->>O: Conversión RGB
    O->>P: Frame RGB
    P->>P: Resize a tamaño modelo
    P->>P: Normalización
    P->>T: Tensor de entrada
    T->>T: Inferencia (forward pass)
    T->>D: Boxes, Classes, Scores
    D->>D: Filtrado por confianza
    D->>D: Filtrado por clase permitida
    D->>D: Non-Maximum Suppression
    D->>S: Detecciones finales
    S->>S: Dibujar bounding boxes
    S->>S: Guardar en video
    S->>S: Registrar en CSV
    S->>S: Subir a ThingSpeak
```

### 3. Sistema Operativo Mínimo con Yocto

La imagen Yocto está diseñada para incluir solo lo necesario:

**Componentes nativos compilados:**
- OpenCV 4.x con optimizaciones ARM
- Python 3.x + pip + setuptools
- Soporte completo para cámara USB (V4L2)
- GStreamer para procesamiento multimedia
- Servicios administrados mediante systemd
- WiFi y Bluetooth (firmware bcm43430/43455)
- wpa_supplicant para redes
- GPIO y buses I2C

**Dependencias instaladas vía pip:**
- TensorFlow Lite runtime
- NumPy
- PyYAML
- imageio + ffmpeg

---

## Capas de Yocto Utilizadas

```mermaid
graph LR
    subgraph Base["Poky Base"]
        A[meta]
        B[meta-poky]
        C[meta-yocto-bsp]
    end
    
    subgraph OpenEmbedded["meta-openembedded"]
        D[meta-oe]
        E[meta-python]
        F[meta-networking]
    end
    
    subgraph BSP["Board Support"]
        G[meta-raspberrypi]
    end
    
    
    style A fill:#3498db
    style B fill:#3498db
    style C fill:#3498db
    style D fill:#2ecc71
    style E fill:#2ecc71
    style F fill:#2ecc71
    style G fill:#e74c3c
```

---

## Requisitos de Hardware

| Componente | Especificación |
|-----------|---------------|
| **Placa** | Raspberry Pi 4 (4 GB RAM recomendado) |
| **Cámara** | Cámara USB o CSI compatible |
| **Almacenamiento** | MicroSD 32 GB o más |
| **Alimentación** | Fuente 5 V / 3 A estable |
| **Conectividad** | WiFi (incluida en la imagen) |

---

## Flujo de Detección y Filtrado

```mermaid
flowchart TD
    START[Inicio Detección] --> INFER[Ejecutar Inferencia]
    INFER --> GET[Obtener Boxes, Classes, Scores]
    GET --> LOOP{Para cada detección}
    
    LOOP --> CHECK_CONF{Score > umbral?}
    CHECK_CONF -->|No| DISCARD[Descartar]
    CHECK_CONF -->|Sí| CHECK_CLASS{Clase permitida?}
    
    CHECK_CLASS -->|No| DISCARD
    CHECK_CLASS -->|Sí| VALID[Detección válida]
    
    VALID --> COORDS[Calcular coordenadas]
    COORDS --> CLAMP[Ajustar límites]
    CLAMP --> STORE[Almacenar resultado]
    
    STORE --> LOOP
    DISCARD --> LOOP
    
    LOOP -->|Fin| NMS[Aplicar NMS]
    NMS --> DRAW[Dibujar bounding boxes]
    DRAW --> SAVE[Guardar frame]
    SAVE --> IOT{IoT habilitado?}
    
    IOT -->|Sí| UPLOAD[Subir a ThingSpeak]
    IOT -->|No| END[Fin]
    UPLOAD --> END
    
    style START fill:#2ecc71
    style INFER fill:#3498db
    style VALID fill:#27ae60
    style DISCARD fill:#e74c3c
    style UPLOAD fill:#f39c12
    style END fill:#95a5a6
```

---

## Clases Detectables

El sistema filtra detecciones basándose en clases relevantes para entornos urbanos:

```mermaid
graph TB
    subgraph Vehiculos["Vehículos"]
        V1[0: Persona]
        V2[1: Bicicleta]
        V3[2: Carro]
        V4[3: Motocicleta]
        V5[5: Autobús]
        V6[6: Tren]
        V7[7: Camión]
    end
    
    subgraph Infraestructura["Infraestructura"]
        I1[9: Semáforo]
        I2[10: Hidrante]
        I3[11: Señal de Stop]
        I4[12: Parquímetro]
        I5[13: Banco]
        I6[14: Pájaro]
    end
    
    subgraph Fauna["Fauna Urbana"]
        F1[15: Gato]
        F2[16: Perro]
        F3[17: Caballo]
    end
    
    subgraph Accesorios["Accesorios Peatonales"]
        A1[26: Mochila]
        A2[27: Paraguas]
        A3[30: Bolso]
    end
    
    style Vehiculos fill:#3498db,color:#fff
    style Infraestructura fill:#2ecc71,color:#fff
    style Fauna fill:#e74c3c,color:#fff
    style Accesorios fill:#f39c12,color:#fff
```
---

## Thingspeak

La plataforma ThingSpeak recibe en tiempo real los datos enviados por el sistema embebido. Cada detección procesada por el modelo se publica como un valor numérico que luego se visualiza en forma de gráficos históricos. Esto permite verificar que la comunicación IoT funciona correctamente, monitorear el comportamiento del sistema a lo largo del tiempo y validar que las detecciones generadas en el dispositivo llegan de forma confiable a la nube.

![ThingSpeak Dashboard](Figuras/thingspeak.jpeg)


## Posibles casos de uso

### 1. Monitoreo de Tráfico Urbano

```mermaid
graph TB
    A[Cámara en Intersección] --> B[Detección Vehículos]
    B --> C[Conteo por Clase]
    C --> D[Análisis de Flujo]
    D --> E[Datos a Dashboard]
    D --> F[Optimización Semáforos]
    
    style A fill:#3498db
    style E fill:#2ecc71
    style F fill:#e67e22
```

### 2. Detección de Fauna en Cruces

```mermaid
graph TB
    A[Cámara en Zona Rural] --> B[Detección Animales]
    B --> C[Clasificación Especie]
    C --> D[Alerta Temprana]
    D --> E[Señalización LED]
    D --> F[Notificación Conductores]
    
    style A fill:#3498db
    style D fill:#e74c3c
    style E fill:#f39c12
```

### 3. Análisis de Seguridad Peatonal

```mermaid
graph TB
    A[Cámara Paso Peatonal] --> B[Detección Personas]
    B --> C[Tracking Movimiento]
    C --> D[Detección Comportamiento]
    D --> E[Estadísticas Uso]
    D --> F[Alertas de Riesgo]
    
    style A fill:#3498db
    style E fill:#2ecc71
    style F fill:#e74c3c
```

---

## Mejoras Futuras

```mermaid
mindmap
  root((Evolución del<br/>Sistema))
    Tracking
      DeepSORT
      ByteTrack
      Trayectorias
    Comunicación
      MQTT Broker
      CoAP Lightweight
      Multi-nodo
    Hardware
      Coral TPU
      INT8 Quantization
      NEON SIMD
    Integración
      GPIO Semáforos
      Protocolos Industriales
      Modbus RTU
    Analítica
      Dashboard Web
      Time Series DB
      Predicción ML
```

---

