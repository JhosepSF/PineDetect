# Directorio de Modelos - PineDetect

Este directorio está destinado a almacenar el modelo entrenado de YOLOv8 para la detección y clasificación del estado de madurez de piñas.

## Nombre del Archivo Requerido
```
detector_madurez_pina_yolov8m_v1.pt
```

## Ubicación Exacta
```
PineDetect/
└── models/
    └── detector_madurez_pina_yolov8m_v1.pt
```

## Clases Detectadas por el Modelo
- `0`: **inmadura** (Color institucional: Verde RGB `(44, 160, 44)`)
- `1`: **madura** (Color institucional: Amarillo/Naranja RGB `(242, 177, 52)`)
- `2`: **sobremadura** (Color institucional: Rojo RGB `(214, 39, 40)`)

## Carga del Modelo
La aplicación carga automáticamente este modelo mediante:
```python
from ultralytics import YOLO
model = YOLO("models/detector_madurez_pina_yolov8m_v1.pt")
```
Si el modelo no se encuentra en esta ubicación, la aplicación mostrará una alerta indicativa en la barra lateral sin interrumpir la ejecución.
