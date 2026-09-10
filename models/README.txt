MODELO DE DETECCIÓN Y CLASIFICACIÓN DE MADUREZ DE PIÑAS
================================================================

Versión                  : 1.0
Experimento seleccionado : E3
Arquitectura             : YOLOv8m
Archivo del modelo       : modelo/detector_madurez_pina_yolov8m_v1.pt
Clases                   : inmadura, madura, sobremadura
Tamaño de entrada        : 640 px
Confianza predeterminada : 0.25
IoU/NMS predeterminado   : 0.7
Parámetros               : 25,858,057
Tamaño del modelo        : 49.64 MB

FINALIDAD
El modelo localiza piñas mediante cajas delimitadoras y clasifica cada
detección en inmadura, madura o sobremadura. No genera máscaras de
segmentación por píxel.

SELECCIÓN METODOLÓGICA
E3-YOLOv8m fue seleccionado antes de abrir el conjunto test, utilizando
mAP@0.50:0.95 sobre validación como criterio principal. La evaluación en
test se utilizó posteriormente para estudiar la generalización y no para
reajustar el modelo.

USO RÁPIDO
1. Instalar dependencias:
   pip install -r requirements.txt

2. Colocar una imagen como imagen_pina.jpg en la raíz del paquete.

3. Ejecutar:
   python ejemplo_inferencia/inferencia_basica.py

ESTRUCTURA
- modelo/: pesos entrenados en formato PyTorch/Ultralytics.
- configuracion/: clases y parámetros de inferencia.
- metricas/: resultados finales disponibles.
- ejemplo_inferencia/: script mínimo de uso.

NOTA
Para reproducir los resultados reportados deben mantenerse el tamaño de
imagen y los umbrales documentados. Cambiar estos valores en el aplicativo
es posible, pero debe registrarse como una decisión de implementación.
