from pathlib import Path
from ultralytics import YOLO

# Resolución automática de la ruta del modelo
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

CANDIDATES = [
    PROJECT_ROOT / "models" / "detector_madurez_pina_yolov8m_v1.pt",
    CURRENT_DIR.parent / "detector_madurez_pina_yolov8m_v1.pt",
    Path("models/detector_madurez_pina_yolov8m_v1.pt"),
    Path("modelo/detector_madurez_pina_yolov8m_v1.pt"),
    Path("detector_madurez_pina_yolov8m_v1.pt"),
]

MODEL_PATH = None
for candidate in CANDIDATES:
    if candidate.is_file():
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "No se encontró 'detector_madurez_pina_yolov8m_v1.pt'. "
        "Verifica que el archivo esté en la carpeta 'models/'."
    )

print(f"Cargando modelo desde: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

IMAGE_PATH = Path("imagen_pina.jpg")

if not IMAGE_PATH.is_file():
    print(f"Nota: Coloca una imagen en '{IMAGE_PATH.resolve()}' para ejecutar la prueba completa.")
else:
    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        iou=0.70,
        save=True,
        project="resultados",
        name="inferencia",
        exist_ok=True,
    )

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()

            print({
                "clase": model.names[class_id],
                "confianza": round(confidence, 4),
                "bbox_xyxy": [round(value, 2) for value in coordinates],
            })

    print("Resultados visuales guardados en resultados/inferencia")
