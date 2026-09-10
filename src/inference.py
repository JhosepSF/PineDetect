"""Módulo de inferencia y extracción de métricas para PineDetect."""

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from src.config import (
    CLASS_NAMES,
    CLASS_DISPLAY_NAMES,
    DEFAULT_IMGSZ,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Representa una piña detectada con sus coordenadas y clasificación de madurez."""
    index: int
    class_id: int
    class_name: str
    class_display_name: str
    confidence: float
    confidence_percentage: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la detección a diccionario estructurado con tipos nativos de Python."""
        return {
            "index": int(self.index),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "class_display_name": str(self.class_display_name),
            "confidence": float(self.confidence),
            "confidence_percentage": float(self.confidence_percentage),
            "x_min": float(round(self.x_min, 2)),
            "y_min": float(round(self.y_min, 2)),
            "x_max": float(round(self.x_max, 2)),
            "y_max": float(round(self.y_max, 2)),
            "width": float(round(self.width, 2)),
            "height": float(round(self.height, 2)),
        }

    def to_table_row(self) -> Dict[str, Any]:
        """Formatea la detección para presentación en tabla de Streamlit, Django y CSV."""
        return {
            "N° Detección": int(self.index),
            "Estado de Madurez": str(self.class_display_name),
            "Confianza (%)": f"{float(self.confidence_percentage):.2f}%",
            "X Mínimo": float(round(self.x_min, 2)),
            "Y Mínimo": float(round(self.y_min, 2)),
            "X Máximo": float(round(self.x_max, 2)),
            "Y Máximo": float(round(self.y_max, 2)),
            "Ancho (px)": float(round(self.width, 2)),
            "Alto (px)": float(round(self.height, 2)),
        }


@dataclass
class InferenceSummary:
    """Resumen estadístico de la inferencia."""
    total_detections: int
    average_confidence: Optional[float]
    inference_time_ms: float
    predominant_class: str
    counts_by_class: Dict[str, int]
    percentages_by_class: Dict[str, float]


def count_detections_by_class(detections: List[Detection]) -> Dict[str, int]:
    """Calcula el conteo de piñas detectadas para cada estado de madurez."""
    counts = {
        "inmadura": 0,
        "madura": 0,
        "sobremadura": 0
    }
    for det in detections:
        key = det.class_name.lower()
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = counts.get(key, 0) + 1
    return counts


def determine_predominant_class(counts_by_class: Dict[str, int]) -> str:
    """Determina la clase de madurez predominante.

    Si hay un empate entre los conteos máximos o el total es cero,
    retorna 'Sin clase predominante'.
    """
    total = sum(counts_by_class.values())
    if total == 0:
        return "Sin clase predominante"

    max_count = max(counts_by_class.values())
    if max_count == 0:
        return "Sin clase predominante"

    # Encontrar clases con el conteo máximo
    top_classes = [
        cls_name for cls_name, cnt in counts_by_class.items() if cnt == max_count
    ]

    # Si hay empate entre más de una clase
    if len(top_classes) > 1:
        return "Sin clase predominante"

    top_name = top_classes[0]
    return top_name.capitalize()


def build_inference_summary(
    detections: List[Detection],
    inference_time_ms: float
) -> InferenceSummary:
    """Genera el resumen estadístico a partir de la lista de detecciones."""
    total = len(detections)
    counts = count_detections_by_class(detections)
    
    if total > 0:
        avg_conf = sum(d.confidence for d in detections) / total
        percentages = {
            cls: round((cnt / total) * 100, 2) for cls, cnt in counts.items()
        }
    else:
        avg_conf = None
        percentages = {cls: 0.0 for cls in counts}

    predominant = determine_predominant_class(counts)

    return InferenceSummary(
        total_detections=total,
        average_confidence=avg_conf,
        inference_time_ms=round(inference_time_ms, 2),
        predominant_class=predominant,
        counts_by_class=counts,
        percentages_by_class=percentages
    )


def run_pineapple_detection(
    model: Any,
    image_input: Image.Image | np.ndarray,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU
) -> Tuple[List[Detection], InferenceSummary]:
    """Ejecuta la predicción con YOLOv8 sobre una imagen y procesa los resultados.

    Args:
        model: Objeto YOLO cargado de Ultralytics.
        image_input: Imagen en formato PIL Image o NumPy ndarray (RGB).
        imgsz: Tamaño de entrada para el modelo (predeterminado 640).
        conf: Umbral mínimo de confianza (0.10 a 0.90).
        iou: Umbral de IoU para Non-Maximum Suppression (0.30 a 0.90).

    Returns:
        Tupla con (lista de detecciones, resumen de inferencia).
    """
    if isinstance(image_input, Image.Image):
        # Convertir a array NumPy RGB
        image_array = np.array(image_input.convert("RGB"))
    else:
        image_array = image_input

    logger.info(
        "Iniciando inferencia YOLOv8: imgsz=%d, conf=%.2f, iou=%.2f",
        imgsz, conf, iou
    )

    start_time = time.perf_counter()
    results = model.predict(
        source=image_array,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False
    )
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000.0

    detections: List[Detection] = []
    
    if results and len(results) > 0:
        first_result = results[0]
        boxes = first_result.boxes

        if boxes is not None and len(boxes) > 0:
            # Extraer coordenadas, clases y confianzas a CPU
            xyxy_coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            for i, (coords, cls_id_raw, score) in enumerate(zip(xyxy_coords, class_ids, confidences), start=1):
                cls_id = int(cls_id_raw)
                x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                model_names = getattr(model, "names", {})
                if isinstance(model_names, dict) and cls_id in model_names:
                    cls_name = str(model_names[cls_id]).lower()
                else:
                    cls_name = CLASS_NAMES.get(cls_id, f"clase_{cls_id}").lower()

                disp_name = CLASS_DISPLAY_NAMES.get(cls_id, cls_name.capitalize())
                score_flt = float(score)

                width = float(max(0.0, x2 - x1))
                height = float(max(0.0, y2 - y1))

                detection = Detection(
                    index=int(i),
                    class_id=cls_id,
                    class_name=cls_name,
                    class_display_name=disp_name,
                    confidence=score_flt,
                    confidence_percentage=float(round(score_flt * 100, 2)),
                    x_min=x1,
                    y_min=y1,
                    x_max=x2,
                    y_max=y2,
                    width=width,
                    height=height
                )
                detections.append(detection)

    summary = build_inference_summary(detections, inference_time_ms)
    logger.info(
        "Inferencia completada en %.2f ms con %d piñas detectadas",
        inference_time_ms, len(detections)
    )

    return detections, summary
