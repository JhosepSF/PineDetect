"""Módulo de exportación de resultados en formatos PNG, CSV y JSON para PineDetect."""

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from PIL import Image

from src.config import APP_NAME, MODEL_FILENAME
from src.inference import Detection, InferenceSummary

logger = logging.getLogger(__name__)


def export_image_to_png_bytes(image: Image.Image) -> bytes:
    """Convierte una imagen PIL a bytes en formato PNG para descarga directa."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def export_detections_to_csv_bytes(detections: List[Detection]) -> bytes:
    """Genera un archivo CSV con codificación UTF-8 con BOM (utf-8-sig) para Excel.

    Args:
        detections: Lista de objetos Detection.

    Returns:
        Bytes del archivo CSV con BOM incluido.
    """
    column_names = [
        "N° Detección",
        "Estado de Madurez",
        "Confianza (%)",
        "X Mínimo",
        "Y Mínimo",
        "X Máximo",
        "Y Máximo",
        "Ancho (px)",
        "Alto (px)"
    ]

    if detections:
        rows = [det.to_table_row() for det in detections]
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=column_names)

    csv_buffer = io.StringIO()
    # Escribir dataframe a string
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    
    # Codificar con utf-8-sig (BOM) para compatibilidad directa con Excel en español
    return csv_string.encode("utf-8-sig")


def export_summary_to_json_bytes(
    summary: InferenceSummary,
    detections: List[Detection],
    thresholds: Dict[str, Any],
    model_name: str = MODEL_FILENAME,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> bytes:
    """Genera un archivo JSON estructurado con metadatos, parámetros y detalle de predicciones.

    Args:
        summary: Resumen estadístico de la inferencia.
        detections: Lista de detecciones extraídas.
        thresholds: Diccionario con umbrales utilizados (conf, iou, imgsz).
        model_name: Nombre del modelo utilizado.
        additional_metadata: Metadatos adicionales opcionales.

    Returns:
        Bytes con codificación UTF-8 del archivo JSON formateado.
    """
    avg_conf_pct = (
        round(summary.average_confidence * 100, 2)
        if summary.average_confidence is not None
        else None
    )

    data: Dict[str, Any] = {
        "aplicativo": APP_NAME,
        "fecha_generacion": datetime.now(timezone.utc).isoformat(),
        "modelo": {
            "nombre": model_name,
            "arquitectura": "YOLOv8m",
            "tipo_tarea": "Detección y clasificación de madurez"
        },
        "parametros_utilizados": {
            "confianza_minima": thresholds.get("conf", 0.25),
            "iou_nms": thresholds.get("iou", 0.70),
            "tamano_entrada_px": thresholds.get("imgsz", 640)
        },
        "resumen_inferencia": {
            "total_pinas_detectadas": summary.total_detections,
            "tiempo_inferencia_ms": summary.inference_time_ms,
            "confianza_promedio_porcentaje": avg_conf_pct,
            "clase_predominante": summary.predominant_class,
            "conteo_por_clase": summary.counts_by_class,
            "porcentajes_por_clase": summary.percentages_by_class
        },
        "detalle_detecciones": [det.to_dict() for det in detections]
    }

    if additional_metadata:
        data["metadatos_adicionales"] = additional_metadata

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    return json_str.encode("utf-8")
