"""Pruebas unitarias para el módulo de exportación de archivos (src.exporters)."""

import json
from PIL import Image
import pytest

from src.exporters import (
    export_image_to_png_bytes,
    export_detections_to_csv_bytes,
    export_summary_to_json_bytes,
)
from src.inference import Detection, InferenceSummary, build_inference_summary
from tests.test_inference_helpers import sample_detection


def test_export_image_to_png_bytes():
    """Valida la generación de bytes PNG válidos."""
    img = Image.new("RGB", (50, 50), color=(100, 150, 200))
    png_bytes = export_image_to_png_bytes(img)
    
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    # Cabecera mágica de archivo PNG
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_export_detections_to_csv_with_bom():
    """Valida que el CSV incluya cabeceras en español y codificación UTF-8 con BOM."""
    detections = [
        sample_detection(1, 1, "madura", 0.892, 10.0, 20.0, 100.0, 200.0),
        sample_detection(2, 0, "inmadura", 0.941, 15.0, 25.0, 115.0, 225.0)
    ]
    csv_bytes = export_detections_to_csv_bytes(detections)
    
    assert isinstance(csv_bytes, bytes)
    # Verificar firma UTF-8 BOM
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    
    # Decodificar texto
    csv_text = csv_bytes.decode("utf-8-sig")
    assert "N° Detección" in csv_text
    assert "Estado de Madurez" in csv_text
    assert "Madura" in csv_text
    assert "Inmadura" in csv_text
    assert "89.20%" in csv_text


def test_export_detections_to_csv_empty():
    """Valida que el CSV con cero detecciones genere las cabeceras correctamente sin fallar."""
    csv_bytes = export_detections_to_csv_bytes([])
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    
    csv_text = csv_bytes.decode("utf-8-sig")
    assert "N° Detección" in csv_text
    assert "Estado de Madurez" in csv_text


def test_export_summary_to_json():
    """Valida que el resumen en JSON contenga toda la estructura requerida."""
    detections = [
        sample_detection(1, 1, "madura", 0.887)
    ]
    summary = build_inference_summary(detections, inference_time_ms=135.2)
    thresholds = {"conf": 0.25, "iou": 0.70, "imgsz": 640}

    json_bytes = export_summary_to_json_bytes(
        summary=summary,
        detections=detections,
        thresholds=thresholds,
        model_name="detector_madurez_pina_yolov8m_v1.pt"
    )

    assert isinstance(json_bytes, bytes)
    data = json.loads(json_bytes.decode("utf-8"))

    assert data["aplicativo"] == "PineDetect"
    assert "fecha_generacion" in data
    assert data["modelo"]["nombre"] == "detector_madurez_pina_yolov8m_v1.pt"
    assert data["parametros_utilizados"]["confianza_minima"] == 0.25
    assert data["parametros_utilizados"]["iou_nms"] == 0.70
    assert data["resumen_inferencia"]["total_pinas_detectadas"] == 1
    assert data["resumen_inferencia"]["clase_predominante"] == "Madura"
    assert len(data["detalle_detecciones"]) == 1
    assert data["detalle_detecciones"][0]["class_name"] == "madura"
