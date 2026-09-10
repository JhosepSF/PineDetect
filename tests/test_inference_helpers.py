"""Pruebas unitarias para funciones auxiliares de inferencia y métricas (src.inference)."""

import pytest
from src.inference import (
    Detection,
    count_detections_by_class,
    determine_predominant_class,
    build_inference_summary,
)


def sample_detection(
    idx: int,
    cls_id: int,
    cls_name: str,
    conf: float,
    x1: float = 10.555,
    y1: float = 20.333,
    x2: float = 110.888,
    y2: float = 220.444
) -> Detection:
    """Helper para crear una detección de prueba."""
    width = x2 - x1
    height = y2 - y1
    return Detection(
        index=idx,
        class_id=cls_id,
        class_name=cls_name,
        class_display_name=cls_name.capitalize(),
        confidence=conf,
        confidence_percentage=round(conf * 100, 2),
        x_min=x1,
        y_min=y1,
        x_max=x2,
        y_max=y2,
        width=width,
        height=height
    )


def test_count_detections_by_class():
    """Valida el conteo por estado de madurez."""
    detections = [
        sample_detection(1, 0, "inmadura", 0.88),
        sample_detection(2, 1, "madura", 0.92),
        sample_detection(3, 1, "madura", 0.85),
        sample_detection(4, 2, "sobremadura", 0.95),
    ]
    counts = count_detections_by_class(detections)
    assert counts["inmadura"] == 1
    assert counts["madura"] == 2
    assert counts["sobremadura"] == 1


def test_determine_predominant_class_clear_winner():
    """Valida que retorne la clase con mayor conteo."""
    counts = {"inmadura": 1, "madura": 4, "sobremadura": 2}
    predominant = determine_predominant_class(counts)
    assert predominant == "Madura"

    counts_inm = {"inmadura": 3, "madura": 0, "sobremadura": 1}
    assert determine_predominant_class(counts_inm) == "Inmadura"


def test_determine_predominant_class_tie():
    """Valida que en caso de empate retorne 'Sin clase predominante'."""
    # Empate entre inmadura y madura
    counts_tie_2 = {"inmadura": 2, "madura": 2, "sobremadura": 1}
    assert determine_predominant_class(counts_tie_2) == "Sin clase predominante"

    # Empate triple
    counts_tie_3 = {"inmadura": 3, "madura": 3, "sobremadura": 3}
    assert determine_predominant_class(counts_tie_3) == "Sin clase predominante"


def test_determine_predominant_class_zero_detections():
    """Valida que con cero detecciones retorne 'Sin clase predominante'."""
    counts_zero = {"inmadura": 0, "madura": 0, "sobremadura": 0}
    assert determine_predominant_class(counts_zero) == "Sin clase predominante"


def test_build_inference_summary_with_detections():
    """Valida el cálculo de resumen con varias detecciones."""
    detections = [
        sample_detection(1, 1, "madura", 0.80),
        sample_detection(2, 1, "madura", 0.90),
    ]
    summary = build_inference_summary(detections, inference_time_ms=120.45)
    
    assert summary.total_detections == 2
    assert pytest.approx(summary.average_confidence, 0.001) == 0.85
    assert summary.inference_time_ms == 120.45
    assert summary.predominant_class == "Madura"
    assert summary.counts_by_class["madura"] == 2
    assert summary.percentages_by_class["madura"] == 100.0


def test_build_inference_summary_zero_detections():
    """Valida el comportamiento seguro con cero detecciones (sin división por cero ni NaN)."""
    detections = []
    summary = build_inference_summary(detections, inference_time_ms=55.2)
    
    assert summary.total_detections == 0
    assert summary.average_confidence is None
    assert summary.inference_time_ms == 55.2
    assert summary.predominant_class == "Sin clase predominante"
    assert summary.counts_by_class["inmadura"] == 0
    assert summary.counts_by_class["madura"] == 0
    assert summary.counts_by_class["sobremadura"] == 0
    assert summary.percentages_by_class["madura"] == 0.0


def test_detection_table_row_rounding():
    """Valida que las coordenadas y valores de la tabla se redondeen a 2 decimales."""
    det = sample_detection(
        idx=1,
        cls_id=1,
        cls_name="madura",
        conf=0.8872,
        x1=10.1234,
        y1=20.5678,
        x2=110.9999,
        y2=220.4444
    )
    row = det.to_table_row()
    
    assert row["N° Detección"] == 1
    assert row["Estado de Madurez"] == "Madura"
    assert row["Confianza (%)"] == "88.72%"
    assert row["X Mínimo"] == 10.12
    assert row["Y Mínimo"] == 20.57
    assert row["X Máximo"] == 111.00
    assert row["Y Máximo"] == 220.44
    assert row["Ancho (px)"] == 100.88
    assert row["Alto (px)"] == 199.88
