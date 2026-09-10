"""Vistas y controladores de la aplicación Django PineDetect."""

import base64
import io
import json
import logging
from typing import Any, Dict

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from src.config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_DESCRIPTION,
    DISCLAIMER_TEXT,
    PRIVACY_NOTICE_TEXT,
    MODEL_PATH,
    MODEL_FILENAME,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    CONFIDENCE_STEP,
    IOU_MIN,
    IOU_MAX,
    IOU_STEP,
    MAX_FILE_SIZE_MB,
    CLASS_DISPLAY_NAMES,
    CLASS_COLORS_HEX,
)
from src.model_loader import load_model, check_model_exists, get_device_info, get_model_info, resolve_model_path
from src.validators import validate_image_file, ImageValidationResult
from src.inference import run_pineapple_detection, Detection, InferenceSummary, build_inference_summary
from src.visualization import draw_detections_on_image
from src.exporters import (
    export_image_to_png_bytes,
    export_detections_to_csv_bytes,
    export_summary_to_json_bytes,
)

import numpy as np
from django.core.serializers.json import DjangoJSONEncoder

class NumpySafeJSONEncoder(DjangoJSONEncoder):
    """Codificador JSON seguro contra tipos de datos numéricos de NumPy."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

logger = logging.getLogger(__name__)


@ensure_csrf_cookie
def index_view(request: HttpRequest) -> HttpResponse:
    """Renderiza la vista principal de PineDetect."""
    model_info = get_model_info()
    model_available = model_info['exists']
    device_info = model_info['device']

    context = {
        'app_name': APP_NAME,
        'app_subtitle': APP_SUBTITLE,
        'app_description': APP_DESCRIPTION,
        'disclaimer_text': DISCLAIMER_TEXT,
        'privacy_notice_text': PRIVACY_NOTICE_TEXT,
        'model_available': model_available,
        'model_filename': model_info['filename'],
        'model_size_mb': model_info['size_mb'],
        'model_path': model_info['path'],
        'device_info': device_info,
        'default_conf': DEFAULT_CONFIDENCE,
        'default_iou': DEFAULT_IOU,
        'default_imgsz': DEFAULT_IMGSZ,
        'conf_min': CONFIDENCE_MIN,
        'conf_max': CONFIDENCE_MAX,
        'conf_step': CONFIDENCE_STEP,
        'iou_min': IOU_MIN,
        'iou_max': IOU_MAX,
        'iou_step': IOU_STEP,
        'max_file_size_mb': MAX_FILE_SIZE_MB,
        'class_colors': CLASS_COLORS_HEX,
    }
    return render(request, 'detector/index.html', context)


@require_GET
def status_api_view(request: HttpRequest) -> JsonResponse:
    """Retorna el estado de disponibilidad del modelo y parámetros técnicos."""
    model_info = get_model_info()
    return JsonResponse({
        'status': 'ok',
        'app_name': APP_NAME,
        'model_available': model_info['exists'],
        'model_filename': model_info['filename'],
        'model_path': model_info['path'],
        'model_size_mb': model_info['size_mb'],
        'device_info': model_info['device'],
        'classes': {0: 'inmadura', 1: 'madura', 2: 'sobremadura'},
        'default_parameters': {
            'imgsz': DEFAULT_IMGSZ,
            'confidence': DEFAULT_CONFIDENCE,
            'iou': DEFAULT_IOU
        }
    })


@require_POST
def analyze_api_view(request: HttpRequest) -> JsonResponse:
    """Procesa una imagen subida por AJAX, ejecuta YOLOv8 y retorna resultados estructurados."""
    if 'image' not in request.FILES:
        return JsonResponse({
            'success': False,
            'error': 'No se proporcionó ningún archivo de imagen en la solicitud.'
        }, status=400)

    image_file = request.FILES['image']
    
    # 1. Leer bytes en memoria
    file_bytes = image_file.read()
    
    # 2. Validar imagen
    val_result: ImageValidationResult = validate_image_file(file_bytes, image_file.name)
    if not val_result.is_valid:
        return JsonResponse({
            'success': False,
            'error': val_result.error_message
        }, status=400)

    # 3. Leer umbrales de la petición
    raw_conf = str(request.POST.get('confidence', DEFAULT_CONFIDENCE)).replace(',', '.').strip()
    try:
        conf_thresh = float(raw_conf)
        if conf_thresh > 1.0 and conf_thresh <= 100.0:
            conf_thresh = conf_thresh / 100.0
        conf_thresh = max(0.01, min(0.99, conf_thresh))
    except (ValueError, TypeError):
        conf_thresh = DEFAULT_CONFIDENCE

    raw_iou = str(request.POST.get('iou', DEFAULT_IOU)).replace(',', '.').strip()
    try:
        iou_thresh = float(raw_iou)
        if iou_thresh > 1.0 and iou_thresh <= 100.0:
            iou_thresh = iou_thresh / 100.0
        iou_thresh = max(0.01, min(0.99, iou_thresh))
    except (ValueError, TypeError):
        iou_thresh = DEFAULT_IOU

    # 4. Cargar modelo Singleton
    model, load_err = load_model(use_cache=True)
    if load_err or model is None:
        return JsonResponse({
            'success': False,
            'error': (
                f"El modelo no está disponible: {load_err}. "
                f"Por favor, copia '{MODEL_FILENAME}' en la carpeta 'models/'."
            )
        }, status=503)

    # 5. Ejecutar inferencia
    try:
        valid_pil_image = val_result.image
        print(f"\n[PineDetect] >>> Procesando '{image_file.name}' ({val_result.width}x{val_result.height} px) | Conf: {conf_thresh:.2f} | IoU: {iou_thresh:.2f}")
        detections, summary = run_pineapple_detection(
            model=model,
            image_input=valid_pil_image,
            imgsz=DEFAULT_IMGSZ,
            conf=conf_thresh,
            iou=iou_thresh
        )
        print(f"[PineDetect] <<< Resultado: {summary.total_detections} piña(s) detectada(s) en {summary.inference_time_ms:.1f} ms | Conteo: {summary.counts_by_class} | Predominante: {summary.predominant_class}\n")

        # 6. Dibujar cajas en alta resolución
        annotated_pil_image = draw_detections_on_image(valid_pil_image, detections)

        # 7. Convertir imágenes a Base64 data URL
        orig_buf = io.BytesIO()
        valid_pil_image.save(orig_buf, format='JPEG', quality=92)
        orig_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_buf.getvalue()).decode('utf-8')

        annotated_png_bytes = export_image_to_png_bytes(annotated_pil_image)
        annotated_b64 = "data:image/png;base64," + base64.b64encode(annotated_png_bytes).decode('utf-8')

        # 8. Construir respuesta estructurada
        response_data: Dict[str, Any] = {
            'success': True,
            'filename': image_file.name,
            'image_info': {
                'width': val_result.width,
                'height': val_result.height,
                'size_mb': val_result.size_mb,
                'format': val_result.format
            },
            'original_image': orig_b64,
            'annotated_image': annotated_b64,
            'thresholds': {
                'conf': conf_thresh,
                'iou': iou_thresh,
                'imgsz': DEFAULT_IMGSZ
            },
            'summary': {
                'total_detections': summary.total_detections,
                'average_confidence': summary.average_confidence,
                'average_confidence_percentage': (
                    round(summary.average_confidence * 100, 2)
                    if summary.average_confidence is not None
                    else None
                ),
                'inference_time_ms': summary.inference_time_ms,
                'predominant_class': summary.predominant_class,
                'counts_by_class': summary.counts_by_class,
                'percentages_by_class': summary.percentages_by_class
            },
            'detections': [d.to_dict() for d in detections],
            'table_rows': [d.to_table_row() for d in detections]
        }

        return JsonResponse(response_data, encoder=NumpySafeJSONEncoder)

    except Exception as exc:
        logger.error("Error durante la inferencia en Django: %s", str(exc), exc_info=True)
        return JsonResponse({
            'success': False,
            'error': f"Error interno durante el procesamiento de la imagen: {str(exc)}"
        }, status=500)


@require_POST
def export_csv_api_view(request: HttpRequest) -> HttpResponse:
    """Genera y descarga el reporte CSV con codificación UTF-8 con BOM."""
    try:
        body = json.loads(request.body.decode('utf-8'))
        raw_detections = body.get('detections', [])
        
        detections: list[Detection] = []
        for d in raw_detections:
            det = Detection(
                index=d.get('index', 1),
                class_id=d.get('class_id', 0),
                class_name=d.get('class_name', 'inmadura'),
                class_display_name=d.get('class_display_name', 'Inmadura'),
                confidence=float(d.get('confidence', 0.0)),
                confidence_percentage=float(d.get('confidence_percentage', 0.0)),
                x_min=float(d.get('x_min', 0.0)),
                y_min=float(d.get('y_min', 0.0)),
                x_max=float(d.get('x_max', 0.0)),
                y_max=float(d.get('y_max', 0.0)),
                width=float(d.get('width', 0.0)),
                height=float(d.get('height', 0.0)),
            )
            detections.append(det)

        csv_bytes = export_detections_to_csv_bytes(detections)
        filename = body.get('filename', 'pinedetect_resultados')
        clean_name = f"{filename}_detecciones.csv"

        response = HttpResponse(csv_bytes, content_type='text/csv; charset=utf-8-sig')
        response['Content-Disposition'] = f'attachment; filename="{clean_name}"'
        return response
    except Exception as exc:
        logger.error("Error al exportar CSV: %s", str(exc))
        return HttpResponse("Error al generar CSV", status=400)


@require_POST
def export_json_api_view(request: HttpRequest) -> HttpResponse:
    """Genera y descarga el archivo JSON estructurado de resultados."""
    try:
        body = json.loads(request.body.decode('utf-8'))
        raw_detections = body.get('detections', [])
        raw_summary = body.get('summary', {})
        thresholds = body.get('thresholds', {'conf': DEFAULT_CONFIDENCE, 'iou': DEFAULT_IOU, 'imgsz': DEFAULT_IMGSZ})
        filename = body.get('filename', 'pinedetect_resultados')

        detections: list[Detection] = []
        for d in raw_detections:
            det = Detection(
                index=d.get('index', 1),
                class_id=d.get('class_id', 0),
                class_name=d.get('class_name', 'inmadura'),
                class_display_name=d.get('class_display_name', 'Inmadura'),
                confidence=float(d.get('confidence', 0.0)),
                confidence_percentage=float(d.get('confidence_percentage', 0.0)),
                x_min=float(d.get('x_min', 0.0)),
                y_min=float(d.get('y_min', 0.0)),
                x_max=float(d.get('x_max', 0.0)),
                y_max=float(d.get('y_max', 0.0)),
                width=float(d.get('width', 0.0)),
                height=float(d.get('height', 0.0)),
            )
            detections.append(det)

        summary = InferenceSummary(
            total_detections=raw_summary.get('total_detections', len(detections)),
            average_confidence=raw_summary.get('average_confidence'),
            inference_time_ms=float(raw_summary.get('inference_time_ms', 0.0)),
            predominant_class=raw_summary.get('predominant_class', 'Sin clase predominante'),
            counts_by_class=raw_summary.get('counts_by_class', {'inmadura': 0, 'madura': 0, 'sobremadura': 0}),
            percentages_by_class=raw_summary.get('percentages_by_class', {'inmadura': 0.0, 'madura': 0.0, 'sobremadura': 0.0}),
        )

        json_bytes = export_summary_to_json_bytes(
            summary=summary,
            detections=detections,
            thresholds=thresholds,
            model_name=MODEL_FILENAME,
            additional_metadata={"archivo_procesado": filename}
        )

        response = HttpResponse(json_bytes, content_type='application/json; charset=utf-8')
        response['Content-Disposition'] = f'attachment; filename="{filename}_resumen.json"'
        return response
    except Exception as exc:
        logger.error("Error al exportar JSON: %s", str(exc))
        return HttpResponse("Error al generar JSON", status=400)
