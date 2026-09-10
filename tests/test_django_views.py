"""Pruebas de integración para las vistas y endpoints API de Django (detector.views)."""

import io
import json
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from PIL import Image

@pytest.fixture(autouse=True)
def enable_db_access_for_all_tests(db):
    pass

@pytest.fixture
def client():
    return Client()

def create_dummy_image_file(filename="test_pina.jpg", fmt="JPEG"):
    """Crea un archivo subido en memoria para pruebas."""
    img = Image.new("RGB", (120, 120), color=(180, 220, 100))
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return SimpleUploadedFile(
        name=filename,
        content=buffer.getvalue(),
        content_type=f"image/{fmt.lower()}"
    )

def test_index_view_status_code_and_template(client):
    """Valida que la vista principal cargue exitosamente con código 200."""
    response = client.get('/')
    assert response.status_code == 200
    assert "PineDetect" in response.content.decode('utf-8')
    assert "Detección y clasificación inteligente" in response.content.decode('utf-8')

def test_status_api_endpoint(client):
    """Valida el endpoint de estado /api/status/."""
    response = client.get('/api/status/')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert data['app_name'] == 'PineDetect'
    assert 'model_available' in data
    assert 'default_parameters' in data

def test_analyze_endpoint_missing_file(client):
    """Valida que /api/analyze/ retorne error 400 si no se envía archivo."""
    response = client.post('/api/analyze/', {})
    assert response.status_code == 400
    data = response.json()
    assert data['success'] is False
    assert "No se proporcionó ningún archivo" in data['error']

def test_analyze_endpoint_invalid_extension(client):
    """Valida que /api/analyze/ rechace archivos con extensión no soportada."""
    invalid_file = SimpleUploadedFile(
        name="notas.txt",
        content=b"esto no es una imagen",
        content_type="text/plain"
    )
    response = client.post('/api/analyze/', {'image': invalid_file})
    assert response.status_code == 400
    data = response.json()
    assert data['success'] is False
    assert "Formato de archivo no soportado" in data['error']

def test_export_csv_endpoint(client):
    """Valida la generación de CSV desde /api/export/csv/."""
    payload = {
        'filename': 'test_pina',
        'detections': [
            {
                'index': 1,
                'class_id': 1,
                'class_name': 'madura',
                'class_display_name': 'Madura',
                'confidence': 0.912,
                'confidence_percentage': 91.20,
                'x_min': 10.0,
                'y_min': 20.0,
                'x_max': 100.0,
                'y_max': 200.0,
                'width': 90.0,
                'height': 180.0
            }
        ]
    }
    response = client.post(
        '/api/export/csv/',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    assert response['Content-Type'] == 'text/csv; charset=utf-8-sig'
    assert response.content.startswith(b"\xef\xbb\xbf")
    assert "Madura" in response.content.decode('utf-8-sig')

def test_export_json_endpoint(client):
    """Valida la generación de JSON desde /api/export/json/."""
    payload = {
        'filename': 'test_pina',
        'summary': {
            'total_detections': 1,
            'average_confidence': 0.89,
            'inference_time_ms': 110.5,
            'predominant_class': 'Madura',
            'counts_by_class': {'inmadura': 0, 'madura': 1, 'sobremadura': 0},
            'percentages_by_class': {'inmadura': 0.0, 'madura': 100.0, 'sobremadura': 0.0}
        },
        'detections': [
            {
                'index': 1,
                'class_id': 1,
                'class_name': 'madura',
                'class_display_name': 'Madura',
                'confidence': 0.89,
                'confidence_percentage': 89.0,
                'x_min': 10.0,
                'y_min': 20.0,
                'x_max': 100.0,
                'y_max': 200.0,
                'width': 90.0,
                'height': 180.0
            }
        ],
        'thresholds': {'conf': 0.25, 'iou': 0.70, 'imgsz': 640}
    }
    response = client.post(
        '/api/export/json/',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    assert 'application/json' in response['Content-Type']
    data = json.loads(response.content.decode('utf-8'))
    assert data['aplicativo'] == 'PineDetect'
    assert data['resumen_inferencia']['clase_predominante'] == 'Madura'
