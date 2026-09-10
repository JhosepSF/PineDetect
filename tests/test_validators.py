"""Pruebas unitarias para el módulo de validación de archivos de imagen (src.validators)."""

import io
import pytest
from PIL import Image

from src.validators import validate_image_file
from src.config import MAX_FILE_SIZE_BYTES


def create_test_image_bytes(
    width: int = 100,
    height: int = 100,
    mode: str = "RGB",
    fmt: str = "JPEG",
    color: tuple = (200, 150, 50)
) -> bytes:
    """Crea una imagen válida en memoria para pruebas."""
    img = Image.new(mode, (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def test_validate_valid_jpeg_image():
    """Valida que una imagen JPEG estándar sea aceptada correctamente."""
    img_bytes = create_test_image_bytes(fmt="JPEG")
    res = validate_image_file(img_bytes, "foto_pina.jpg")
    
    assert res.is_valid is True
    assert res.error_message is None
    assert res.image is not None
    assert res.image.mode == "RGB"
    assert res.width == 100
    assert res.height == 100


def test_validate_valid_png_and_webp_images():
    """Valida formatos PNG y WEBP permitidos."""
    png_bytes = create_test_image_bytes(fmt="PNG")
    res_png = validate_image_file(png_bytes, "pina.png")
    assert res_png.is_valid is True
    assert res_png.image.mode == "RGB"

    webp_bytes = create_test_image_bytes(fmt="WEBP")
    res_webp = validate_image_file(webp_bytes, "pina.webp")
    assert res_webp.is_valid is True


def test_validate_rgba_converted_to_rgb():
    """Valida que imágenes con canal alfa (RGBA) se conviertan de manera segura a RGB."""
    img_rgba = Image.new("RGBA", (80, 80), (100, 200, 100, 255))
    buf = io.BytesIO()
    img_rgba.save(buf, format="PNG")
    res = validate_image_file(buf.getvalue(), "pina_transparente.png")
    
    assert res.is_valid is True
    assert res.image.mode == "RGB"


def test_reject_unsupported_extension():
    """Rechaza archivos con extensiones no permitidas."""
    img_bytes = create_test_image_bytes(fmt="JPEG")
    
    res_txt = validate_image_file(img_bytes, "documento.txt")
    assert res_txt.is_valid is False
    assert "Formato de archivo no soportado" in res_txt.error_message

    res_pdf = validate_image_file(img_bytes, "reporte.pdf")
    assert res_pdf.is_valid is False

    res_bmp = validate_image_file(img_bytes, "pina.bmp")
    assert res_bmp.is_valid is False


def test_reject_oversized_file():
    """Rechaza archivos que superen los 10 MB."""
    # Simular un archivo de 11 MB con extensión permitida
    oversized_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
    res = validate_image_file(oversized_bytes, "pina_gigante.jpg")
    
    assert res.is_valid is False
    assert "excede el tamaño máximo permitido" in res.error_message


def test_reject_empty_file():
    """Rechaza archivos vacíos (0 bytes)."""
    res = validate_image_file(b"", "vacio.jpg")
    assert res.is_valid is False
    assert "está vacío" in res.error_message


def test_reject_corrupt_image_bytes():
    """Rechaza archivos con bytes corruptos que no representan una imagen válida."""
    corrupt_bytes = b"\xFF\xD8\xFF\xE0" + b"datos_completamente_invalidos_y_corruptos"
    res = validate_image_file(corrupt_bytes, "corrupta.jpg")
    
    assert res.is_valid is False
    assert res.image is None
    assert "no es una imagen válida" in res.error_message or "error de lectura" in res.error_message
