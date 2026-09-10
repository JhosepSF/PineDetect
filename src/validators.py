"""Módulo de validación de archivos de imagen para PineDetect."""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError

from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)


@dataclass
class ImageValidationResult:
    """Resultado estructurado de la validación de una imagen."""
    is_valid: bool
    error_message: Optional[str] = None
    image: Optional[Image.Image] = None
    format: Optional[str] = None
    width: int = 0
    height: int = 0
    size_mb: float = 0.0


def validate_image_file(
    file_bytes: bytes,
    filename: str
) -> ImageValidationResult:
    """Valida un archivo de imagen en memoria según extensión, tamaño, integridad y dimensiones.

    Args:
        file_bytes: Contenido binario del archivo subido.
        filename: Nombre original del archivo con su extensión.

    Returns:
        ImageValidationResult con el estado de la validación, mensaje de error (si aplica)
        y la imagen cargada en modo RGB.
    """
    if not file_bytes:
        logger.warning("Intento de validación con archivo vacío: %s", filename)
        return ImageValidationResult(
            is_valid=False,
            error_message="El archivo proporcionado está vacío o no contiene datos."
        )

    # 1. Validación de extensión
    file_path = Path(filename)
    extension = file_path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        logger.warning("Extensión no permitida: %s en archivo %s", extension, filename)
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return ImageValidationResult(
            is_valid=False,
            error_message=(
                f"Formato de archivo no soportado ({extension or 'sin extensión'}). "
                f"Formatos permitidos: {allowed_str}."
            )
        )

    # 2. Validación de tamaño
    file_size_bytes = len(file_bytes)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        logger.warning(
            "Archivo excede tamaño máximo: %.2f MB > %d MB",
            file_size_mb,
            MAX_FILE_SIZE_MB
        )
        return ImageValidationResult(
            is_valid=False,
            error_message=(
                f"El archivo ({file_size_mb:.2f} MB) excede el tamaño máximo permitido "
                f"de {MAX_FILE_SIZE_MB} MB."
            ),
            size_mb=file_size_mb
        )

    # 3. Validación de integridad de la imagen con Pillow
    try:
        # Primero verificar integridad estructural
        temp_buffer = io.BytesIO(file_bytes)
        with Image.open(temp_buffer) as img_verify:
            img_verify.verify()
            detected_format = img_verify.format

        # Abrir imagen para carga real y conversión a RGB
        load_buffer = io.BytesIO(file_bytes)
        img = Image.open(load_buffer)
        
        # Corregir orientación automáticamente según metadatos EXIF (fotos de móviles/cámaras)
        from PIL import ImageOps
        try:
            img = ImageOps.exif_transpose(img)
        except Exception as exif_err:
            logger.debug("No se aplicó exif_transpose: %s", str(exif_err))

        # 4. Validación de dimensiones
        width, height = img.size
        if width <= 0 or height <= 0:
            logger.warning("Dimensiones inválidas (%dx%d) en %s", width, height, filename)
            return ImageValidationResult(
                is_valid=False,
                error_message="La imagen tiene dimensiones no válidas (ancho o alto igual a cero)."
            )

        # 5. Conversión a modo RGB garantizado
        if img.mode != "RGB":
            img = img.convert("RGB")

        return ImageValidationResult(
            is_valid=True,
            image=img,
            format=detected_format or extension.lstrip(".").upper(),
            width=width,
            height=height,
            size_mb=round(file_size_mb, 2)
        )

    except UnidentifiedImageError:
        logger.error("No se pudo identificar el archivo como una imagen válida: %s", filename)
        return ImageValidationResult(
            is_valid=False,
            error_message="El archivo no es una imagen válida o está dañado."
        )
    except Exception as exc:
        logger.error("Error al procesar la imagen %s: %s", filename, str(exc))
        return ImageValidationResult(
            is_valid=False,
            error_message=f"No se pudo cargar la imagen debido a un error de lectura: {str(exc)}"
        )
