"""Módulo de carga y verificación del modelo YOLOv8 para PineDetect (compatible con Django y Streamlit)."""

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Any
import torch

from src.config import BASE_DIR, MODEL_PATH, MODEL_FILENAME

logger = logging.getLogger(__name__)

# Cache de modelo Singleton en memoria para servidores web (Django, Flask, etc.)
_MODEL_LOCK = threading.Lock()
_CACHED_MODEL: Optional[Any] = None
_CACHED_PATH: Optional[str] = None


def resolve_model_path(custom_path: Optional[Path] = None) -> Optional[Path]:
    """Busca el archivo del modelo .pt en rutas estándar y relativas para máxima robustez.

    Args:
        custom_path: Ruta opcional proporcionada por el usuario.

    Returns:
        Path absoluto resuelto si el archivo existe, o None si no se encuentra.
    """
    if custom_path is not None:
        p = Path(custom_path)
        if p.is_file():
            return p.resolve()

    candidate_paths = [
        MODEL_PATH,                                              # PineDetect/models/detector_madurez_pina_yolov8m_v1.pt
        BASE_DIR / "models" / MODEL_FILENAME,
        BASE_DIR / "models" / "modelo" / MODEL_FILENAME,
        BASE_DIR / "modelo" / MODEL_FILENAME,
        BASE_DIR / MODEL_FILENAME,
        Path("models") / MODEL_FILENAME,
        Path("models") / "modelo" / MODEL_FILENAME,
        Path("modelo") / MODEL_FILENAME,
        Path(MODEL_FILENAME),
    ]

    for candidate in candidate_paths:
        try:
            if candidate.is_file():
                logger.debug("Modelo encontrado en ruta candidata: %s", candidate)
                return candidate.resolve()
        except Exception:
            continue

    return None


def check_model_exists(custom_path: Optional[Path] = None) -> bool:
    """Verifica si el archivo del modelo .pt existe en alguna de las rutas válidas."""
    return resolve_model_path(custom_path) is not None


def get_model_info() -> dict:
    """Retorna información detallada sobre el modelo detectado en el sistema."""
    resolved_path = resolve_model_path()
    if resolved_path is None:
        return {
            "exists": False,
            "filename": MODEL_FILENAME,
            "path": str(MODEL_PATH),
            "size_mb": 0.0,
            "device": get_device_info(),
        }

    size_mb = resolved_path.stat().st_size / (1024 * 1024)
    return {
        "exists": True,
        "filename": resolved_path.name,
        "path": str(resolved_path),
        "size_mb": round(size_mb, 2),
        "device": get_device_info(),
    }


def get_device_info() -> str:
    """Identifica el dispositivo de aceleración disponible (GPU CUDA o CPU)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"GPU ({gpu_name})"
    return "CPU"


def _raw_load_yolo_model(model_path_str: str) -> Any:
    """Carga interna del modelo YOLOv8 utilizando Ultralytics."""
    from ultralytics import YOLO
    
    path = Path(model_path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en '{model_path_str}'. "
            f"Por favor, copia '{MODEL_FILENAME}' dentro del directorio 'models/'."
        )

    logger.info("Cargando modelo YOLO desde: %s", model_path_str)
    model = YOLO(model_path_str)
    return model


def load_model(
    model_path: Optional[Path] = None,
    use_cache: bool = True
) -> Tuple[Optional[Any], Optional[str]]:
    """Carga el modelo de detección y madurez de piñas como Singleton en memoria.

    Args:
        model_path: Ruta opcional al archivo .pt. Si no se especifica, busca automáticamente.
        use_cache: Si es True, reutiliza la instancia previamente cargada en memoria.

    Returns:
        Tupla (modelo, mensaje_error). Si la carga fue exitosa, mensaje_error es None.
    """
    global _CACHED_MODEL, _CACHED_PATH

    resolved_path = resolve_model_path(model_path)

    if resolved_path is None:
        error_msg = (
            f"El archivo del modelo no está disponible en '{MODEL_PATH}'. "
            f"Asegúrate de colocar '{MODEL_FILENAME}' dentro de la carpeta 'models/'."
        )
        logger.warning(error_msg)
        return None, error_msg

    target_str = str(resolved_path)

    if use_cache and _CACHED_MODEL is not None and _CACHED_PATH == target_str:
        return _CACHED_MODEL, None

    with _MODEL_LOCK:
        # Doble verificación dentro del bloqueo thread-safe
        if use_cache and _CACHED_MODEL is not None and _CACHED_PATH == target_str:
            return _CACHED_MODEL, None

        try:
            model = _raw_load_yolo_model(target_str)
            if use_cache:
                _CACHED_MODEL = model
                _CACHED_PATH = target_str
            return model, None
        except Exception as exc:
            err_msg = f"Error al inicializar el modelo YOLOv8: {str(exc)}"
            logger.error(err_msg, exc_info=True)
            return None, err_msg
