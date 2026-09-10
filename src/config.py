"""Configuración global, constantes, rutas y parámetros predeterminados para PineDetect."""

from pathlib import Path
from typing import Dict, Set, Tuple

# Rutas base del proyecto
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
ASSETS_DIR: Path = BASE_DIR / "assets"

# Ruta del modelo YOLOv8m entrenado
MODEL_FILENAME: str = "detector_madurez_pina_yolov8m_v1.pt"
MODEL_PATH: Path = MODELS_DIR / MODEL_FILENAME

# Identificadores y nombres de clases
CLASS_NAMES: Dict[int, str] = {
    0: "inmadura",
    1: "madura",
    2: "sobremadura"
}

# Nombres con capitalización formal para visualización
CLASS_DISPLAY_NAMES: Dict[int, str] = {
    0: "Inmadura",
    1: "Madura",
    2: "Sobremadura"
}

# Paleta de colores consistente según especificación
CLASS_COLORS_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (44, 160, 44),     # Verde
    1: (242, 177, 52),    # Amarillo / Naranja
    2: (214, 39, 40)      # Rojo
}

CLASS_COLORS_HEX: Dict[int, str] = {
    0: "#2CA02C",
    1: "#F2B134",
    2: "#D62728"
}

# Mapeo por nombre de clase
NAME_TO_COLOR_HEX: Dict[str, str] = {
    "inmadura": "#2CA02C",
    "madura": "#F2B134",
    "sobremadura": "#D62728",
    "Inmadura": "#2CA02C",
    "Madura": "#F2B134",
    "Sobremadura": "#D62728"
}

# Parámetros predeterminados de inferencia
DEFAULT_IMGSZ: int = 640
DEFAULT_CONFIDENCE: float = 0.25
DEFAULT_IOU: float = 0.70

CONFIDENCE_MIN: float = 0.10
CONFIDENCE_MAX: float = 0.90
CONFIDENCE_STEP: float = 0.05

IOU_MIN: float = 0.30
IOU_MAX: float = 0.90
IOU_STEP: float = 0.05

# Parámetros de validación de imágenes
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
MAX_FILE_SIZE_MB: int = 10
ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# Textos informativos y descriptivos
APP_NAME: str = "PineDetect"
APP_SUBTITLE: str = "Detección y clasificación inteligente del estado de madurez de piñas"
APP_DESCRIPTION: str = "Carga una imagen para localizar automáticamente las piñas y conocer su estado de madurez."

DISCLAIMER_TEXT: str = (
    "Los resultados constituyen una estimación automática y pueden verse "
    "afectados por la iluminación, orientación, oclusión y calidad de la imagen."
)

PRIVACY_NOTICE_TEXT: str = (
    "Las imágenes se procesan temporalmente durante la sesión y no se "
    "almacenan de forma permanente."
)
