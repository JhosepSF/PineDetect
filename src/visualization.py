"""Módulo de visualización gráfica, dibujo de bounding boxes y gráficos Plotly para PineDetect."""

import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go

from src.config import (
    CLASS_COLORS_RGB,
    CLASS_COLORS_HEX,
    CLASS_DISPLAY_NAMES,
)
from src.inference import Detection

logger = logging.getLogger(__name__)


def get_text_color_for_background(rgb: Tuple[int, int, int]) -> str:
    """Calcula el color de texto (blanco o negro) para un contraste óptimo según luminancia."""
    r, g, b = rgb
    # Fórmula estándar de luminancia perceptual ITU-R BT.709
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#000000" if luminance > 0.65 else "#FFFFFF"


def get_adaptive_font(font_size: int) -> ImageFont.ImageFont:
    """Intenta cargar una fuente TrueType escalable del sistema o retorna la predeterminada."""
    font_candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "calibri.ttf",
        "segoeui.ttf",
        "Helvetica.ttf"
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=font_size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=font_size)
    except Exception:
        return ImageFont.load_default()


def draw_detections_on_image(
    original_image: Image.Image,
    detections: List[Detection]
) -> Image.Image:
    """Dibuja las cajas delimitadoras y etiquetas personalizadas sobre la imagen original.

    Conserva la resolución nativa de la imagen, aplica colores específicos para cada
    estado de madurez y garantiza que las etiquetas no se corten en los bordes.

    Args:
        original_image: Imagen PIL original en modo RGB.
        detections: Lista de objetos Detection con coordenadas y clases.

    Returns:
        Nueva imagen PIL con las cajas y etiquetas renderizadas.
    """
    # Crear una copia en modo RGB para no mutar la original
    annotated_image = original_image.copy().convert("RGB")
    width, height = annotated_image.size
    
    if not detections:
        return annotated_image

    draw = ImageDraw.Draw(annotated_image, "RGBA")

    # Calcular grosor de línea y tamaño de fuente proporcional a la resolución
    min_dim = min(width, height)
    line_width = max(3, int(min_dim / 180))
    font_size = max(14, int(min_dim / 36))
    font = get_adaptive_font(font_size)

    for det in detections:
        color_rgb = CLASS_COLORS_RGB.get(det.class_id, (44, 160, 44))
        hex_color = CLASS_COLORS_HEX.get(det.class_id, "#2CA02C")
        text_color = get_text_color_for_background(color_rgb)

        x1, y1 = max(0.0, det.x_min), max(0.0, det.y_min)
        x2, y2 = min(float(width), det.x_max), min(float(height), det.y_max)

        # 1. Dibujar el rectángulo de la caja delimitadora
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color_rgb,
            width=line_width
        )

        # 2. Formatear la etiqueta: "Clase — confianza %"
        label_text = f"{det.class_display_name} — {det.confidence_percentage:.1f} %"

        # 3. Medir dimensiones del texto
        try:
            bbox = font.getbbox(label_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = font.getsize(label_text) if hasattr(font, "getsize") else (100, 20)

        padding_x = max(6, int(font_size * 0.35))
        padding_y = max(4, int(font_size * 0.25))
        badge_w = text_w + (2 * padding_x)
        badge_h = text_h + (2 * padding_y)

        # 4. Posicionar el badge evitando desbordamientos
        badge_x1 = x1
        badge_y2 = y1
        badge_y1 = badge_y2 - badge_h

        # Si se sale por la parte superior, colocarlo dentro de la caja
        if badge_y1 < 0:
            badge_y1 = y1
            badge_y2 = y1 + badge_h

        # Si se sale por la derecha
        if badge_x1 + badge_w > width:
            badge_x1 = max(0.0, width - badge_w)

        badge_x2 = badge_x1 + badge_w

        # 5. Dibujar fondo de la etiqueta con esquinas nítidas
        draw.rectangle(
            [badge_x1, badge_y1, badge_x2, badge_y2],
            fill=(*color_rgb, 245)
        )

        # 6. Dibujar texto centrado dentro del badge
        text_pos_x = badge_x1 + padding_x
        text_pos_y = badge_y1 + padding_y
        draw.text(
            (text_pos_x, text_pos_y),
            label_text,
            fill=text_color,
            font=font
        )

    return annotated_image


def create_class_distribution_chart(
    counts_by_class: Dict[str, int]
) -> Optional[go.Figure]:
    """Genera un gráfico de barras interactivo de Plotly con la distribución de madurez.

    Args:
        counts_by_class: Diccionario con conteos por clase ('inmadura', 'madura', 'sobremadura').

    Returns:
        Figura de Plotly estilizada o None si el total es cero.
    """
    total = sum(counts_by_class.values())
    if total == 0:
        return None

    categories = ["Inmadura", "Madura", "Sobremadura"]
    values = [
        counts_by_class.get("inmadura", 0),
        counts_by_class.get("madura", 0),
        counts_by_class.get("sobremadura", 0)
    ]
    colors = [
        CLASS_COLORS_HEX[0],  # Verde (#2CA02C)
        CLASS_COLORS_HEX[1],  # Amarillo (#F2B134)
        CLASS_COLORS_HEX[2]   # Rojo (#D62728)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            text=values,
            textposition="auto",
            textfont=dict(size=14, color="#FFFFFF", family="Inter, Arial, sans-serif"),
            marker=dict(
                color=colors,
                line=dict(color="rgba(0,0,0,0.15)", width=1.5),
                cornerradius=8
            ),
            hovertemplate="<b>%{x}</b><br>Cantidad: %{y} piñas<br>Proporción: %{customdata:.1f}%<extra></extra>",
            customdata=[(val / total) * 100 if total > 0 else 0 for val in values]
        )
    )

    max_val = max(values)
    y_limit = max(4, int(max_val * 1.25) + 1)

    fig.update_layout(
        title=dict(
            text="<b>Distribución de Piñas por Estado de Madurez</b>",
            x=0.02,
            font=dict(size=16, color="#1B4332", family="Inter, Arial, sans-serif")
        ),
        xaxis=dict(
            title=None,
            tickfont=dict(size=13, color="#2D3748", family="Inter, Arial, sans-serif"),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(
                text="Cantidad de Piñas",
                font=dict(size=12, color="#4A5568")
            ),
            tickfont=dict(size=11, color="#4A5568"),
            range=[0, y_limit],
            dtick=1 if y_limit <= 10 else None,
            gridcolor="#E2E8F0",
            zeroline=False
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=50, b=30),
        height=320,
        dragmode=False
    )

    return fig
