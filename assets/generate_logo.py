"""Script utilitario para generar un logotipo profesional de PineDetect."""

from pathlib import Path
from PIL import Image, ImageDraw

def generate_logo(output_path: Path) -> None:
    size = (256, 256)
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Fondo circular suave verde oscuro
    draw.ellipse([8, 8, 248, 248], fill=(27, 67, 50, 255), outline=(242, 177, 52, 255), width=6)

    # Hojas superiores de la piña (Verde esmeralda y claro)
    leaf_color_dark = (45, 106, 79, 255)
    leaf_color_light = (64, 145, 108, 255)
    
    # Hojas centrales y laterales
    draw.polygon([(128, 38), (116, 95), (140, 95)], fill=leaf_color_light)
    draw.polygon([(128, 48), (96, 92), (118, 98)], fill=leaf_color_dark)
    draw.polygon([(128, 48), (160, 92), (138, 98)], fill=leaf_color_dark)
    draw.polygon([(128, 62), (80, 105), (106, 108)], fill=leaf_color_light)
    draw.polygon([(128, 62), (176, 105), (150, 108)], fill=leaf_color_light)

    # Cuerpo de la piña (Óvalo dorado cálido)
    body_bbox = [82, 95, 174, 205]
    draw.ellipse(body_bbox, fill=(242, 177, 52, 255), outline=(217, 148, 20, 255), width=3)

    # Patrón de rombos / cuadrícula de la piña
    pattern_color = (217, 148, 20, 200)
    lines = [
        [(96, 115), (160, 185)],
        [(90, 140), (145, 198)],
        [(115, 102), (170, 160)],
        [(160, 115), (96, 185)],
        [(166, 140), (111, 198)],
        [(141, 102), (86, 160)],
    ]
    for p1, p2 in lines:
        draw.line([p1, p2], fill=pattern_color, width=3)

    # Bounding box tecnológica estilizada representando detección inteligente
    tech_cyan = (242, 177, 52, 255)
    # Esquinas del visor de detección
    corner_len = 16
    draw.line([(68, 85), (68 + corner_len, 85)], fill=tech_cyan, width=4)
    draw.line([(68, 85), (68, 85 + corner_len)], fill=tech_cyan, width=4)

    draw.line([(188, 85), (188 - corner_len, 85)], fill=tech_cyan, width=4)
    draw.line([(188, 85), (188, 85 + corner_len)], fill=tech_cyan, width=4)

    draw.line([(68, 215), (68 + corner_len, 215)], fill=tech_cyan, width=4)
    draw.line([(68, 215), (68, 215 - corner_len)], fill=tech_cyan, width=4)

    draw.line([(188, 215), (188 - corner_len, 215)], fill=tech_cyan, width=4)
    draw.line([(188, 215), (188, 215 - corner_len)], fill=tech_cyan, width=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    print(f"Logotipo generado exitosamente en: {output_path}")

if __name__ == "__main__":
    out = Path(__file__).parent / "logo_placeholder.png"
    generate_logo(out)
