import os
import logging

logger = logging.getLogger("PaintByNumbers")


def export_to_svg(polygons: list, placements: list, canvas_shape: tuple, output_path: str):
    """
    Генерирует векторный SVG-файл, готовый для плоттера или печати.
    """
    h, w = canvas_shape[:2]

    # 1. Заголовок SVG файла и базовые стили
    svg_content = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%">',
        '  <style>',
        # Стили для линий контура (толщина 2px для огромного холста - это тонкая изящная линия)
        '    .line { fill: none; stroke: black; stroke-width: 2px; stroke-linejoin: round; stroke-linecap: round; }',
        # Стили для текста (центрирование и цвет)
        '    .text { font-family: "Arial", sans-serif; fill: #666666; text-anchor: middle; dominant-baseline: central; }',
        '  </style>'
    ]

    # 2. Отрисовка контуров (векторные пути)
    svg_content.append('  <g id="contours">')
    for poly in polygons:
        cnt = poly["contour"]
        if len(cnt) < 3:
            continue

        # Превращаем массив координат OpenCV в строку пути SVG: "M x,y L x,y L x,y Z"
        # M = MoveTo, L = LineTo, Z = ClosePath
        points_str = " L ".join([f"{pt[0][0]},{pt[0][1]}" for pt in cnt])
        path_d = f"M {points_str} Z"

        svg_content.append(f'    <path class="line" d="{path_d}" />')
    svg_content.append('  </g>')

    # 3. Отрисовка текста (номера зон)
    svg_content.append('  <g id="numbers">')
    for p in placements:
        text = p["text"]
        cx, cy = p["x"], p["y"]
        radius = p["radius"]

        # Переводим радиус (из OpenCV) в размер шрифта SVG.
        # Подбираем коэффициент так, чтобы цифра хорошо вписывалась в свою зону.
        font_size = max(10, int(radius * 1.2))

        svg_content.append(f'    <text class="text" x="{cx}" y="{cy}" font-size="{font_size}px">{text}</text>')
    svg_content.append('  </g>')

    # Закрываем тег
    svg_content.append('</svg>')

    # Сохраняем в файл
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_content))

    logger.info(f"Векторный файл успешно сохранен: {output_path}")