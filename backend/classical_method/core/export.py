import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def export_svg(output_components, labels: list, config, save_path: str) -> None:
    """
    Экспортирует изображение с номерами в SVG формат.
    
    Args:
        output_components: np.ndarray - маска компонент
        labels: list[LabelInfo] - список информации о размещении меток
        config: dict - конфигурация
        save_path: str - путь для сохранения SVG файла
    """
    h, w = output_components.shape
    
    # Начинаем SVG строку
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<defs><style>text { font-family: Arial, sans-serif; }</style></defs>'
    ]
    
    # Рисуем контуры для всех компонент
    unique_components = np.unique(output_components)
    for comp_id in unique_components:
        if comp_id == 0:
            continue
        
        # Создаём маску компоненты
        mask = (output_components == comp_id).astype(np.uint8)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Рисуем контуры как path
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Конвертируем контур в SVG path
            path_d = _contour_to_svg_path(contour)
            svg_lines.append(
                f'<path d="{path_d}" fill="none" stroke="black" stroke-width="0.5"/>'
            )
    
    # Добавляем текст для размещённых меток
    for label in labels:
        if not label.placed:
            continue
        
        svg_lines.append(
            f'<text x="{label.cx}" y="{label.cy}" font-size="{label.font_size}" '
            f'text-anchor="middle" dominant-baseline="central" fill="black">'
            f'{label.color_number}</text>'
        )
    
    # Закрываем SVG
    svg_lines.append('</svg>')
    
    # Записываем в файл
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_lines))


def _contour_to_svg_path(contour) -> str:
    """
    Конвертирует OpenCV контур в SVG path строку.
    
    Args:
        contour: np.ndarray - контур из cv2.findContours
    
    Returns:
        str - SVG path d атрибут
    """
    if len(contour) == 0:
        return ""
    
    path_parts = []
    
    # Начинаем с первой точки
    first_point = contour[0][0]
    path_parts.append(f"M {int(first_point[0])} {int(first_point[1])}")
    
    # Добавляем остальные точки
    for point in contour[1:]:
        x, y = int(point[0][0]), int(point[0][1])
        path_parts.append(f"L {x} {y}")
    
    # Закрываем контур
    path_parts.append("Z")
    
    return " ".join(path_parts)


def export_pdf(svg_path: str, legend_img: np.ndarray, save_path: str) -> None:
    """
    Экспортирует SVG и легенду в PDF формат.
    
    Args:
        svg_path: str - путь к SVG файлу
        legend_img: np.ndarray (RGB) - изображение легенды
        save_path: str - путь для сохранения PDF файла
    """
    # Пытаемся загрузить SVG через cairosvg
    svg_img = None
    try:
        import cairosvg
        from io import BytesIO
        
        # Конвертируем SVG в PNG через cairosvg
        png_bytes = BytesIO()
        cairosvg.svg2png(url=svg_path, write_to=png_bytes)
        png_bytes.seek(0)
        svg_img = Image.open(png_bytes).convert('RGB')
        svg_img = np.array(svg_img, dtype=np.uint8)
    except ImportError:
        # cairosvg недоступен, используем fallback
        svg_img = None
    
    # Создаём figure с двумя subplots
    if svg_img is not None:
        # Есть SVG изображение
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Верхний subplot - SVG
        ax1.imshow(svg_img)
        ax1.set_title('Paint by Numbers - SVG', fontsize=16)
        ax1.axis('off')
        
        # Нижний subplot - легенда
        ax2.imshow(legend_img)
        ax2.set_title('Легенда', fontsize=16)
        ax2.axis('off')
    else:
        # Fallback: только легенда + текст
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.imshow(legend_img)
        ax.set_title('Легенда', fontsize=16)
        ax.axis('off')
        
        # Добавляем текст о SVG файле
        fig.text(0.5, 0.05, f'SVG: {svg_path}', 
                ha='center', fontsize=12, style='italic')
    
    # Сохраняем в PDF
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
