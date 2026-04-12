import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Загружает шрифт с fallback для разных платформ."""
    candidates = []
    if sys.platform == 'win32':
        candidates = ['arial.ttf', 'C:/Windows/Fonts/arial.ttf']
    elif sys.platform == 'darwin':
        candidates = ['/System/Library/Fonts/Helvetica.ttc',
                      '/Library/Fonts/Arial.ttf']
    else:  # Linux
        candidates = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                      '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                      '/usr/share/fonts/TTF/DejaVuSans.ttf']
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def add_numbers(contours_img: np.ndarray, labels: list) -> np.ndarray:
    """
    Добавляет номера цветов на изображение контуров на основе готового списка LabelInfo.
    
    Args:
        contours_img: np.ndarray (RGB) - изображение с контурами
        labels: list[LabelInfo] - список информации о размещении меток
    
    Returns:
        np.ndarray (RGB) - изображение с добавленными номерами
    """
    # Копируем изображение, чтобы не мутировать оригинал
    img_copy = contours_img.copy()
    
    # Конвертируем в PIL Image для рисования
    pil_img = Image.fromarray(img_copy, mode='RGB')
    draw = ImageDraw.Draw(pil_img)
    
    # Проходим по меткам и рисуем только размещённые
    for label in labels:
        if not label.placed:
            continue
        
        # Загружаем шрифт с fallback для разных платформ
        font = _load_font(label.font_size)
        
        # Рисуем номер черным цветом по центру
        text = str(label.color_number)
        try:
            # Попытка использовать anchor (доступно в Pillow 8.2.0+)
            draw.text((label.cx, label.cy), text, fill=(0, 0, 0), font=font, anchor="mm")
        except TypeError:
            # Fallback для старых версий Pillow
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((label.cx - text_width // 2, label.cy - text_height // 2), text, fill=(0, 0, 0), font=font)
    
    # Конвертируем обратно в np.ndarray
    result = np.array(pil_img, dtype=np.uint8)
    return result
