import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image


def generate_legend(labels: list, config, save_path: str) -> np.ndarray:
    """
    Генерирует легенду с номерами цветов и их RGB значениями.
    
    Args:
        labels: list[LabelInfo] - список информации о размещении меток
        config: dict - конфигурация
        save_path: str - путь для сохранения PNG файла
    
    Returns:
        np.ndarray (RGB) - легенда как изображение
    """
    # Собираем уникальные color_number → rgb
    color_map = {}
    for label in labels:
        if label.color_number not in color_map:
            color_map[label.color_number] = label.rgb
    
    # Сортируем по color_number
    sorted_colors = sorted(color_map.items())
    
    num_colors = len(sorted_colors)
    
    # Определяем количество колонок (2 если > 8 цветов, иначе 1)
    num_cols = 2 if num_colors > 8 else 1
    num_rows = (num_colors + num_cols - 1) // num_cols
    
    # Размеры элементов
    square_size = 20
    text_offset = 30
    row_height = 30
    col_width = 150
    margin = 10
    
    # Общие размеры
    fig_width = col_width * num_cols + margin * 2
    fig_height = row_height * num_rows + margin * 2
    
    # Создаём фигуру
    fig, ax = plt.subplots(figsize=(fig_width / 100, fig_height / 100), dpi=100)
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Рисуем элементы легенды
    for idx, (color_number, rgb) in enumerate(sorted_colors):
        # Определяем позицию (строка, колонка)
        row = idx % num_rows
        col = idx // num_rows
        
        x = margin + col * col_width
        y = fig_height - margin - (row + 1) * row_height
        
        # Нормализуем RGB в диапазон [0, 1] для matplotlib
        rgb_normalized = tuple(c / 255.0 for c in rgb)
        
        # Рисуем цветной квадрат
        rect = patches.Rectangle(
            (x, y),
            square_size,
            square_size,
            linewidth=1,
            edgecolor='black',
            facecolor=rgb_normalized
        )
        ax.add_patch(rect)
        
        # Добавляем текст с номером
        text_x = x + square_size + 5
        text_y = y + square_size / 2
        ax.text(
            text_x,
            text_y,
            f'No {color_number}',
            fontsize=10,
            verticalalignment='center',
            color='black'
        )
    
    # Сохраняем в файл
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    
    # Читаем обратно как np.ndarray
    img = Image.open(save_path).convert('RGB')
    result = np.array(img, dtype=np.uint8)
    
    plt.close(fig)
    
    return result
