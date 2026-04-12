from dataclasses import dataclass
import numpy as np


@dataclass
class LabelInfo:
    comp_id: int
    cluster_id: int
    color_number: int
    cx: int           # финальная x-координата метки
    cy: int           # финальная y-координата метки
    font_size: int
    rgb: tuple        # (R, G, B) цвета зоны
    placed: bool      # False = не удалось разместить без перекрытия


def compute_label_layout(output_components, color_index_map, comp_to_cluster,
                         comp_sizes, final_colors, config) -> list[LabelInfo]:
    """
    Вычисляет оптимальное размещение меток с номерами цветов.
    
    Args:
        output_components: np.ndarray - маска компонент
        color_index_map: dict[int, int] - маппинг cluster_id → номер цвета
        comp_to_cluster: dict[int, int] - маппинг component_id → cluster_id
        comp_sizes: np.ndarray - размеры компонент в пикселях
        final_colors: dict[int, tuple] - маппинг component_id → RGB цвет
        config: dict - конфигурация
    
    Returns:
        list[LabelInfo] - список информации о размещении меток
    """
    numbering_config = config['algorithm']['numbering']
    min_number_area_px = numbering_config['min_number_area_px']
    font_scale_factor = numbering_config['font_scale_factor']
    
    # Собираем компоненты, которые достаточно большие
    valid_components = []
    for comp_id in range(len(comp_sizes)):
        if comp_id == 0 or comp_sizes[comp_id] < min_number_area_px:
            continue
        
        cluster_id = comp_to_cluster.get(int(comp_id))
        if cluster_id is None:
            continue
        
        color_number = color_index_map.get(cluster_id)
        if color_number is None:
            continue
        
        valid_components.append((comp_id, cluster_id, color_number))
    
    # Сортируем по убыванию размера (большие зоны размещаем первыми)
    valid_components.sort(key=lambda x: comp_sizes[x[0]], reverse=True)
    
    # Список занятых bbox'ов: [(x1, y1, x2, y2), ...]
    occupied_bboxes = []
    
    # Результирующий список
    label_infos = []
    
    for comp_id, cluster_id, color_number in valid_components:
        # Находим центроид
        ys, xs = np.where(output_components == comp_id)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        
        # Проверяем, что centroid находится внутри компоненты
        if output_components[cy, cx] != comp_id:
            distances = np.sqrt((ys - cy)**2 + (xs - cx)**2)
            best_idx = np.argmin(distances)
            cy, cx = int(ys[best_idx]), int(xs[best_idx])
        
        # Рассчитываем размер шрифта
        font_size = max(8, int(np.sqrt(comp_sizes[comp_id]) * font_scale_factor))
        
        # Считаем bbox метки
        # Приближение: ширина ≈ 2 символа × font_size, высота ≈ font_size
        bbox_width = 2 * font_size
        bbox_height = font_size
        
        x1 = cx - bbox_width // 2
        y1 = cy - bbox_height // 2
        x2 = x1 + bbox_width
        y2 = y1 + bbox_height
        
        bbox = (x1, y1, x2, y2)
        
        # Проверяем пересечение с занятыми bbox'ами
        placed = True
        for occupied_bbox in occupied_bboxes:
            if _bboxes_intersect(bbox, occupied_bbox):
                placed = False
                break
        
        # Если успешно размещена, добавляем в список занятых
        if placed:
            occupied_bboxes.append(bbox)
        
        # Получаем RGB цвет
        rgb = final_colors.get(comp_id, (0, 0, 0))
        
        # Создаём LabelInfo
        label_info = LabelInfo(
            comp_id=comp_id,
            cluster_id=cluster_id,
            color_number=color_number,
            cx=cx,
            cy=cy,
            font_size=font_size,
            rgb=rgb,
            placed=placed
        )
        
        label_infos.append(label_info)
    
    return label_infos


def _bboxes_intersect(bbox1, bbox2) -> bool:
    """
    Проверяет пересечение двух bbox'ов.
    
    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)
    
    Returns:
        bool - True если пересекаются
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Проверяем пересечение по осям
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False
    
    return True
