import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def calculate_physical_metrics(img_shape: tuple, config: dict) -> tuple:
    """
    Шаг А: Перевод физических размеров холста и кисти в пиксели.
    Возвращает (min_brush_px, min_area_px).
    """
    canvas_config = config.get("canvas", {})
    canvas_w_mm = canvas_config.get("width_mm", 300)
    canvas_h_mm = canvas_config.get("height_mm", 400)
    min_brush_mm = canvas_config.get("min_brush_mm", 3.0)

    img_h, img_w = img_shape[:2]

    # Считаем количество пикселей в одном миллиметре.
    # Берем минимальное значение между шириной и высотой,
    # чтобы гарантированно не сделать кисть слишком маленькой.
    px_per_mm_w = img_w / canvas_w_mm
    px_per_mm_h = img_h / canvas_h_mm
    px_per_mm = min(px_per_mm_w, px_per_mm_h)

    # Переводим размер кисти в пиксели (минимум 1px)
    min_brush_px = max(1, int(min_brush_mm * px_per_mm))

    # Минимальная площадь пятна в пикселях.
    # Считаем, что кисточка оставляет пятно, близкое к квадрату.
    min_area_px = min_brush_px ** 2

    logger.info(f"Физика: 1 мм = {px_per_mm:.2f} px. Кисть: {min_brush_px} px, Мин. площадь: {min_area_px} px^2")

    return min_brush_px, min_area_px


def remove_small_regions(labels: np.ndarray, min_area_px: int, palette: np.ndarray,
                         contrast_threshold: float = 80.0) -> np.ndarray:
    """
    Шаг С: Умное удаление мелких пятен.
    Если пятно меньше min_area_px, но сильно контрастирует с фоном (например, глаза),
    оно НЕ удаляется!
    """
    out_labels = labels.copy()
    connectivity = 8
    total_removed = 0

    for pass_idx in range(2):
        removed_in_pass = 0
        unique_classes = np.unique(out_labels)

        for cls in unique_classes:
            mask = np.uint8(out_labels == cls)
            num_components, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

            for i in range(1, num_components):
                area = stats[i, cv2.CC_STAT_AREA]

                if area < min_area_px:
                    island_mask = np.uint8(comp_labels == i)
                    dilated = cv2.dilate(island_mask, np.ones((3, 3), np.uint8))
                    boundary = dilated - island_mask

                    neighbor_labels = out_labels[boundary == 1]
                    neighbor_labels = neighbor_labels[neighbor_labels != cls]

                    if len(neighbor_labels) > 0:
                        best_neighbor_cls = np.bincount(neighbor_labels).argmax()

                        # --- НАШ НОВЫЙ ИНТЕЛЛЕКТ: ПРОВЕРКА КОНТРАСТА ---
                        # Получаем RGB цвета (переводим в float для расчетов)
                        color_self = palette[cls].astype(np.float32)
                        color_neighbor = palette[best_neighbor_cls].astype(np.float32)

                        # Считаем Евклидово расстояние между цветами
                        color_distance = np.linalg.norm(color_self - color_neighbor)

                        # Если контраст выше порога - это важная деталь, пропускаем удаление!
                        if color_distance > contrast_threshold:
                            continue

                        out_labels[island_mask == 1] = best_neighbor_cls
                        removed_in_pass += 1

        total_removed += removed_in_pass
        if removed_in_pass == 0:
            break

    logger.info(f"Удалено мелких зон (пылинок): {total_removed}")
    return out_labels


def apply_postprocessing(labels: np.ndarray, palette: np.ndarray, config: dict) -> tuple:
    min_brush_px, min_area_px = calculate_physical_metrics(labels.shape, config)
    contrast_thresh = config.get("postprocessing", {}).get("contrast_threshold", 60.0)

    # Шаг B: Сглаживание границ (Убираем "усы" толщиной в 1 пиксель)
    labels_uint8 = labels.astype(np.uint8)
    ksize = min_brush_px if min_brush_px % 2 != 0 else min_brush_px + 1
    ksize = max(3, ksize)
    smoothed_labels = cv2.medianBlur(labels_uint8, ksize)

    # Шаг C: Удаление островков меньше площади кисти (с защитой контраста)
    final_labels = remove_small_regions(smoothed_labels, min_area_px, palette, contrast_threshold=contrast_thresh)

    # Возврат в RGB для отрисовки
    h, w = final_labels.shape
    final_image_rgb = palette[final_labels.flatten()].reshape((h, w, 3))
    return final_labels, final_image_rgb
