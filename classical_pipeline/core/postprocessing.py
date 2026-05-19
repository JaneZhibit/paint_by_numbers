import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def calculate_physical_metrics(img_shape: tuple, config: dict) -> tuple:
    canvas_config = config.get("canvas", {})
    canvas_w_mm = canvas_config.get("width_mm", 300)
    canvas_h_mm = canvas_config.get("height_mm", 400)
    min_brush_mm = canvas_config.get("min_brush_mm", 3.0)

    img_h, img_w = img_shape[:2]

    px_per_mm_w = img_w / canvas_w_mm
    px_per_mm_h = img_h / canvas_h_mm
    px_per_mm = min(px_per_mm_w, px_per_mm_h)

    min_brush_px = max(1, int(min_brush_mm * px_per_mm))
    min_area_px = min_brush_px ** 2

    logger.info(f"Физика: 1 мм = {px_per_mm:.2f} px. Кисть: {min_brush_px} px, Мин. площадь: {min_area_px} px^2")

    return min_brush_px, min_area_px


def remove_small_regions(labels: np.ndarray, min_area_px: int, min_brush_px: int, palette: np.ndarray, config: dict) -> np.ndarray:
    """
    Шаг С: Интеллектуальное удаление мелких пятен с поддержкой Color-Weighted Merging
    и Semantic Dilation (динамическое отращивание важных деталей до размера кисти).
    """
    post_config = config.get("postprocessing", {})
    contrast_threshold = post_config.get("contrast_threshold", 60.0)
    merge_strategy = post_config.get("merge_strategy", "color_weighted")
    grow_high_contrast = post_config.get("grow_high_contrast", True)

    out_labels = labels.copy()
    connectivity = 8
    total_removed = 0
    total_grown = 0

    # 1. ПРАВИЛЬНЫЙ ПЕРЕВОД В LAB (с учетом нашей нормализации L-канала!)
    palette_img = palette.reshape(1, -1, 3).astype(np.uint8)
    palette_lab = cv2.cvtColor(palette_img, cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    palette_lab[:, 0] *= (100.0 / 255.0)  # Синхронизация математики с quantization.py

    # 2. ПОДГОТОВКА ЯДРА ДЛЯ ОТРАЩИВАНИЯ
    # Ядро должно быть нечетным, чтобы расширение было симметричным
    grow_ksize = min_brush_px if min_brush_px % 2 != 0 else min_brush_px + 1
    grow_ksize = max(3, grow_ksize) # Защита от слишком мелкой кисти
    grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_ksize, grow_ksize))

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

                        # --- ЛОГИКА СЛИЯНИЯ (MERGE STRATEGY) ---
                        unique_neighbors, counts = np.unique(neighbor_labels, return_counts=True)

                        if merge_strategy == "color_weighted":
                            best_cost = float('inf')
                            best_neighbor_cls = unique_neighbors[0]

                            for n_cls, count in zip(unique_neighbors, counts):
                                color_dist = np.linalg.norm(palette_lab[cls] - palette_lab[n_cls])
                                cost = color_dist / (np.sqrt(count) + 1e-5)

                                if cost < best_cost:
                                    best_cost = cost
                                    best_neighbor_cls = n_cls
                        else:
                            best_neighbor_cls = unique_neighbors[np.argmax(counts)]

                        # --- ЛОГИКА СОХРАНЕНИЯ КОНТРАСТА (SEMANTIC PRESERVATION) ---
                        color_distance = np.linalg.norm(palette_lab[cls] - palette_lab[best_neighbor_cls])

                        if color_distance > contrast_threshold:
                            if grow_high_contrast:
                                expanded_island = cv2.dilate(island_mask, grow_kernel)
                                growth_ring = (expanded_island == 1) & (island_mask == 0)
                                out_labels[growth_ring] = cls
                                total_grown += 1

                            continue

                        out_labels[island_mask == 1] = best_neighbor_cls
                        removed_in_pass += 1

        total_removed += removed_in_pass
        if removed_in_pass == 0:
            break

    logger.info(f"Удалено мелких зон (пылинок): {total_removed}")
    if grow_high_contrast:
        logger.info(f"Искусственно расширено контрастных деталей: {total_grown}")

    return out_labels


def apply_postprocessing(labels: np.ndarray, palette: np.ndarray, config: dict) -> tuple:
    min_brush_px, min_area_px = calculate_physical_metrics(labels.shape, config)

    # Шаг B: Сглаживание границ (Убираем "усы" толщиной в 1 пиксель)
    labels_uint8 = labels.astype(np.uint8)
    # ksize должен быть нечетным
    ksize = min_brush_px if min_brush_px % 2 != 0 else min_brush_px + 1
    ksize = max(3, ksize)
    smoothed_labels = cv2.medianBlur(labels_uint8, ksize)

    # Шаг C: Удаление островков меньше площади кисти (с защитой контраста)
    # Передаем config целиком, чтобы функция имела доступ к новым настройкам
    final_labels = remove_small_regions(smoothed_labels, min_area_px, min_brush_px, palette, config)

    # Возврат в RGB для отрисовки
    h, w = final_labels.shape
    final_image_rgb = palette[final_labels.flatten()].reshape((h, w, 3))
    return final_labels, final_image_rgb
