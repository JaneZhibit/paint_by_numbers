import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def calculate_physical_metrics(img_shape: tuple, config: dict) -> tuple:
    """Возвращает словари с размерами кистей и минимальных площадей в пикселях."""
    canvas_config = config.get("canvas", {})
    canvas_w_mm = canvas_config.get("width_mm", 300)
    canvas_h_mm = canvas_config.get("height_mm", 400)

    img_h, img_w = img_shape[:2]
    px_per_mm = min(img_w / canvas_w_mm, img_h / canvas_h_mm)

    brush_mm = config.get("postprocessing", {}).get("brush_sizes_mm", {})

    brush_px = {
        "background": max(1, int(brush_mm.get("background", 4.0) * px_per_mm)),
        "foreground": max(1, int(brush_mm.get("foreground", 2.0) * px_per_mm)),
        "face": max(1, int(brush_mm.get("face_keypoints", 1.0) * px_per_mm))
    }

    area_px = {k: v ** 2 for k, v in brush_px.items()}

    logger.info(
        f"Физика кистей (px): BG={brush_px['background']}, FG={brush_px['foreground']}, Face={brush_px['face']}")
    return brush_px, area_px


def remove_small_regions(labels: np.ndarray, brush_px: dict, area_px: dict,
                         palette: np.ndarray, config: dict,
                         fg_mask: np.ndarray, protection_mask: np.ndarray) -> np.ndarray:
    post_config = config.get("postprocessing", {})
    contrast_threshold = post_config.get("contrast_threshold", 60.0)
    grow_high_contrast = post_config.get("grow_high_contrast", True)

    out_labels = labels.copy()
    total_removed = 0
    total_grown = 0

    palette_img = palette.reshape(1, -1, 3).astype(np.uint8)
    palette_lab = cv2.cvtColor(palette_img, cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    palette_lab[:, 0] *= (100.0 / 255.0)

    for pass_idx in range(2):
        removed_in_pass = 0

        for cls in np.unique(out_labels):
            mask = np.uint8(out_labels == cls)
            num_components, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, num_components):
                area = stats[i, cv2.CC_STAT_AREA]

                # Вырезаем BBox зоны для быстрого поиска пересечений (оптимизация)
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1]
                island_slice = np.uint8(comp_labels[y:y + h, x:x + w] == i)
                prot_slice = protection_mask[y:y + h, x:x + w]
                fg_slice = fg_mask[y:y + h, x:x + w]

                # ДИНАМИЧЕСКИЙ ВЫБОР ПЛОЩАДИ: Где находится эта пылинка?
                overlap_face = np.any((island_slice == 1) & (prot_slice == 255))
                overlap_fg = np.any((island_slice == 1) & (fg_slice == 255))

                if overlap_face:
                    req_area = area_px["face"]
                    req_brush = brush_px["face"]
                elif overlap_fg:
                    req_area = area_px["foreground"]
                    req_brush = brush_px["foreground"]
                else:
                    req_area = area_px["background"]
                    req_brush = brush_px["background"]

                # Если зона меньше ТРЕБУЕМОГО для этой области размера
                if area < req_area:
                    island_mask = np.uint8(comp_labels == i)
                    dilated = cv2.dilate(island_mask, np.ones((3, 3), np.uint8))
                    boundary = dilated - island_mask

                    neighbor_labels = out_labels[boundary == 1]
                    neighbor_labels = neighbor_labels[neighbor_labels != cls]

                    if len(neighbor_labels) > 0:
                        unique_neighbors, counts = np.unique(neighbor_labels, return_counts=True)
                        best_cost = float('inf')
                        best_neighbor_cls = unique_neighbors[0]

                        for n_cls, count in zip(unique_neighbors, counts):
                            color_dist = np.linalg.norm(palette_lab[cls] - palette_lab[n_cls])
                            cost = color_dist / (np.sqrt(count) + 1e-5)
                            if cost < best_cost:
                                best_cost = cost
                                best_neighbor_cls = n_cls

                        # Защита контраста
                        color_distance = np.linalg.norm(palette_lab[cls] - palette_lab[best_neighbor_cls])

                        if color_distance > contrast_threshold:
                            if grow_high_contrast:
                                # Отращиваем ядром ТРЕБУЕМОГО размера
                                grow_ksize = req_brush if req_brush % 2 != 0 else req_brush + 1
                                grow_ksize = max(3, grow_ksize)
                                grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_ksize, grow_ksize))

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

    return out_labels


def apply_postprocessing(labels: np.ndarray, palette: np.ndarray, config: dict,
                         fg_mask: np.ndarray, protection_mask: np.ndarray) -> tuple:
    brush_px, area_px = calculate_physical_metrics(labels.shape, config)

    # 1. ДИНАМИЧЕСКОЕ СГЛАЖИВАНИЕ (Spatially Varying Median Blur)
    labels_uint8 = labels.astype(np.uint8)

    k_bg = max(3, brush_px["background"] if brush_px["background"] % 2 != 0 else brush_px["background"] + 1)
    k_fg = max(3, brush_px["foreground"] if brush_px["foreground"] % 2 != 0 else brush_px["foreground"] + 1)
    k_face = max(3, brush_px["face"] if brush_px["face"] % 2 != 0 else brush_px["face"] + 1)

    blur_bg = cv2.medianBlur(labels_uint8, k_bg)
    blur_fg = cv2.medianBlur(labels_uint8, k_fg)
    blur_face = cv2.medianBlur(labels_uint8, k_face)

    # Собираем "Слоеный пирог" сглаживания
    smoothed_labels = blur_bg.copy()
    smoothed_labels[fg_mask == 255] = blur_fg[fg_mask == 255]
    smoothed_labels[protection_mask == 255] = blur_face[protection_mask == 255]

    # 2. ДИНАМИЧЕСКОЕ УДАЛЕНИЕ ПЫЛИНОК
    final_labels = remove_small_regions(smoothed_labels, brush_px, area_px, palette, config, fg_mask, protection_mask)

    h, w = final_labels.shape
    final_image_rgb = palette[final_labels.flatten()].reshape((h, w, 3))
    return final_labels, final_image_rgb
