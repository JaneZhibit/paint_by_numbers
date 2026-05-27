import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def calculate_dynamic_maps(
    img_shape: tuple,
    depth_map: np.ndarray,
    protection_mask: np.ndarray,
    config: dict,
    fg_mask: np.ndarray,
) -> tuple:
    """Генерирует 2D-карты требуемого размера кисти с плавным переходом по глубине, но жестко защищает Foreground."""
    canvas_config = config.get("canvas", {})
    canvas_w_mm = canvas_config.get("width_mm", 300)
    canvas_h_mm = canvas_config.get("height_mm", 400)

    h, w = img_shape[:2]
    px_per_mm = min(w / canvas_w_mm, h / canvas_h_mm)

    post_config = config.get("postprocessing", {})
    use_depth = post_config.get("use_depth_morphology", True)

    # Читаем размеры кистей из конфига. Если `brush_sizes_mm` ещё не завели,
    # используем значения из предыдущей версии конфига как fallback.
    brush_mm = post_config.get("brush_sizes_mm", {})
    bg_max_mm = brush_mm.get("background", post_config.get("morphology_max_brush_mm", 3.5))
    fg_mm = brush_mm.get("foreground", post_config.get("morphology_min_brush_mm", 1.5))
    face_mm = brush_mm.get("face_keypoints", post_config.get("face_keypoints_mm", 1.0))

    max_px = max(3, int(bg_max_mm * px_per_mm))
    min_px = max(3, int(fg_mm * px_per_mm))
    face_px = max(3, int(face_mm * px_per_mm))

    # Делаем нечетными для медианного фильтра
    if max_px % 2 == 0:
        max_px += 1
    if min_px % 2 == 0:
        min_px += 1
    if face_px % 2 == 0:
        face_px += 1

    # 1) Фон — плавно по глубине (или просто max_px)
    if use_depth and depth_map is not None:
        brush_float = max_px - (depth_map / 255.0) * (max_px - min_px)
        brush_map_px = np.round(brush_float).astype(np.uint16)
        brush_map_px = brush_map_px + (1 - brush_map_px % 2)
    else:
        brush_map_px = np.full((h, w), max_px, dtype=np.uint16)

    # 2) Жесткий приоритет: Foreground всегда тонкой кистью
    brush_map_px[fg_mask == 255] = min_px

    # 3) Абсолютный приоритет: лица
    brush_map_px[protection_mask == 255] = face_px

    area_map_px = brush_map_px ** 2

    logger.info(
        f"Морфология кистей: Фон до {int(np.max(brush_map_px))}px, "
        f"Объекты {min_px}px, Лица {face_px}px."
    )
    return brush_map_px, area_map_px


def remove_small_regions(
    labels: np.ndarray,
    brush_map_px: np.ndarray,
    area_map_px: np.ndarray,
    palette: np.ndarray,
    config: dict,
    fg_mask: np.ndarray,
    protection_mask: np.ndarray,
) -> np.ndarray:

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

                x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1]
                island_slice = np.uint8(comp_labels[y:y + h, x:x + w] == i)

                local_areas = area_map_px[y:y + h, x:x + w][island_slice == 1]
                local_brushes = brush_map_px[y:y + h, x:x + w][island_slice == 1]

                req_area = np.median(local_areas)
                req_brush = int(np.median(local_brushes))

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

                        color_distance = np.linalg.norm(palette_lab[cls] - palette_lab[best_neighbor_cls])

                        if color_distance > contrast_threshold:
                            if grow_high_contrast:
                                grow_ksize = req_brush if req_brush % 2 != 0 else req_brush + 1
                                grow_ksize = max(3, grow_ksize)
                                grow_kernel = cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (grow_ksize, grow_ksize)
                                )

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
    return out_labels


def apply_postprocessing(
    labels: np.ndarray,
    palette: np.ndarray,
    config: dict,
    fg_mask: np.ndarray,
    protection_mask: np.ndarray,
    depth_map: np.ndarray,
) -> tuple:

    brush_map_px, area_map_px = calculate_dynamic_maps(
        labels.shape, depth_map, protection_mask, config, fg_mask
    )

    labels_uint8 = labels.astype(np.uint8)
    smoothed_labels = labels_uint8.copy()

    unique_brushes = np.unique(brush_map_px)
    for k in unique_brushes:
        blurred = cv2.medianBlur(labels_uint8, int(k))
        smoothed_labels[brush_map_px == k] = blurred[brush_map_px == k]

    final_labels = remove_small_regions(
        smoothed_labels, brush_map_px, area_map_px, palette, config, fg_mask, protection_mask
    )

    h, w = final_labels.shape
    final_image_rgb = palette[final_labels.flatten()].reshape((h, w, 3))
    return final_labels, final_image_rgb
