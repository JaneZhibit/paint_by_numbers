import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def calculate_label_placements(high_res_labels: np.ndarray, config: dict) -> list:
    """
    Вычисляет координаты для размещения текста.
    """
    scale_factor = config.get("vectorization", {}).get("scale_factor", 4)
    min_area = 50 * (scale_factor ** 2)

    max_radius = 25 * scale_factor
    min_dist = 4 * scale_factor

    unique_colors = sorted(list(np.unique(high_res_labels)))
    color_to_num = {color: str(i + 1) for i, color in enumerate(unique_colors)}

    placements = []

    for cls in unique_colors:
        color_mask = np.uint8(high_res_labels == cls)

        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            blob_mask = np.uint8(comp_labels[y:y + h, x:x + w] == i)

            # --- ИСПРАВЛЕНИЕ: Добавляем рамку из нулей (толщиной 1px), чтобы края картинки стали "стеной"
            padded_mask = cv2.copyMakeBorder(blob_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

            for attempt in range(5):
                # Считаем дистанцию по padded_mask
                dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)
                _, max_dist, _, max_loc = cv2.minMaxLoc(dist)

                if max_dist < min_dist:
                    break

                    # Так как мы добавили рамку слева и сверху, max_loc сместился на +1.
                # Поэтому при возврате в глобальные координаты мы вычитаем 1:
                center_x = max_loc[0] - 1 + x
                center_y = max_loc[1] - 1 + y

                used_radius = min(max_dist, max_radius)

                placements.append({
                    "text": color_to_num[cls],
                    "x": center_x,
                    "y": center_y,
                    "radius": used_radius
                })

                # Закрашиваем найденную зону на padded_mask для поиска следующих точек
                cv2.circle(padded_mask, max_loc, int(used_radius * 2.5), 0, -1)

    logger.info(f"Сгенерировано {len(placements)} меток для цифр.")
    return placements
