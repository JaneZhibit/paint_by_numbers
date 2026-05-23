import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def find_optimal_k(pixel_data: np.ndarray, threshold: float, config: dict, max_k: int = 16) -> int:
    """Ищет оптимальное количество цветов на основе Marginal Gain (прирост точности)."""
    if len(pixel_data) < 10:
        return 1

    sample_size = 20000
    if pixel_data.shape[0] > sample_size:
        indices = np.random.choice(pixel_data.shape[0], sample_size, replace=False)
        sample_pixels = pixel_data[indices]
    else:
        sample_pixels = pixel_data

    prev_inertia = None
    best_k = 2
    fast_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Идем с шагом 2
    for k in range(2, max_k + 1, 2):
        actual_k = min(k, len(sample_pixels))
        inertia, _, _ = cv2.kmeans(
            sample_pixels, actual_k, None, fast_criteria, 1, cv2.KMEANS_PP_CENTERS
        )

        if prev_inertia is not None:
            # Считаем, насколько упала ошибка (инерция)
            improvement = (prev_inertia - inertia) / (prev_inertia + 1e-5)

            if improvement < threshold:
                return best_k

        prev_inertia = inertia
        best_k = k

    return best_k


def run_kmeans(pixel_data: np.ndarray, k: int, config: dict) -> tuple:
    """Вспомогательная функция для запуска финального K-Means."""
    if len(pixel_data) == 0:
        return np.array([]), np.array([])

    actual_k = min(k, len(pixel_data))
    params = config.get("quantization", {}).get("params", {}).get("kmeans", {})
    attempts = params.get("attempts", 10)
    max_iter = params.get("criteria_max_iter", 100)
    eps = params.get("criteria_eps", 0.2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    _, labels, centers = cv2.kmeans(
        pixel_data, actual_k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    return labels.flatten(), centers


def apply_split_quantization(image: np.ndarray, semantic_objects: list, config: dict) -> tuple:
    q_config = config.get("quantization", {})
    total_colors = q_config.get("total_colors", 16)

    h, w = image.shape[:2]

    # 1. Формируем маски Foreground и Background
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    for obj in semantic_objects:
        if obj.get("is_foreground", False):
            fg_mask[obj["mask"] == 255] = 255

    bg_mask = cv2.bitwise_not(fg_mask)

    # 2. Перевод в LAB
    working_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    working_img[:, :, 0] *= (100.0 / 255.0)

    # 3. Извлекаем данные
    fg_data = working_img[fg_mask == 255]
    bg_data = working_img[bg_mask == 255]

    # 4. АВТОМАТИЧЕСКОЕ ИЛИ РУЧНОЕ ОПРЕДЕЛЕНИЕ K
    if q_config.get("auto_colors", True):
        logger.info("Автоподбор количества цветов (Marginal Gain)...")
        fg_k = find_optimal_k(fg_data, q_config.get("fg_threshold", 0.02), config, max_k=16) if len(fg_data) > 0 else 0
        bg_k = find_optimal_k(bg_data, q_config.get("bg_threshold", 0.06), config, max_k=10) if len(bg_data) > 0 else 0

        # Корректировка, если мы вылезли за лимит total_colors
        current_total = fg_k + bg_k
        if current_total > total_colors and current_total > 0:
            logger.info(f"Сумма ({current_total}) превышает лимит {total_colors}. Пропорциональное сжатие...")
            fg_k = max(1, int(round((fg_k / current_total) * total_colors)))
            bg_k = max(1, total_colors - fg_k)
    else:
        fg_k = q_config.get("foreground_colors", 12)
        bg_k = q_config.get("background_colors", 4)
        if len(fg_data) == 0: bg_k = total_colors
        if len(bg_data) == 0: fg_k = total_colors

    logger.info(f"Запуск Split K-Means: Foreground ({fg_k} цветов), Background ({bg_k} цветов)")

    # 5. Запускаем K-Means
    fg_labels, fg_centers = run_kmeans(fg_data, fg_k, config)
    bg_labels, bg_centers = run_kmeans(bg_data, bg_k, config)

    # 6. Дедупликация (Объединение похожих цветов)
    final_palette = []
    fg_mapped_indices = []
    bg_mapped_indices = []

    if len(fg_centers) > 0:
        for center in fg_centers:
            final_palette.append(center)
            fg_mapped_indices.append(len(final_palette) - 1)

    MERGE_THRESHOLD = 12.0

    if len(bg_centers) > 0:
        for center in bg_centers:
            if len(final_palette) > 0:
                dists = np.linalg.norm(np.array(final_palette) - center, axis=1)
                min_dist_idx = int(np.argmin(dists))

                if dists[min_dist_idx] < MERGE_THRESHOLD:
                    bg_mapped_indices.append(min_dist_idx)
                else:
                    final_palette.append(center)
                    bg_mapped_indices.append(len(final_palette) - 1)
            else:
                final_palette.append(center)
                bg_mapped_indices.append(len(final_palette) - 1)

    final_palette = np.array(final_palette, dtype=np.float32)
    logger.info(f"Уникальных цветов в итоговой палитре (после слияния дубликатов): {len(final_palette)}")

    # 7. Сборка матрицы индексов
    final_labels_2d = np.zeros((h, w), dtype=np.int32)

    if len(fg_data) > 0:
        mapped_fg_labels = np.array(fg_mapped_indices)[fg_labels]
        final_labels_2d[fg_mask == 255] = mapped_fg_labels

    if len(bg_data) > 0:
        mapped_bg_labels = np.array(bg_mapped_indices)[bg_labels]
        final_labels_2d[bg_mask == 255] = mapped_bg_labels

    # 8. Возврат палитры и картинки в RGB
    centers_lab = final_palette.copy()
    centers_lab[:, 0] *= (255.0 / 100.0)
    centers_lab = np.clip(centers_lab, 0, 255).astype(np.uint8)

    palette_rgb = cv2.cvtColor(centers_lab.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    quantized_img_rgb = palette_rgb[final_labels_2d.flatten()].reshape((h, w, 3))

    return quantized_img_rgb, final_labels_2d, palette_rgb
