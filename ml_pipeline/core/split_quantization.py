import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def get_super_category(label: str) -> str:
    """Группирует сотни классов в 5 логических макро-категорий."""
    lbl = label.lower().replace("[fg]", "").replace("[bg]", "").strip()

    flora = ["tree", "grass", "plant", "field", "flower", "earth", "dirt", "mountain", "hill", "land", "rock", "sand",
             "nature", "leaf"]
    sky_water = ["sky", "water", "sea", "lake", "river", "cloud", "ocean", "pool", "waterfall"]
    architecture = ["building", "wall", "road", "bridge", "ceiling", "floor", "house", "sidewalk", "street", "path",
                    "architecture", "city", "tower", "fence"]
    foreground = ["person", "dog", "cat", "bear", "bird", "horse", "animal", "backpack", "car", "bicycle", "boat",
                  "chair", "table", "bottle"]

    if any(f in lbl for f in flora): return "flora"
    if any(s in lbl for s in sky_water): return "sky_water"
    if any(a in lbl for a in architecture): return "architecture"
    if any(fg in lbl for fg in foreground): return "foreground"

    return "other"


def find_optimal_k(pixel_data: np.ndarray, threshold: float, config: dict, max_k: int = 16) -> int:
    """Ищет оптимальное количество цветов на основе Marginal Gain (прирост точности)."""
    if len(pixel_data) < 10:
        return 1
    if max_k <= 1:
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

    # Идем с шагом 2 для скорости
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
    total_colors = q_config.get("total_colors", 32)

    h, w = image.shape[:2]

    # 1. Формируем маски по Супер-Категориям
    super_masks = {
        "foreground": np.zeros((h, w), dtype=np.uint8),
        "flora": np.zeros((h, w), dtype=np.uint8),
        "sky_water": np.zeros((h, w), dtype=np.uint8),
        "architecture": np.zeros((h, w), dtype=np.uint8),
        "other": np.zeros((h, w), dtype=np.uint8)
    }

    for obj in semantic_objects:
        if obj.get("is_foreground", False):
            sc = "foreground"
        else:
            sc = get_super_category(obj["label"])
        super_masks[sc][obj["mask"] == 255] = 255

    # Собираем пиксели, не попавшие ни в одну маску объектов.
    all_assigned = np.zeros((h, w), dtype=np.uint8)
    for mask in super_masks.values():
        all_assigned = cv2.bitwise_or(all_assigned, mask)
    unassigned_mask = cv2.bitwise_not(all_assigned)
    super_masks["other"] = cv2.bitwise_or(super_masks["other"], unassigned_mask)

    # 2. Перевод в LAB с нормализацией L-канала (0-100)
    working_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    working_img[:, :, 0] *= (100.0 / 255.0)

    # 3. Извлекаем данные пикселей для каждой категории
    sc_data = {}
    for sc, mask in super_masks.items():
        data = working_img[mask == 255]
        if len(data) > 0:
            sc_data[sc] = data

    # 4. ДИНАМИЧЕСКОЕ БЮДЖЕТИРОВАНИЕ ЦВЕТОВ (AUTO-K)
    sc_budgets = {}

    if q_config.get("auto_colors", True):
        logger.info("Автоподбор цветов для супер-категорий (Marginal Gain)...")
        fg_thresh = q_config.get("fg_threshold", 0.02)
        bg_thresh = q_config.get("bg_threshold", 0.06)

        current_total = 0
        for sc, data in sc_data.items():
            thresh = fg_thresh if sc == "foreground" else bg_thresh
            max_allowed = 16 if sc == "foreground" else 10

            optimal_k = find_optimal_k(data, thresh, config, max_k=max_allowed)
            sc_budgets[sc] = optimal_k
            current_total += optimal_k

        if current_total > total_colors and current_total > 0:
            logger.info(f"Сумма ({current_total}) превышает лимит {total_colors}. Пропорциональное сжатие...")
            allocated = 0
            for sc in sc_budgets:
                sc_budgets[sc] = max(1, int(round((sc_budgets[sc] / current_total) * total_colors)))
                allocated += sc_budgets[sc]

            if allocated != total_colors:
                largest_sc = max(sc_budgets, key=sc_budgets.get)
                sc_budgets[largest_sc] += (total_colors - allocated)
                sc_budgets[largest_sc] = max(1, sc_budgets[largest_sc])
    else:
        logger.info("Auto-K отключен. Используем жесткое распределение бюджета.")
        fg_budget = q_config.get("foreground_colors", 12)
        bg_budget = total_colors - fg_budget

        bg_categories = [sc for sc in sc_data.keys() if sc != "foreground"]

        if "foreground" in sc_data:
            sc_budgets["foreground"] = fg_budget
        if len(bg_categories) > 0:
            colors_per_bg = max(1, bg_budget // len(bg_categories))
            for sc in bg_categories:
                sc_budgets[sc] = colors_per_bg

    # 5. ЗАПУСК K-MEANS ДЛЯ КАЖДОЙ КАТЕГОРИИ
    sc_labels = {}
    sc_centers = {}

    for sc, budget in sc_budgets.items():
        logger.info(f" -> Категория [{sc.upper()}]: Выделено {budget} цветов.")
        labels, centers = run_kmeans(sc_data[sc], budget, config)
        sc_labels[sc] = labels
        sc_centers[sc] = centers

    # 6. ГЛОБАЛЬНАЯ ДЕДУПЛИКАЦИЯ В ПРОСТРАНСТВЕ LAB
    final_palette = []
    mapped_labels_dict = {}
    MERGE_THRESHOLD = 8.0

    for sc, centers in sc_centers.items():
        mapped_indices = []
        for center in centers:
            if len(final_palette) > 0:
                dists = np.linalg.norm(np.array(final_palette) - center, axis=1)
                min_dist_idx = int(np.argmin(dists))

                if dists[min_dist_idx] < MERGE_THRESHOLD:
                    mapped_indices.append(min_dist_idx)
                else:
                    final_palette.append(center)
                    mapped_indices.append(len(final_palette) - 1)
            else:
                final_palette.append(center)
                mapped_indices.append(0)
        mapped_labels_dict[sc] = mapped_indices

    final_palette = np.array(final_palette, dtype=np.float32)
    logger.info(f"Уникальных цветов в итоговой палитре (после слияния дубликатов): {len(final_palette)}")

    # 7. СБОРКА МАТРИЦЫ ИНДЕКСОВ
    final_labels_2d = np.zeros((h, w), dtype=np.int32)

    for sc, labels in sc_labels.items():
        if len(labels) > 0:
            mask = super_masks[sc]
            mapped_sc_labels = np.array(mapped_labels_dict[sc])[labels]
            final_labels_2d[mask == 255] = mapped_sc_labels

    # 8. ВОЗВРАТ ПАЛИТРЫ И КАРТИНКИ В RGB
    centers_lab = final_palette.copy()
    centers_lab[:, 0] *= (255.0 / 100.0)
    centers_lab = np.clip(centers_lab, 0, 255).astype(np.uint8)

    palette_rgb = cv2.cvtColor(centers_lab.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    quantized_img_rgb = palette_rgb[final_labels_2d.flatten()].reshape((h, w, 3))

    return quantized_img_rgb, final_labels_2d, palette_rgb