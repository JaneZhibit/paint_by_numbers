import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def find_optimal_k(pixel_data: np.ndarray, threshold: float) -> int:
    """
    Ищет оптимальное количество цветов на основе процента улучшения точности.
    """
    # Выбираем случайную подвыборку пикселей для ускорения расчетов
    sample_size = 60000
    if pixel_data.shape[0] > sample_size:
        indices = np.random.choice(pixel_data.shape[0], sample_size, replace=False)
        sample_pixels = pixel_data[indices]
    else:
        sample_pixels = pixel_data

    prev_inertia = None
    best_k = 4

    # Критерии для быстрого поиска в цикле
    fast_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    logger.info(f"Запуск автоподбора цветов (порог: {threshold * 100}%)...")

    # Идем с шагом 2 или 4 для скорости, либо по порядку
    for k in range(4, 33, 2):
        inertia, _, _ = cv2.kmeans(
            sample_pixels, k, None, fast_criteria, 1, cv2.KMEANS_PP_CENTERS
        )

        if prev_inertia is not None:
            improvement = (prev_inertia - inertia) / prev_inertia
            logger.info(f"K={k}, Улучшение: {improvement:.2%}")

            if improvement < threshold:
                logger.info(f"Достигнут порог. Оптимальное K: {best_k}")
                return best_k

        prev_inertia = inertia
        best_k = k

    return best_k


def apply_kmeans(image: np.ndarray, config: dict) -> tuple:
    q_config = config.get("quantization", {})
    color_space = q_config.get("color_space", "lab").upper()

    params = q_config.get("params", {}).get("kmeans", {})
    attempts = params.get("attempts", 10)
    max_iter = params.get("criteria_max_iter", 100)
    eps = params.get("criteria_eps", 0.2)

    # 1. Подготовка данных
    if color_space == 'LAB':
        working_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        working_image[:, :, 0] *= (100.0 / 255.0)
    else:
        working_image = image.astype(np.float32)

    h, w, c = working_image.shape
    pixel_data = working_image.reshape((-1, 3))

    # 2. Автоматический подбор K, если включен
    if q_config.get("auto_colors", False):
        k = find_optimal_k(pixel_data, q_config.get("auto_threshold", 0.05))
    else:
        k = q_config.get("colors_count", 16)

    # 3. Финальный запуск K-Means с полными параметрами
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    _, labels, centers = cv2.kmeans(
        pixel_data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    # 4. Сборка результата
    quantized_pixels = centers[labels.flatten()]
    quantized_image = quantized_pixels.reshape((h, w, c))

    if color_space == 'LAB':
        quantized_image[:, :, 0] *= (255.0 / 100.0)
        quantized_image = np.clip(quantized_image, 0, 255).astype(np.uint8)
        quantized_image_rgb = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)

        centers_lab = centers.copy()
        centers_lab[:, 0] *= (255.0 / 100.0)
        centers_lab = np.clip(centers_lab, 0, 255).astype(np.uint8)
        palette_rgb = cv2.cvtColor(centers_lab.reshape(1, k, 3), cv2.COLOR_LAB2RGB).reshape(k, 3)
    else:
        quantized_image_rgb = np.clip(quantized_image, 0, 255).astype(np.uint8)
        palette_rgb = np.clip(centers, 0, 255).astype(np.uint8)

    return quantized_image_rgb, labels.reshape((h, w)), palette_rgb


def apply_quantization(image: np.ndarray, config: dict) -> tuple:
    method = config.get("quantization", {}).get("method", "kmeans")
    if method == "kmeans":
        return apply_kmeans(image, config)
    else:
        raise ValueError(f"Unknown method: {method}")
