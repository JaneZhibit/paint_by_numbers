import cv2
import numpy as np


def apply_kmeans(image: np.ndarray, config: dict) -> tuple:
    """
    Кластеризация цветов изображения методом K-Means.
    Возвращает: (квантованное_изображение_RGB, матрица_меток, палитра_цветов_RGB)
    """
    k = config.get("colors_count", 16)
    color_space = config.get("color_space", "lab").upper()

    params = config.get("params", {}).get("kmeans", {})
    attempts = params.get("attempts", 10)
    max_iter = params.get("criteria_max_iter", 100)
    eps = params.get("criteria_eps", 0.2)

    # 1. Перевод в выбранное цветовое пространство
    if color_space == 'LAB':
        # Переводим в LAB (cv2 дает диапазон 0-255 для всех каналов)
        working_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        # Уменьшаем масштаб L-канала, чтобы цвета (a, b) имели больший вес при кластеризации
        working_image[:, :, 0] *= (100.0 / 255.0)
    else:
        # Для RGB просто переводим во float32
        working_image = image.astype(np.float32)

    # 2. Подготовка данных для K-Means
    h, w, c = working_image.shape
    pixel_data = working_image.reshape((-1, 3))

    # 3. Настройка критериев и запуск алгоритма
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    _, labels, centers = cv2.kmeans(
        pixel_data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    # 4. Собираем квантованное изображение из палитры (пока во float32)
    quantized_pixels = centers[labels.flatten()]
    quantized_image = quantized_pixels.reshape((h, w, c))

    # 5. Возврат в RGB
    if color_space == 'LAB':
        # Возвращаем L-каналу его cv2-диапазон (0-255)
        quantized_image[:, :, 0] *= (255.0 / 100.0)
        # Ограничиваем значения, чтобы при переводе в uint8 не было переполнения
        quantized_image = np.clip(quantized_image, 0, 255).astype(np.uint8)
        quantized_image_rgb = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)

        # То же самое делаем для палитры (centers), чтобы UI и экспорт получили правильные RGB цвета
        centers_lab = centers.copy()
        centers_lab[:, 0] *= (255.0 / 100.0)
        centers_lab = np.clip(centers_lab, 0, 255).astype(np.uint8)
        palette_rgb = cv2.cvtColor(centers_lab.reshape(1, k, 3), cv2.COLOR_LAB2RGB).reshape(k, 3)
    else:
        # Для RGB просто клипаем и переводим в uint8
        quantized_image_rgb = np.clip(quantized_image, 0, 255).astype(np.uint8)
        palette_rgb = np.clip(centers, 0, 255).astype(np.uint8)

    # Возвращаем квантованную картинку, 2D карту индексов и палитру RGB
    return quantized_image_rgb, labels.reshape((h, w)), palette_rgb


def apply_quantization(image: np.ndarray, config: dict) -> tuple:
    """Диспетчер для этапа квантования."""
    method = config.get("method", "kmeans")

    if method == "kmeans":
        return apply_kmeans(image, config)
    else:
        raise ValueError(f"Неизвестный метод квантования: {method}")