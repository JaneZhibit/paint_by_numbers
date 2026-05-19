import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def apply_watershed_refinement(original_img: np.ndarray, labels: np.ndarray, config: dict) -> np.ndarray:
    """
    Уточнение границ зон с помощью алгоритма Watershed.
    Возвращает обновленную матрицу labels.
    """
    params = config.get("postprocessing", {}).get("watershed_params", {})
    erosion_size = params.get("erosion_size", 3)
    blur_size = params.get("gradient_blur", 5)

    # 1. Готовим "рельеф" (Gradient)
    # Используем морфологический градиент — это разница между расширением и сжатием картинки
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

    # 2. Готовим "маркеры" (Seeds)
    # Нам нужно сжать каждую зону, чтобы маркеры не касались границ
    # Watershed в OpenCV требует int32
    markers = np.int32(labels) + 1  # Сдвигаем на 1, так как 0 в Watershed - это неизвестная зона

    # Сжимаем зоны
    kernel_ero = np.ones((erosion_size, erosion_size), np.uint8)
    # Чтобы не сжимать всё скопом, нужно обрабатывать маркеры аккуратно.
    # Самый быстрый способ — оставить только те пиксели, которые не меняются при эрозии
    eroded_markers = cv2.erode(np.uint8(markers > 0), kernel_ero)
    markers[eroded_markers == 0] = 0

    # 3. Запуск Watershed
    # OpenCV требует трехканальное изображение (BGR/RGB) для работы
    cv2.watershed(original_img, markers)

    # 4. Постобработка результатов Watershed
    # markers теперь содержит:
    # -1 там, где прошли границы
    # [1...N] индексы цветов

    # Возвращаем к 0...N-1
    result_labels = markers - 1

    # Убираем границы (-1), заменяя их цветом соседей (через дилатацию)
    # Это важно, так как плоттер не должен видеть "пустоту" между зонами
    if np.any(result_labels < 0):
        mask_minus_one = np.uint8(result_labels < 0)
        # Расширяем нормальные зоны на место границ -1
        refined = cv2.dilate(result_labels.astype(np.float32), kernel, iterations=1)
        result_labels[mask_minus_one == 1] = refined[mask_minus_one == 1]

    logger.info("Уточнение границ (Watershed) завершено.")
    return result_labels.astype(np.uint8)
