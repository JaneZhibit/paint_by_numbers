import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def upscale_and_smooth_labels(labels: np.ndarray, config: dict) -> np.ndarray:
    vect_config = config.get("vectorization", {})
    scale_factor = vect_config.get("scale_factor", 4)
    # Если хочешь более плавные дуги, увеличивай blur_size (например до 25 или 31)
    blur_size = vect_config.get("blur_size", 15)

    h, w = labels.shape
    new_h, new_w = h * scale_factor, w * scale_factor

    upscaled = cv2.resize(labels, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    unique_classes = np.unique(upscaled)
    num_classes = len(unique_classes)

    smoothed_volume = np.zeros((new_h, new_w, num_classes), dtype=np.float32)

    logger.info(f"Сглаживание контуров в высоком разрешении ({new_w}x{new_h})...")

    for idx, cls in enumerate(unique_classes):
        mask = np.float32(upscaled == cls)
        blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        smoothed_volume[:, :, idx] = blurred

    max_indices = np.argmax(smoothed_volume, axis=2)

    final_high_res_labels = np.zeros_like(max_indices, dtype=np.uint8)
    for idx, cls in enumerate(unique_classes):
        final_high_res_labels[max_indices == idx] = cls

    return final_high_res_labels


def apply_vectorization(labels: np.ndarray, config: dict) -> tuple:
    # Делаем эпсилон микроскопическим, чтобы он только сжимал массив точек, но не гнул линии
    eps_factor = config.get("vectorization", {}).get("epsilon_factor", 0.0001)

    high_res_labels = upscale_and_smooth_labels(labels, config)

    unique_classes = np.unique(high_res_labels)
    polygons = []
    total_contours = 0

    for cls in unique_classes:
        mask = np.uint8(high_res_labels == cls)

        # RETR_CCOMP или RETR_LIST находят контуры. Форма уже идеальная из-за размытия.
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 3:
                continue

            perimeter = cv2.arcLength(cnt, True)
            epsilon = eps_factor * perimeter
            approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx_cnt) >= 3:
                polygons.append({
                    "color_idx": cls,
                    "contour": approx_cnt,
                    "area": cv2.contourArea(approx_cnt)
                })
                total_contours += 1

    logger.info(f"Векторизация завершена. Найдено полигонов: {total_contours}")
    return high_res_labels, polygons


def draw_vectorized_canvas(high_res_labels: np.ndarray) -> np.ndarray:
    """
    Отрисовывает единые математические границы напрямую из растра высокого разрешения.
    Гарантирует отсутствие двойных линий и нахлестов.
    """
    h, w = high_res_labels.shape
    edges = np.zeros((h, w), dtype=np.uint8)

    # 1. Извлекаем границы (скелет) между всеми цветами
    for cls in np.unique(high_res_labels):
        mask = np.uint8(high_res_labels == cls)
        # Уменьшаем маску на 1 пиксель
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        # Граница - это разница между исходной маской и уменьшенной
        boundary = mask - eroded
        # Добавляем границу текущего цвета к общей карте границ
        edges = cv2.bitwise_or(edges, boundary)

    # Утолщаем границы до 2-3 пикселей, чтобы на огромном разрешении их было хорошо видно
    #edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    # 2. Создаем белый холст
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 3. Красим найденные единые границы в черный
    canvas[edges == 1] = (0, 0, 0)

    return canvas
