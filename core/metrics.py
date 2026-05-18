import cv2
import numpy as np


def get_image_complexity(image: np.ndarray) -> float:
    """Возвращает индекс сложности текстуры изображения."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_edge_preservation(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Возвращает процент сохранившихся границ (0.0 - 1.0).
    Полезно для отладки: если метрика < 0.5, значит мы превратили картинку в "мыло".
    """

    def get_edges(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Выделяем границы Собелем
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx ** 2 + sobely ** 2)

    edges_orig = get_edges(original)
    edges_proc = get_edges(processed)

    # Сумма градиентов обработанного / Сумма градиентов оригинала
    # Чем ближе к 1.0, тем больше границ сохранилось
    score = np.sum(edges_proc) / (np.sum(edges_orig) + 1e-5)
    return min(score, 1.0)