import numpy as np
import cv2


def find_optimal_k(preprocessed_img, config):
    """
    Находит оптимальное количество цветов (K) для K-Means используя метод "Локтя" (Elbow method).
    
    Алгоритм:
    1. Уменьшает изображение пропорционально subsample_ratio для быстрого расчета
    2. Переводит в LAB цветовое пространство
    3. Запускает K-Means для разных значений K (от k_min до k_max с шагом k_step)
    4. Вычисляет WCSS (Within-Cluster Sum of Squares) для каждого K
    5. Находит "локоть" кривой WCSS - точку максимального отклонения от линии
    6. Возвращает оптимальное значение K
    """
    logging = config['logging']
    auto_color_config = config['algorithm']['quantizing']['auto_color']
    
    k_min = auto_color_config.get('k_min', 10)
    k_max = auto_color_config.get('k_max', 26)
    k_step = auto_color_config.get('k_step', 2)
    subsample_ratio = auto_color_config.get('subsample_ratio', 0.2)
    
    if logging:
        print('-' * 20, "\nМодуль auto_color (поиск оптимального K)\n", '-' * 20)
        print(f"Диапазон K: [{k_min}, {k_max}], шаг: {k_step}")
        print(f"Коэффициент подвыборки: {subsample_ratio}")
    
    # 1. Уменьшаем изображение для быстрого расчета
    h, w = preprocessed_img.shape[:2]
    new_h = max(1, int(h * np.sqrt(subsample_ratio)))
    new_w = max(1, int(w * np.sqrt(subsample_ratio)))
    
    subsampled_img = cv2.resize(preprocessed_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if logging:
        print(f"Оригинальный размер: {w}x{h}, подвыборка: {new_w}x{new_h}")
    
    # 2. Переводим в LAB и нормализуем L
    img_converted = cv2.cvtColor(subsampled_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_converted[:, :, 0] *= (100.0 / 255.0)
    
    # 3. Подготавливаем массив пикселей
    pixels = img_converted.reshape((-1, 3))
    
    # 4. Запускаем K-Means для разных K и вычисляем WCSS
    wcss_values = []
    k_values = list(range(k_min, k_max + 1, k_step))
    
    if logging:
        print(f"Тестирование K значений: {k_values}")
    
    for k in k_values:
        # Используем облегченные параметры K-Means для скорости
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data=pixels, K=k, bestLabels=None, criteria=criteria,
            attempts=1, flags=cv2.KMEANS_PP_CENTERS
        )
        
        # Вычисляем WCSS: сумма квадратов расстояний от пикселей до их центров
        distances = pixels - centers[labels.flatten()]
        wcss = np.sum(distances ** 2)
        wcss_values.append(wcss)
        
        if logging:
            print(f"  K={k}: WCSS={wcss:.2e}")
    
    # 5. Находим "локоть" кривой WCSS
    optimal_k = _find_elbow(k_values, wcss_values, logging)
    
    if logging:
        print(f"Оптимальное K найдено: {optimal_k}")
        print('-' * 20 + "\n")
    
    return optimal_k


def _find_elbow(k_values, wcss_values, logging=False):
    """
    Находит "локоть" (elbow point) на кривой WCSS.
    
    Алгоритм:
    1. Строит вектор от первой точки кривой к последней
    2. Для каждой точки на кривой вычисляет расстояние до этой линии
    3. Возвращает K, соответствующий точке с максимальным расстоянием
    """
    k_values = np.array(k_values)
    wcss_values = np.array(wcss_values)
    
    # Координаты первой и последней точки
    p1 = np.array([k_values[0], wcss_values[0]])
    p2 = np.array([k_values[-1], wcss_values[-1]])
    
    # Вектор от p1 к p2
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    
    # Нормализуем вектор
    line_unitvec = line_vec / line_len
    
    # Для каждой точки на кривой вычисляем расстояние до линии
    max_distance = 0
    elbow_idx = 0
    
    for i, (k, wcss) in enumerate(zip(k_values, wcss_values)):
        point = np.array([k, wcss])
        
        # Вектор от p1 к текущей точке
        point_vec = point - p1
        
        # Проекция point_vec на line_unitvec
        proj_length = np.dot(point_vec, line_unitvec)
        
        # Точка на линии, ближайшая к текущей точке
        proj_point = p1 + proj_length * line_unitvec
        
        # Расстояние от точки до линии
        distance = np.linalg.norm(point - proj_point)
        
        if distance > max_distance:
            max_distance = distance
            elbow_idx = i
    
    optimal_k = k_values[elbow_idx]
    
    if logging:
        print(f"Elbow point найден на индексе {elbow_idx}, расстояние от линии: {max_distance:.2e}")
    
    return int(optimal_k)
