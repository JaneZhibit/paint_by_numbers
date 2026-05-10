import numpy as np
import cv2


def find_optimal_k(preprocessed_img, config):
    """
    Находит оптимальное количество цветов (K) для K-Means используя анализ относительного падения WCSS.
    
    Алгоритм:
    1. Уменьшает изображение пропорционально subsample_ratio для быстрого расчета
    2. Переводит в LAB цветовое пространство
    3. Запускает K-Means для разных значений K (от k_min до k_max с шагом k_step)
    4. Вычисляет WCSS (Within-Cluster Sum of Squares) для каждого K
    5. Анализирует относительное падение WCSS: на сколько процентов улучшается результат при добавлении нового цвета
    6. Останавливается, когда улучшение становится меньше 5% от начальной дисперсии
    7. Возвращает оптимальное значение K (не зависит от k_min, только от качества улучшений)
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
    threshold_percent = auto_color_config.get('stop_threshold_percent', 5.0)
    optimal_k = _find_elbow(k_values, wcss_values, threshold_percent, logging)
    
    if logging:
        print(f"Оптимальное K найдено: {optimal_k}")
        print('-' * 20 + "\n")
    
    return optimal_k


def _find_elbow(k_values, wcss_values, threshold_percent=5.0, logging=False):
    """
    Находит оптимальное K на основе относительного падения WCSS.
    Алгоритм смотрит, на сколько процентов уменьшается ошибка при добавлении новых цветов.
    Если добавление цвета улучшает результат менее чем на заданный порог (например, 5%),
    мы считаем, что достигли оптимального количества.
    """
    k_values = np.array(k_values)
    wcss_values = np.array(wcss_values)
    
    # Порог остановки: если улучшение меньше заданного процента от общей дисперсии
    threshold = threshold_percent / 100.0
    
    # Нормализуем WCSS, чтобы падение считалось в процентах от начального (худшего) состояния
    max_wcss = wcss_values[0]
    
    optimal_idx = len(k_values) - 1  # По умолчанию берем максимальное K
    
    for i in range(1, len(k_values)):
        # Считаем, на сколько упал WCSS по сравнению с предыдущим шагом
        drop = wcss_values[i-1] - wcss_values[i]
        relative_drop = drop / max_wcss
        
        if logging:
            print(f"    Шаг K={k_values[i-1]} -> {k_values[i]}: улучшение на {relative_drop*100:.1f}%")
            
        # Если падение ошибки меньше порога, значит цвета больше не дают значимого эффекта
        if relative_drop < threshold:
            optimal_idx = i - 1  # Берем предыдущее K, до того как улучшения стали мизерными
            break
            
    optimal_k = k_values[optimal_idx]
    
    if logging:
        print(f"Elbow point (по падению < {threshold*100}%) найден на K={optimal_k}")
    
    return int(optimal_k)
