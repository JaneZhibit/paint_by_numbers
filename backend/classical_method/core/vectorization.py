import numpy as np
import cv2


def _chaikin_corner_cutting(points, iterations=2):
    """
    Алгоритм Чайкина (Chaikin's corner cutting) для скругления полигона.
    
    На каждой итерации каждое ребро заменяется двумя новыми точками,
    расположенными на 1/4 и 3/4 длины ребра. Это плавно скругляет углы.
    
    Args:
        points: numpy array формы (N, 2) — вершины полигона.
        iterations: количество итераций скругления.
    
    Returns:
        numpy array формы (M, 2) — скруглённый полигон (M > N).
    """
    if len(points) < 3:
        return points

    pts = np.array(points, dtype=np.float64)

    for _ in range(iterations):
        n = len(pts)
        # Сдвинутый массив: следующая точка для каждой текущей (замыкание полигона)
        pts_next = np.roll(pts, -1, axis=0)

        # Q_i = 3/4 * P_i + 1/4 * P_{i+1}
        q = 0.75 * pts + 0.25 * pts_next
        # R_i = 1/4 * P_i + 3/4 * P_{i+1}
        r = 0.25 * pts + 0.75 * pts_next

        # Чередуем Q и R: [Q0, R0, Q1, R1, ...]
        new_pts = np.empty((2 * n, 2), dtype=np.float64)
        new_pts[0::2] = q
        new_pts[1::2] = r

        pts = new_pts

    return pts


def _vectorize_raster_shift(cluster_labels, logging=False):
    """
    Быстрый метод: находит границы сравнением соседних пикселей через матричные сдвиги.
    Граница = пиксель, чей правый или нижний сосед принадлежит другому кластеру.
    
    Args:
        cluster_labels: 2D numpy array (H, W) с метками кластеров.
        logging: выводить ли отладочную информацию.
    
    Returns:
        borders_mask: uint8 массив (H, W), 255 = граница, 0 = внутренность.
        svg_paths: пустой словарь (raster_shift не создаёт SVG-контуры).
    """
    h, w = cluster_labels.shape

    # Сравнение с правым соседом: если метка отличается — граница
    diff_right = np.zeros((h, w), dtype=bool)
    diff_right[:, :-1] = cluster_labels[:, :-1] != cluster_labels[:, 1:]

    # Сравнение с нижним соседом: если метка отличается — граница
    diff_bottom = np.zeros((h, w), dtype=bool)
    diff_bottom[:-1, :] = cluster_labels[:-1, :] != cluster_labels[1:, :]

    borders_mask = np.where(diff_right | diff_bottom, 255, 0).astype(np.uint8)

    if logging:
        border_count = np.count_nonzero(borders_mask)
        total = h * w
        print(f"  [raster_shift] Границы: {border_count} px ({border_count / total * 100:.1f}% от изображения)")

    return borders_mask, {}


def _vectorize_contours(cluster_labels, config, logging=False):
    """
    Метод на основе cv2.findContours: извлекает контуры каждого кластера,
    опционально сглаживает их (Douglas-Peucker или Chaikin) и рисует на маске.
    
    Args:
        cluster_labels: 2D numpy array (H, W) с метками кластеров.
        config: полный конфиг (config['algorithm']['vectorization']).
        logging: выводить ли отладочную информацию.
    
    Returns:
        borders_mask: uint8 массив (H, W), 255 = граница, 0 = внутренность.
        svg_paths: dict { cluster_id: [ [ (x,y), ... ], ... ] } — контуры для будущего SVG.
    """
    vec_config = config['algorithm']['vectorization']
    smoothing_cfg = vec_config['smoothing']
    smoothing_enabled = smoothing_cfg.get('enabled', True)
    smoothing_method = smoothing_cfg.get('method', 'approx_poly_dp')

    h, w = cluster_labels.shape
    borders_mask = np.zeros((h, w), dtype=np.uint8)
    svg_paths = {}

    unique_labels = np.unique(cluster_labels)

    if logging:
        print(f"  [contours] Уникальных кластеров: {len(unique_labels)}")
        print(f"  [contours] Сглаживание: {'вкл' if smoothing_enabled else 'выкл'}"
              + (f" ({smoothing_method})" if smoothing_enabled else ""))

    total_contours = 0

    for label_id in unique_labels:
        # Бинарная маска для текущего кластера
        binary_mask = (cluster_labels == label_id).astype(np.uint8) * 255

        # Извлекаем контуры (RETR_LIST — все контуры без иерархии)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        smoothed_contours = []

        for cnt in contours:
            # Пропускаем слишком мелкие контуры (меньше 3 точек)
            if len(cnt) < 3:
                continue

            if smoothing_enabled:
                if smoothing_method == 'approx_poly_dp':
                    # Алгоритм Дугласа-Пекера: упрощение контура
                    epsilon_factor = smoothing_cfg['approx_poly_dp'].get('epsilon_factor', 0.005)
                    epsilon = epsilon_factor * cv2.arcLength(cnt, closed=True)
                    approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
                    smoothed_contours.append(approx)

                elif smoothing_method == 'chaikin':
                    # Алгоритм Чайкина: скругление углов
                    iterations = smoothing_cfg['chaikin'].get('iterations', 2)
                    # cnt имеет shape (N, 1, 2) — убираем лишнюю ось
                    pts_2d = cnt.reshape(-1, 2).astype(np.float64)
                    smoothed_pts = _chaikin_corner_cutting(pts_2d, iterations)
                    # Возвращаем в формат OpenCV: (M, 1, 2) int32
                    smoothed = smoothed_pts.astype(np.int32).reshape(-1, 1, 2)
                    smoothed_contours.append(smoothed)
            else:
                smoothed_contours.append(cnt)

        total_contours += len(smoothed_contours)

        # Рисуем все контуры этого кластера на общей маске (толщина 1px)
        if smoothed_contours:
            cv2.polylines(borders_mask, smoothed_contours, isClosed=True, color=255, thickness=1)

        # Сохраняем контуры для будущего SVG-экспорта
        svg_paths[int(label_id)] = [
            cnt.reshape(-1, 2).tolist() for cnt in smoothed_contours
        ]

    if logging:
        border_count = np.count_nonzero(borders_mask)
        print(f"  [contours] Всего контуров: {total_contours}")
        print(f"  [contours] Границы: {border_count} px ({border_count / (h * w) * 100:.1f}%)")

    return borders_mask, svg_paths


def vectorize_image(cluster_labels, config, logging=False):
    """
    Точка входа: извлекает контуры из cluster_labels.
    
    Args:
        cluster_labels: 2D numpy array (H, W) с метками кластеров после постобработки.
        config: полный конфиг (содержит config['algorithm']['vectorization']).
        logging: выводить ли отладочную информацию.
    
    Returns:
        borders_mask: uint8 массив (H, W), 255 = граница, 0 = внутренность.
        svg_paths: dict с контурами для SVG (пустой для raster_shift).
    """
    method = config['algorithm']['vectorization'].get('method', 'raster_shift')

    if logging:
        print(f"\n--- Векторизация (метод: {method}) ---")

    if method == 'raster_shift':
        return _vectorize_raster_shift(cluster_labels, logging)
    elif method == 'contours':
        return _vectorize_contours(cluster_labels, config, logging)
    else:
        raise ValueError(f"Неизвестный метод векторизации: {method}")
