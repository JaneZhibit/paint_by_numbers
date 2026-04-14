import math
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from PIL import Image


def quantize_image(preprocessed_img, config):
    colours_cnt = config['colours_cnt']
    logging = config['logging']
    algo_params = config['algorithm']['quantizing']

    if logging: print('-' * 20, "\nМодуль quantizing\n", '-' * 20)

    color_space = algo_params['color_space']

    # 1. Перевод в выбранное цветовое пространство
    if color_space == 'LAB':
        img_converted = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        # Нормализация L канала для лучшего распределения (опционально)
        img_converted[:, :, 0] *= (100.0 / 255.0)
    else:
        img_converted = preprocessed_img.astype(np.float32)  # RGB

    # 2. Выбор пространственного метода сегментации
    spatial_method = algo_params.get('spatial_method', 'none')
    superpixels_config = algo_params.get('superpixels', {})
    superpixels_enabled = superpixels_config.get('enabled', False)

    # 3. Выбор метода квантования палитры
    palette_method = algo_params.get('palette_method', 'kmeans')
    
    # Настройка K-Means (используется если palette_method == 'kmeans')
    km_params = algo_params['kmeans']
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, km_params['max_iter'], km_params['epsilon'])
    attempts = km_params['attempts']

    # Проверка: использовать ли взвешенный K-Means
    weighted_kmeans_config = algo_params.get('weighted_kmeans', {})
    use_weighted_kmeans = weighted_kmeans_config.get('enabled', False)

    # ============================================================================
    # БЛОК ПРОСТРАНСТВЕННОЙ СЕГМЕНТАЦИИ: SLIC или Watershed
    # ============================================================================
    if spatial_method == 'slic' and superpixels_enabled:
        if logging: print(f"Инициализация SLIC Superpixels (region_size={superpixels_config['region_size']}, ruler={superpixels_config['ruler']})...")
        
        # Рассчитываем минимальный размер суперпикселя в пикселях
        # (аналогично логике в postprocessing.py, шаг 1)
        h, w = preprocessed_img.shape[:2]
        canvas_w_mm = config['canvas_width_mm']
        canvas_h_mm = config['canvas_height_mm']
        min_diam_mm = config['min_diameter_mm']
        
        px_per_mm = max(w, h) / max(canvas_w_mm, canvas_h_mm)
        min_radius_mm = min_diam_mm / 2.0
        min_area_mm2 = np.pi * (min_radius_mm ** 2)
        pixel_area_mm2 = (1.0 / px_per_mm) ** 2
        min_region_pixels = max(1, int(np.ceil(min_area_mm2 / pixel_area_mm2)))
        
        if logging:
            print(f"  Минимальный размер региона: {min_region_pixels} пикселей")
        
        # Инициализируем SLIC с параметрами из конфига
        # MSLIC = Multi-level SLIC (более стабильный вариант)
        slic = cv2.ximgproc.createSuperpixelSLIC(
            preprocessed_img,
            algorithm=cv2.ximgproc.MSLIC,
            region_size=superpixels_config['region_size'],
            ruler=superpixels_config['ruler']
        )
        
        # Запускаем итерации сегментации
        slic.iterate(10)
        
        if logging: print("  Выполнение итераций SLIC...")
        
        # Применяем "магию" OpenCV для удаления мелких шумовых суперпикселей
        # Это переназначает пиксели мелких регионов их соседям
        slic.enforceLabelConnectivity(min_element_size=min_region_pixels)
        
        # Получаем маску сегментов (каждый пиксель имеет ID суперпикселя)
        sp_labels = slic.getLabels()
        num_superpixels = slic.getNumberOfSuperpixels()
        
        if logging: print(f"  Получено суперпикселей: {num_superpixels}")
        
        # Рассчитываем средний цвет для КАЖДОГО суперпикселя
        # Результат: массив размером (num_superpixels, 3) с цветами в пространстве img_converted
        sp_colors = np.zeros((num_superpixels, 3), dtype=np.float32)
        
        for sp_id in range(num_superpixels):
            mask = (sp_labels == sp_id)
            if mask.sum() > 0:
                # Берем средний цвет пикселей этого суперпикселя из img_converted
                sp_colors[sp_id] = img_converted[mask].mean(axis=0)
        
        if logging: print(f"  Рассчитаны средние цвета {num_superpixels} суперпикселей")
        
        # Теперь запускаем K-Means на МАЛЕНЬКОМ массиве sp_colors вместо всех пикселей
        # Это значительно ускоряет процесс и улучшает качество кластеризации
        pixels = sp_colors
        
    elif spatial_method == 'watershed':
        if logging: print("Инициализация Watershed сегментации...")
        
        # Получаем grayscale изображение для анализа
        if color_space == 'LAB':
            # Используем L канал из LAB (уже нормализован)
            gray_img = (img_converted[:, :, 0] * 255.0 / 100.0).astype(np.uint8)
        else:
            # Преобразуем RGB в grayscale
            gray_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
        
        # Применяем морфологический градиент для выделения границ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
        
        if logging: print("  Применён морфологический градиент")
        
        # Бинаризация для получения маркеров
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Нахождение уверенного фона (дилатация)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        sure_bg = cv2.dilate(binary, kernel_bg, iterations=3)
        
        # Нахождение уверенных объектов (эрозия)
        kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sure_fg = cv2.erode(binary, kernel_fg, iterations=3)
        
        # Неизвестная область (разница между фоном и объектами)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        if logging: print("  Вычислены уверенные области")
        
        # Поиск маркеров через connected components
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Маркер 0 - фон, маркер 1+ - объекты, маркер 0 - неизвестная область
        markers = markers + 1
        markers[unknown == 255] = 0
        
        if logging: print(f"  Найдено маркеров: {markers.max()}")
        
        # Применяем Watershed
        # Преобразуем в BGR для cv2.watershed
        if color_space == 'LAB':
            temp_converted = img_converted.copy()
            temp_converted[:, :, 0] *= (255.0 / 100.0)
            temp_converted = np.clip(temp_converted, 0, 255)
            img_bgr = cv2.cvtColor(temp_converted.astype(np.uint8), cv2.COLOR_LAB2BGR)
        else:
            img_bgr = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR)
        
        markers = cv2.watershed(img_bgr, markers)
        
        if logging: print("  Watershed завершён")
        
        # Получаем метки бассейнов (исключаем фон и границы)
        sp_labels = markers.copy()
        unique_labels = np.unique(sp_labels)
        # Исключаем -1 (границы) и 1 (фон)
        valid_labels = unique_labels[unique_labels > 1]
        num_watersheds = len(valid_labels)
        
        if logging: print(f"  Получено бассейнов: {num_watersheds}")
        
        # Рассчитываем средний цвет для КАЖДОГО бассейна
        sp_colors = np.zeros((num_watersheds, 3), dtype=np.float32)
        
        for idx, label_id in enumerate(valid_labels):
            mask = (sp_labels == label_id)
            if mask.sum() > 0:
                # Берем средний цвет пикселей этого бассейна из img_converted
                sp_colors[idx] = img_converted[mask].mean(axis=0)
        
        if logging: print(f"  Рассчитаны средние цвета {num_watersheds} бассейнов")
        
        # Используем усредненные цвета бассейнов для последующего квантования
        pixels = sp_colors
        
        # Сохраняем valid_labels для использования в блоке восстановления
        watershed_valid_labels = valid_labels
        
    else:
        # Старая логика: используем все пиксели изображения (spatial_method == 'none')
        pixels = img_converted.reshape((-1, 3))
        watershed_valid_labels = None

    # ============================================================================
    # ВЫБОР МЕТОДА КВАНТОВАНИЯ ПАЛИТРЫ
    # ============================================================================
    if palette_method in ['median_cut', 'octree']:
        if logging: print(f"Квантование методом {palette_method.upper()} (K={colours_cnt}, Пространство={color_space})...")
        
        # Преобразуем в PIL Image для использования встроенного квантования
        # PIL работает с RGB в диапазоне 0-255
        if color_space == 'LAB':
            # Восстанавливаем RGB из LAB для PIL
            temp_converted = img_converted.copy()
            temp_converted[:, :, 0] *= (255.0 / 100.0)
            temp_converted = np.clip(temp_converted, 0, 255)
            pil_img = Image.fromarray(cv2.cvtColor(temp_converted.astype(np.uint8), cv2.COLOR_LAB2RGB))
        else:
            pil_img = Image.fromarray(preprocessed_img.astype(np.uint8))
        
        # Выбираем метод квантования
        if palette_method == 'median_cut':
            method = Image.Quantize.MEDIANCUT
        else:  # octree
            method = Image.Quantize.OCTREE
        
        # Применяем квантование
        pil_quant = pil_img.quantize(colors=colours_cnt, method=method)
        
        # Получаем палитру (RGB в диапазоне 0-255)
        palette_data = pil_quant.getpalette()
        palette_rgb = np.array(palette_data[:colours_cnt*3]).reshape((colours_cnt, 3)).astype(np.float32)
        
        # Преобразуем палитру в выбранное цветовое пространство
        if color_space == 'LAB':
            palette_rgb_uint8 = palette_rgb.astype(np.uint8)
            # Добавляем размерность для cv2.cvtColor
            palette_rgb_uint8 = palette_rgb_uint8.reshape((1, colours_cnt, 3))
            palette_lab = cv2.cvtColor(palette_rgb_uint8, cv2.COLOR_RGB2LAB).reshape((colours_cnt, 3)).astype(np.float32)
            # Нормализуем L канал как в основном коде
            palette_lab[:, 0] *= (100.0 / 255.0)
            centers = palette_lab
        else:
            centers = palette_rgb
        
        # Находим ближайший цвет из палитры для каждого пикселя
        distances = cdist(pixels, centers, metric='euclidean')
        labels = np.argmin(distances, axis=1).reshape(-1, 1).astype(np.int32)
        
    elif palette_method == 'kmeans':
        if use_weighted_kmeans:
            if logging: print(f"K-means с взвешиванием (K={colours_cnt}, Пространство={color_space})...")
            
            # Получение параметров для взвешивания
            edge_threshold1 = weighted_kmeans_config.get('edge_threshold1', 50)
            edge_threshold2 = weighted_kmeans_config.get('edge_threshold2', 150)
            weight_factor = weighted_kmeans_config.get('weight_factor', 5)
            
            # Преобразование в градации серого для поиска границ
            gray_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
            
            # Применение Canny для поиска границ
            edges = cv2.Canny(gray_img, edge_threshold1, edge_threshold2)
            
            if superpixels_enabled:
                # Если используются SLIC: взвешиваем суперпиксели, которые содержат границы
                # Для каждого суперпикселя проверяем, содержит ли он пиксели границ
                edge_superpixels = set()
                edge_indices = np.where(edges > 0)
                for y, x in zip(edge_indices[0], edge_indices[1]):
                    sp_id = sp_labels[y, x]
                    edge_superpixels.add(sp_id)
                
                # Дублируем суперпиксели, которые содержат границы
                pixels_weighted_list = [pixels]
                for sp_id in edge_superpixels:
                    pixels_weighted_list.append(np.repeat(pixels[sp_id:sp_id+1], weight_factor, axis=0))
                
                pixels_weighted = np.vstack(pixels_weighted_list).astype(np.float32)
            else:
                # Старая логика для обычного K-Means: взвешиваем пиксели границ
                edge_indices = np.where(edges > 0)
                edge_pixel_indices = np.ravel_multi_index(edge_indices, edges.shape)
                
                # Получение пикселей с границами и их дублирование
                edge_pixels = pixels[edge_pixel_indices]
                weighted_edge_pixels = np.repeat(edge_pixels, weight_factor, axis=0)
                
                # Объединение оригинального массива с взвешенными пикселями границ
                pixels_weighted = np.vstack([pixels, weighted_edge_pixels]).astype(np.float32)
            
            # Запуск K-Means на расширенном массиве
            _, _, centers = cv2.kmeans(
                data=pixels_weighted, K=colours_cnt, bestLabels=None, criteria=criteria,
                attempts=attempts, flags=cv2.KMEANS_PP_CENTERS
            )
            
            # Применение полученных центров к ОРИГИНАЛЬНОМУ массиву пикселей
            # Используем cdist для поиска ближайшего центра
            distances = cdist(pixels, centers, metric='euclidean')
            labels = np.argmin(distances, axis=1).reshape(-1, 1).astype(np.int32)
            
        else:
            if logging: print(f"K-means (K={colours_cnt}, Пространство={color_space}, Attempts={attempts})...")

            _, labels, centers = cv2.kmeans(
                data=pixels, K=colours_cnt, bestLabels=None, criteria=criteria,
                attempts=attempts, flags=cv2.KMEANS_PP_CENTERS
            )
    else:
        raise ValueError(f"Неизвестный метод квантования палитры: {palette_method}")

    # ============================================================================
    # Восстановление результатов квантования на полноразмерное изображение
    # ============================================================================
    if spatial_method == 'slic' and superpixels_enabled:
        # После K-Means на суперпикселях нужно развернуть результаты обратно
        # на полноразмерное изображение, используя sp_labels
        
        # labels содержит кластер для каждого суперпикселя
        # Создаем массив, где каждому пикселю назначается кластер его суперпикселя
        cluster_labels = labels.flatten()[sp_labels]
        
        # Восстанавливаем quantized_converted: для каждого пикселя берем цвет центра его кластера
        quantized_converted = centers[cluster_labels].reshape(img_converted.shape)
        
        if logging: print(f"  Развернуты результаты квантования на полноразмерное изображение")
        
    elif spatial_method == 'watershed':
        # После K-Means на бассейнах Watershed нужно развернуть результаты обратно
        # на полноразмерное изображение, используя sp_labels (маркеры Watershed)
        
        # labels содержит кластер для каждого бассейна
        # Создаем массив, где каждому пикселю назначается кластер его бассейна
        # Нужно переиндексировать, так как watershed_valid_labels может быть не последовательным
        label_to_cluster = {}
        for idx, label_id in enumerate(watershed_valid_labels):
            label_to_cluster[label_id] = labels[idx, 0]
        
        cluster_labels = np.zeros_like(sp_labels, dtype=np.int32)
        for label_id, cluster_id in label_to_cluster.items():
            cluster_labels[sp_labels == label_id] = cluster_id
        
        # Восстанавливаем quantized_converted: для каждого пикселя берем цвет центра его кластера
        quantized_converted = centers[cluster_labels].reshape(img_converted.shape)
        
        if logging: print(f"  Развернуты результаты квантования на полноразмерное изображение")
        
    else:
        # Старая логика для обычного K-Means (spatial_method == 'none')
        quantized_converted = centers[labels.flatten()].reshape(img_converted.shape)
        cluster_labels = labels.reshape(preprocessed_img.shape[:2])

    # ============================================================================
    # ИЕРАРХИЧЕСКОЕ РАСЩЕПЛЕНИЕ КРУПНЫХ КЛАСТЕРОВ
    # ============================================================================
    hierarchical_split_config = algo_params.get('hierarchical_split', {})
    hierarchical_split_enabled = hierarchical_split_config.get('enabled', False)
    
    if hierarchical_split_enabled:
        if logging: print("Иерархическое расщепление крупных кластеров...")
        
        min_area_pixels = hierarchical_split_config.get('min_area_pixels', 2000)
        variance_threshold = hierarchical_split_config.get('variance_threshold', 30.0)
        
        # Находим связные компоненты в cluster_labels
        num_components, components = cv2.connectedComponents(cluster_labels.astype(np.uint8))
        
        if logging: print(f"  Найдено компонент: {num_components}")
        
        # Список для новых центров
        new_centers_list = [centers]
        current_max_label = colours_cnt - 1
        
        # Обновляемая копия cluster_labels
        updated_cluster_labels = cluster_labels.copy()
        
        # Проходим по каждой компоненте
        for comp_id in range(1, num_components):
            mask = (components == comp_id)
            area = mask.sum()
            
            # Проверяем размер компоненты
            if area > min_area_pixels:
                # Получаем пиксели этой компоненты из оригинального изображения
                comp_pixels = preprocessed_img[mask]
                
                # Рассчитываем стандартное отклонение цветов
                color_std = np.std(comp_pixels, axis=0).mean()
                
                if logging:
                    print(f"  Компонента {comp_id}: площадь={area}, std={color_std:.2f}")
                
                # Если дисперсия высока, расщепляем кластер
                if color_std > variance_threshold:
                    if logging: print(f"    → Расщепление (std > {variance_threshold})")
                    
                    # Определяем количество подкластеров (K=2 или K=3 в зависимости от дисперсии)
                    K_split = 3 if color_std > variance_threshold * 1.5 else 2
                    
                    # Преобразуем пиксели в нужное цветовое пространство
                    if color_space == 'LAB':
                        comp_img_converted = cv2.cvtColor(comp_pixels.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
                        comp_img_converted[:, 0] *= (100.0 / 255.0)
                    else:
                        comp_img_converted = comp_pixels.astype(np.float32)
                    
                    # Запускаем локальный K-Means на пикселях этой маски
                    _, local_labels, local_centers = cv2.kmeans(
                        data=comp_img_converted, K=K_split, bestLabels=None,
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0),
                        attempts=3, flags=cv2.KMEANS_PP_CENTERS
                    )
                    
                    # Добавляем новые центры в общий массив
                    new_centers_list.append(local_centers)
                    
                    # Переиндексируем локальные метки в глобальные индексы
                    local_labels_flat = local_labels.flatten()
                    mask_indices = np.where(mask)
                    
                    for local_label in range(K_split):
                        global_label = current_max_label + local_label + 1
                        local_mask = (local_labels_flat == local_label)
                        # Обновляем только пиксели, которые принадлежат этому локальному кластеру
                        updated_cluster_labels[mask_indices[0][local_mask], mask_indices[1][local_mask]] = global_label
                    
                    current_max_label += K_split
                    
                    if logging: print(f"    → Добавлено {K_split} новых центров")
        
        # Объединяем все центры
        if len(new_centers_list) > 1:
            centers = np.vstack(new_centers_list)
            cluster_labels = updated_cluster_labels
            
            if logging: print(f"  Итого центров после расщепления: {len(centers)}")
            
            # Пересчитываем quantized_converted на основе обновленных labels и centers
            quantized_converted = centers[cluster_labels].reshape(img_converted.shape)
    
    # 4. Возврат в RGB
    if color_space == 'LAB':
        # L был нормализован на (100/255), возвращаем в cv2-диапазон (0–255)
        quantized_converted[:, :, 0] *= (255.0 / 100.0)
        # Clip для избежания переполнения при округлении
        quantized_converted = np.clip(quantized_converted, 0, 255)
        quant_rgb = cv2.cvtColor(quantized_converted.astype(np.uint8), cv2.COLOR_LAB2RGB)
    else:
        quant_rgb = quantized_converted.astype(np.uint8)

    if logging: print("Квантование завершено.")
    return quant_rgb, quantized_converted, centers, cluster_labels
