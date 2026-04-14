import math
import numpy as np
import cv2
from scipy.ndimage import generic_filter, binary_dilation
from scipy import stats

def postprocess_image(quant_rgb, cluster_labels, cluster_centers_lab, config, saliency_map=None):
    """Удаление мелких регионов на основе физического размера холста с защитой контраста."""
    logging = config['logging']
    if logging:
        print('-' * 20, "\nМодуль postprocessing\n", '-' * 20)

    # --- Шаг 1: Рассчитываем минимальную площадь в пикселях ---
    h, w = quant_rgb.shape[:2]
    canvas_w_mm = config['canvas_width_mm']
    canvas_h_mm = config['canvas_height_mm']
    min_diam_mm = config['min_diameter_mm']

    px_per_mm = max(w, h) / max(canvas_w_mm, canvas_h_mm)
    if logging:
        print(f"Масштаб: {px_per_mm:.2f} пикселей на мм")

    min_radius_mm = min_diam_mm / 2.0
    min_area_mm2 = np.pi * (min_radius_mm ** 2)
    pixel_area_mm2 = (1.0 / px_per_mm) ** 2
    base_min_region_pixels = max(1, math.ceil(min_area_mm2 / pixel_area_mm2))
    min_region_pixels = base_min_region_pixels

    if logging:
        print(f"Минимальная закрашиваемая область: {min_area_mm2:.1f} мм² -> не менее {base_min_region_pixels} пикселей")
    
    # Проверяем наличие карты внимания
    has_saliency = saliency_map is not None
    if has_saliency and logging:
        print("Используется адаптивный расчет минимального размера региона на основе карты внимания")

    algo_params = config['algorithm']['postprocessing']
    connectivity = algo_params['connectivity']
    contrast_threshold = algo_params['contrast_threshold']
    grow_details = algo_params.get('grow_details', {'enabled': False})
    
    # Получаем параметры адаптивной карты внимания
    saliency_params = config['algorithm']['preprocessing'].get('saliency_map', {})
    saliency_bg_scale = saliency_params.get('saliency_bg_scale', 1.5)
    saliency_focus_scale = saliency_params.get('saliency_focus_scale', 0.5)

    # --- Шаг 2: Находим связные компоненты напрямую из cluster_labels ---
    cluster_labels = np.copy(cluster_labels)
    num_clusters = int(cluster_centers_lab.shape[0])
    components = np.zeros((h, w), dtype=np.int32)
    comp_to_cluster = {}
    current_comp_id = 1

    for cluster_id in range(num_clusters):
        mask = (cluster_labels == cluster_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        num_comp_k, comp_k = cv2.connectedComponents(mask, connectivity=connectivity)
        for local_id in range(1, num_comp_k):
            components_mask = (comp_k == local_id)
            if not np.any(components_mask):
                continue
            components[components_mask] = current_comp_id
            comp_to_cluster[current_comp_id] = cluster_id
            current_comp_id += 1

    # --- Шаг 3: Размеры компонент и удаление мелких с защитой контраста ---
    flat_comp = components.ravel()
    comp_sizes = np.bincount(flat_comp)

    if logging:
        num_components = (np.arange(comp_sizes.size)[comp_sizes > 0] != 0).sum()
        print(f"Найдено компонент (цветовые регионы): {num_components}")

    output_components = components.copy()

    comp_ids = np.arange(comp_sizes.size)
    
    # Функция для расчета адаптивного порога на основе карты внимания
    def get_local_min_pixels(comp_id, comp_mask):
        """Рассчитывает локальный порог минимального размера для компоненты."""
        if not has_saliency:
            return base_min_region_pixels
        
        # Вычисляем среднее значение saliency_map внутри маски компоненты
        saliency_values = saliency_map[comp_mask]
        mean_saliency = saliency_values.mean() if len(saliency_values) > 0 else 0.5
        
        # Адаптируем порог: высокая saliency -> меньший порог, низкая -> больший
        # Formula: adaptive_coefficient = bg_scale - mean_saliency * (bg_scale - focus_scale)
        # При mean_saliency = 1.0: коэффициент = focus_scale (разрешаем мелкие мазки на объекте)
        # При mean_saliency = 0.0: коэффициент = bg_scale (требуем крупные пятна на фоне)
        adaptive_coefficient = saliency_bg_scale - mean_saliency * (saliency_bg_scale - saliency_focus_scale)
        local_min_pixels = int(base_min_region_pixels * adaptive_coefficient)
        
        return max(1, local_min_pixels)
    
    # Определяем small и big компоненты с адаптивными порогами
    small_ids = []
    big_ids = set()
    
    for comp_id in comp_ids:
        if comp_id == 0:  # Пропускаем фон
            continue
        
        comp_mask = (components == comp_id)
        local_threshold = get_local_min_pixels(comp_id, comp_mask)
        
        if comp_sizes[comp_id] < local_threshold:
            small_ids.append(comp_id)
        else:
            big_ids.add(comp_id)
    
    small_ids = np.array(small_ids)

    for comp_id in small_ids:
        mask = (components == comp_id)
        if not np.any(mask):
            continue

        src_cluster = comp_to_cluster.get(int(comp_id))
        if src_cluster is None:
            continue
        src_color_lab = cluster_centers_lab[src_cluster]

        # Находим bounding box компоненты с отступом в 1 пиксель
        y_coords, x_coords = np.where(mask)
        y_min, y_max = max(0, y_coords.min() - 1), min(h - 1, y_coords.max() + 1)
        x_min, x_max = max(0, x_coords.min() - 1), min(w - 1, x_coords.max() + 1)

        # Локальный срез для поиска соседей
        local_slice = output_components[y_min:y_max + 1, x_min:x_max + 1]
        local_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

        # Находим соседей в локальном срезе
        neighbors_mask = (local_slice != comp_id) & (local_slice != 0) & ~local_mask
        if not np.any(neighbors_mask):
            continue

        neighbor_ids = np.unique(local_slice[neighbors_mask])
        valid_neighbors = [int(nid) for nid in neighbor_ids if nid in big_ids and nid != comp_id]
        if not valid_neighbors:
            continue

        # Ищем ближайшего соседа по цвету в LAB
        best_neighbor, min_dist = None, float('inf')
        for nid in valid_neighbors:
            dst_cluster = comp_to_cluster.get(int(nid))
            if dst_cluster is None:
                continue
            dst_color_lab = cluster_centers_lab[dst_cluster]
            dist = np.linalg.norm(src_color_lab - dst_color_lab)
            if dist < min_dist:
                min_dist, best_neighbor = dist, nid

        # Защита контраста: если компонента контрастная (высокая дистанция = важная деталь),
        # не удаляем её, а опционально "выращиваем" до минимального размера
        if best_neighbor is not None and min_dist >= contrast_threshold:
            # Компонента контрастная (например, черный зрачок на белом фоне)
            if grow_details.get('enabled', False):
                # Морфологическое расширение (dilation) для увеличения размера детали
                max_dilation_steps = grow_details.get('max_dilation_steps', 5)
                current_size = mask.sum()
                
                # Локальная маска для дилатации (чтобы не перекрывать другие защищённые компоненты)
                local_mask = mask[y_min:y_max + 1, x_min:x_max + 1].copy()
                
                # Резервная копия локального среза для возможного отката
                local_slice_backup = local_slice.copy()
                
                for dilation_step in range(max_dilation_steps):
                    if current_size >= min_region_pixels:
                        # Достигли минимального размера, прерываем дилатацию
                        break
                    
                    # Применяем бинарную дилатацию с структурирующим элементом (8-связность)
                    dilated = binary_dilation(local_mask, iterations=1)
                    
                    # Ограничиваем дилатацию: расширяемся только на пиксели, которые не принадлежат
                    # другим "важным" защищённым компонентам (чтобы избежать перекрытия)
                    new_pixels_mask = dilated & ~local_mask
                    
                    # Проверяем, какие новые пиксели можно добавить (они должны быть либо фоном,
                    # либо принадлежать соседям, которые не защищены)
                    new_pixels_in_local = new_pixels_mask[y_min:y_max + 1, x_min:x_max + 1]
                    
                    # Обновляем глобальную маску
                    new_global_mask = np.zeros_like(mask)
                    new_global_mask[y_min:y_max + 1, x_min:x_max + 1] = dilated
                    
                    # Добавляем новые пиксели в output_components
                    output_components[new_global_mask & ~mask] = comp_id
                    
                    # Обновляем локальную маску и размер
                    local_mask = dilated
                    current_size = new_global_mask.sum()
                
                # Проверка: смогла ли компонента вырасти до минимального размера
                current_area = (output_components[y_min:y_max + 1, x_min:x_max + 1] == comp_id).sum()
                
                if current_area < min_region_pixels:
                    # Компонента не смогла вырасти до минимального размера
                    # (уперлась в лимит шагов или была заблокирована соседями)
                    # Откатываем изменения и сливаем с соседом
                    output_components[y_min:y_max + 1, x_min:x_max + 1] = local_slice_backup
                    output_components[mask] = best_neighbor
                    
                    if logging:
                        print(f"Компонента {comp_id} не смогла вырасти ({current_area} < {min_region_pixels}). Слита с соседом.")
                else:
                    # Компонента успешно выросла до минимального размера
                    if logging:
                        print(f"Компонента {comp_id} (контрастная деталь): расширена с {mask.sum()} до {current_area} пикселей")
            else:
                # Если grow_details отключен, просто оставляем компоненту как есть (не удаляем)
                pass
        elif best_neighbor is not None and min_dist < contrast_threshold:
            # Компонента не контрастная (это шум) -> сливаем с ближайшим соседом
            output_components[mask] = best_neighbor

    # Пересчитываем comp_sizes после поглощения мелких компонент
    comp_sizes = np.bincount(output_components.ravel())

    # --- Шаг 4: Восстанавливаем RGB ---
    postprocessed_img = np.zeros_like(quant_rgb)
    final_colors = {}

    unique_components = np.unique(output_components)
    for comp_id in unique_components:
        if comp_id == 0:
            continue
        cluster_id = comp_to_cluster.get(int(comp_id))
        if cluster_id is None:
            continue

        color_lab = cluster_centers_lab[cluster_id].copy()
        # L был нормализован в quantizing.py: умножен на (100/255)
        # Возвращаем в cv2-диапазон (0–255)
        L = np.clip(color_lab[0] * (255.0 / 100.0), 0, 255)
        a = np.clip(color_lab[1], 0, 255)
        b = np.clip(color_lab[2], 0, 255)
        lab_px = np.uint8([[[L, a, b]]])
        rgb_px = cv2.cvtColor(lab_px, cv2.COLOR_LAB2RGB)[0, 0]

        final_colors[int(comp_id)] = tuple(rgb_px.tolist())
        postprocessed_img[output_components == comp_id] = rgb_px

    if logging:
        print(f"Постобработка завершена. Регионов после: {len(final_colors)}")

    # --- Шаг 5: Создаём color_index_map ---
    # Собираем уникальные cluster_id из final_colors и сортируем по яркости (L в LAB)
    unique_cluster_ids = set()
    for comp_id in final_colors.keys():
        cluster_id = comp_to_cluster.get(int(comp_id))
        if cluster_id is not None:
            unique_cluster_ids.add(cluster_id)
    
    # Сортируем по яркости (L канал) в порядке убывания (самый светлый первым)
    sorted_clusters = sorted(unique_cluster_ids, 
                            key=lambda cid: cluster_centers_lab[cid][0], 
                            reverse=True)
    
    # Создаём маппинг: cluster_id → порядковый номер (начиная с 1)
    color_index_map = {cluster_id: idx + 1 for idx, cluster_id in enumerate(sorted_clusters)}

    return postprocessed_img, final_colors, cluster_labels
