import math
import numpy as np
import cv2
from scipy.ndimage import generic_filter
from scipy import stats

def postprocess_image(quant_rgb, cluster_labels, cluster_centers_lab, config):
    """Удаление мелких регионов на основе физического размера холста."""
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
    min_region_pixels = max(1, math.ceil(min_area_mm2 / pixel_area_mm2))

    if logging:
        print(f"Минимальная закрашиваемая область: {min_area_mm2:.1f} мм² -> не менее {min_region_pixels} пикселей")

    algo_params = config['algorithm']['postprocessing']
    clean_iters = algo_params['clean_iterations']
    morph_k = algo_params['morph_kernel_size']
    connectivity = algo_params['connectivity']

    # Шаг 2: Связные компоненты
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    cleaned_labels = np.copy(cluster_labels)

    for cluster_id in range(config['colours_cnt']):
        mask = (cluster_labels == cluster_id).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=clean_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=clean_iters)
        cleaned_labels[mask == 1] = cluster_id

    cluster_labels = cleaned_labels

    num_clusters = int(cluster_centers_lab.shape[0])
    components = np.zeros((h, w), dtype=np.int32)
    comp_to_cluster = {}
    current_comp_id = 1

    for cluster_id in range(num_clusters):
        mask = (cluster_labels == cluster_id).astype(np.uint8)
        if mask.sum() == 0: continue

        num_comp_k, comp_k = cv2.connectedComponents(mask, connectivity=connectivity)
        for local_id in range(1, num_comp_k):
            components_mask = (comp_k == local_id)
            if not np.any(components_mask): continue
            components[components_mask] = current_comp_id
            comp_to_cluster[current_comp_id] = cluster_id
            current_comp_id += 1

    # --- Шаг 3: Размеры компонент и удаление мелких ---
    flat_comp = components.ravel()
    comp_sizes = np.bincount(flat_comp)

    if logging:
        num_components = (np.arange(comp_sizes.size)[comp_sizes > 0] != 0).sum()
        print(f"Найдено компонент (цветовые регионы): {num_components}")

    output_components = components.copy()
    kernel_dil = np.ones((3, 3), dtype=np.uint8)

    comp_ids = np.arange(comp_sizes.size)
    small_ids = comp_ids[(comp_ids != 0) & (comp_sizes < min_region_pixels)]
    big_ids = set(comp_ids[(comp_ids != 0) & (comp_sizes >= min_region_pixels)].tolist())

    # Примечание: smoothed_labels вычисляется, но не используется в исходном коде
    # def mode_filter(x): return stats.mode(x, keepdims=False)[0]
    # smoothed_labels = generic_filter(cluster_labels, mode_filter, size=5)

    for comp_id in small_ids:
        mask = (components == comp_id)
        if not np.any(mask): continue

        mask_uint8 = mask.astype(np.uint8)
        mask_dilated = cv2.dilate(mask_uint8, kernel_dil, iterations=1)
        neighbors_mask = (mask_dilated - mask_uint8).astype(bool)

        if not np.any(neighbors_mask): continue

        neighbor_ids = np.unique(output_components[neighbors_mask])
        valid_neighbors =[int(nid) for nid in neighbor_ids if nid in big_ids and nid != comp_id]
        if not valid_neighbors: continue

        src_cluster = comp_to_cluster.get(int(comp_id))
        if src_cluster is None: continue
        src_color_lab = cluster_centers_lab[src_cluster]

        best_neighbor, min_dist = None, float('inf')
        for nid in valid_neighbors:
            dst_cluster = comp_to_cluster.get(int(nid))
            if dst_cluster is None: continue
            dst_color_lab = cluster_centers_lab[dst_cluster]
            dist = np.linalg.norm(src_color_lab - dst_color_lab)
            if dist < min_dist:
                min_dist, best_neighbor = dist, nid

        if best_neighbor is not None:
            output_components[mask] = best_neighbor

    # Пересчитываем comp_sizes после поглощения мелких компонент
    comp_sizes = np.bincount(output_components.ravel())

    # --- Шаг 4: Восстанавливаем RGB ---
    postprocessed_img = np.zeros_like(quant_rgb)
    final_colors = {}

    unique_components = np.unique(output_components)
    for comp_id in unique_components:
        if comp_id == 0: continue
        cluster_id = comp_to_cluster.get(int(comp_id))
        if cluster_id is None: continue

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

    return postprocessed_img, final_colors, cluster_labels, color_index_map, output_components, comp_to_cluster, comp_sizes
