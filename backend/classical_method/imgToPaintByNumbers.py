"""
Пришло осознание, что необходимо уйти в ООП и писать код в одном файле
"""
import math

import numpy as np
from PIL import Image, ImageFilter
import cv2


class ClassicalPaintByNumbers:
    def __init__(self, config):
        """
        Инициализация генератора картины по номерам.
        В качестве конфига нужен словарь
        """

        self.config = config
        self.validate_config()

        # ---Промежуточные данные — будут заполняться по мере выполнения этапов

        # Предобработка
        self.original_img = None
        self.preprocessed_img = None  # np.array, RGB, uint8

        # Квантование
        self.quant_rgb = None  # квантованное изображение в RGB
        self.quant_lab = None  # в корректном CIELAB (L: 0–100)
        self.cluster_centers_lab = None  # центры кластеров, shape (K, 3)
        self.cluster_labels = None  # метки кластеров, shape (H, W)

        # Постобработка
        self.postprocessed_img = None
        self.final_colors = None  # {region_id: (R, G, B)}

    def validate_config(self):
        """Проверка обязательных параметров."""
        required = ['img_path', 'target_max_side', 'canvas_width_mm', 'canvas_height_mm', 'min_diameter_mm',
                    'colours_cnt', 'logging']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Отсутствует обязательный параметр: {key}")

    def preprocessing(self):
        """Загрузка, масштабирование и сглаживание изображения."""
        if self.config['logging']:
            print('-' * 20,"\nМодуль preprocessing\n", '-' * 20)

        # 1. Загрузка
        self.original_img = Image.open(self.config['img_path']).convert("RGB")
        w, h = self.original_img.size

        if self.config['logging']:
            print(f"Исходное изображение: {w}x{h}")

        # 2. Масштабирование
        target_max = self.config['target_max_side']
        scale = target_max / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if self.config['logging']:
            print(f"Масштаб: {scale:.3f} -> новый размер: {new_w}×{new_h}")

        img_scaled = self.original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 3. Сглаживание
        if self.config['logging']:
            print("Применение медианного фильтра (size=3)...")
        img_smooth = img_scaled.filter(ImageFilter.MedianFilter(size=3))

        # 4. Сохранение
        self.preprocessed_img = np.array(img_smooth)
        if self.config['logging']:
            print("Предобработка завершена.")

    def quantizing(self):
        """Квантование изображения до заданного числа цветов в пространстве CIELAB."""
        if self.preprocessed_img is None:
            raise RuntimeError("Сначала выполните preprocessing()")

        if self.config['logging']:
            print('-' * 20, "\nМодуль quantizing\n", '-' * 20)

        img_np = self.preprocessed_img
        colours_cnt = self.config['colours_cnt']

        # Преобразование в LAB
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = img_lab[:, :, 0] * (100.0 / 255.0)  # L: 0..100
        a = img_lab[:, :, 1]  # 0-255
        b = img_lab[:, :, 2]  # 0-255
        lab_true = np.stack([L, a, b], axis=2)

        # Подготовка данных для k-means
        pixels = lab_true.reshape((-1, 3)).astype("float32")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)
        attempts = 3

        # Запуск k-means
        print(f"Запуск k-means с K={colours_cnt}...")
        compactness, labels, centers = cv2.kmeans(
            data=pixels, K=colours_cnt, bestLabels=None, criteria=criteria,
            attempts=attempts, flags=cv2.KMEANS_PP_CENTERS
        )

        self.quant_lab = centers[labels.flatten()].reshape(lab_true.shape)  # (H, W, 3), L=0–100
        self.cluster_centers_lab = centers  # (K, 3)
        self.cluster_labels = labels.reshape(img_np.shape[:2])  # (H, W)

        # Обратное преобразование в RGB (для визуализации и постобработки)
        L = self.quant_lab[:, :, 0] * (255.0 / 100.0)  # 0..255 (обратно вернули L)
        a = self.quant_lab[:, :, 1]
        b = self.quant_lab[:, :, 2]

        lab_for_cv = np.stack([L, a, b], axis=2).astype(np.uint8)
        self.quant_rgb = cv2.cvtColor(lab_for_cv, cv2.COLOR_LAB2RGB)

        if self.config['logging']:
            print(f"Квантование завершено. Получено {colours_cnt} цветов.")

    def postprocessing(self):
        """Удаление мелких регионов на основе физического размера холста."""
        if self.quant_rgb is None:
            raise RuntimeError("Сначала выполните quantizing()")

        if self.config['logging']:
            print('-' * 20, "\nМодуль postprocessing\n", '-' * 20)

        # --- Шаг 1: Рассчитываем минимальную площадь в пикселях ---
        h, w = self.quant_rgb.shape[:2]
        canvas_w_mm = self.config['canvas_width_mm']
        canvas_h_mm = self.config['canvas_height_mm']
        min_diam_mm = self.config['min_diameter_mm']

        # Сколько пикселей в 1 мм
        px_per_mm = max(w, h) / max(canvas_w_mm, canvas_h_mm)
        if self.config['logging']:
            print(f"Масштаб: {px_per_mm:.2f} пикселей на мм")

        # Минимальная площадь (круг диаметром min_diam_mm)
        min_radius_mm = min_diam_mm / 2.0
        min_area_mm2 = np.pi * (min_radius_mm ** 2)

        # Площадь одного пикселя в мм²
        pixel_area_mm2 = (1.0 / px_per_mm) ** 2
        min_region_pixels = math.ceil(min_area_mm2 / pixel_area_mm2)
        min_region_pixels = max(1, min_region_pixels)

        if self.config['logging']:
            print(
                f"Минимальная закрашиваемая область: "
                f"{min_area_mm2:.1f} мм² -> не менее {min_region_pixels} пикселей"
            )

        # --- Шаг 2: Связные компоненты отдельно внутри каждого кластера ---
        labels = self.cluster_labels  # (H, W), значения 0..K-1
        num_clusters = int(self.cluster_centers_lab.shape[0])

        components = np.zeros((h, w), dtype=np.int32)
        comp_to_cluster = {}
        current_comp_id = 1

        for cluster_id in range(num_clusters):
            # Бинарная маска пикселей данного кластера
            mask = (labels == cluster_id).astype(np.uint8)
            if mask.sum() == 0:
                continue

            # Связные компоненты только внутри этого цвета
            num_comp_k, comp_k = cv2.connectedComponents(mask, connectivity=4)
            # comp_k: 0 – фон, 1..num_comp_k-1 – локальные компоненты

            for local_id in range(1, num_comp_k):
                components_mask = (comp_k == local_id)
                if not np.any(components_mask):
                    continue
                components[components_mask] = current_comp_id
                comp_to_cluster[current_comp_id] = cluster_id
                current_comp_id += 1

        # --- Шаг 3: Размеры компонент и удаление мелких ---
        flat_comp = components.ravel()
        comp_sizes = np.bincount(flat_comp)  # индекс = comp_id, значение = размер

        if self.config['logging']:
            num_components = (np.arange(comp_sizes.size)[comp_sizes > 0] != 0).sum()
            print(f"Найдено компонент (цветовые регионы): {num_components}")

        output_components = components.copy()
        kernel = np.ones((3, 3), dtype=np.uint8)

        # Множество id мелких и крупных компонент
        comp_ids = np.arange(comp_sizes.size)
        small_mask = (comp_ids != 0) & (comp_sizes < min_region_pixels)
        small_ids = comp_ids[small_mask]
        big_mask = (comp_ids != 0) & (comp_sizes >= min_region_pixels)
        big_ids = set(comp_ids[big_mask].tolist())

        for comp_id in small_ids:
            # Маска мелкой компоненты
            mask = (components == comp_id)
            if not np.any(mask):
                continue

            # Расширяем, чтобы найти соседей
            mask_uint8 = mask.astype(np.uint8)
            mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
            neighbors_mask = (mask_dilated - mask_uint8).astype(bool)

            if not np.any(neighbors_mask):
                continue

            # Соседние компоненты (индексы)
            neighbor_ids = np.unique(output_components[neighbors_mask])

            # Только крупные и не фон
            valid_neighbors = [
                int(nid) for nid in neighbor_ids
                if nid in big_ids and nid != comp_id
            ]
            if not valid_neighbors:
                continue

            # Цвет мелкой компоненты (LAB)
            src_cluster = comp_to_cluster.get(int(comp_id), None)
            if src_cluster is None:
                continue
            src_color_lab = self.cluster_centers_lab[src_cluster]

            # Находим соседа с минимальным расстоянием по LAB
            best_neighbor = None
            min_dist = float('inf')

            for nid in valid_neighbors:
                dst_cluster = comp_to_cluster.get(int(nid), None)
                if dst_cluster is None:
                    continue
                dst_color_lab = self.cluster_centers_lab[dst_cluster]
                dist = np.linalg.norm(src_color_lab - dst_color_lab)
                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = nid

            if best_neighbor is not None:
                output_components[mask] = best_neighbor

        # --- Шаг 4: Восстанавливаем RGB-изображение из компонент ---
        self.postprocessed_img = np.zeros_like(self.quant_rgb)
        self.final_colors = {}

        unique_components = np.unique(output_components)
        for comp_id in unique_components:
            if comp_id == 0:
                continue

            cluster_id = comp_to_cluster.get(int(comp_id), None)
            if cluster_id is None:
                continue

            color_lab = self.cluster_centers_lab[cluster_id].copy()

            # LAB -> RGB (как в quantizing)
            L = np.uint8(color_lab[0] * (255.0 / 100.0))
            a = np.uint8(color_lab[1])
            b = np.uint8(color_lab[2])
            lab_px = np.uint8([[[L, a, b]]])
            rgb_px = cv2.cvtColor(lab_px, cv2.COLOR_LAB2RGB)[0, 0]

            self.final_colors[int(comp_id)] = tuple(rgb_px.tolist())
            self.postprocessed_img[output_components == comp_id] = rgb_px

        if self.config['logging']:
            num_components_after = len(self.final_colors)
            print(f"Постобработка завершена. Регионов после: {num_components_after}")
