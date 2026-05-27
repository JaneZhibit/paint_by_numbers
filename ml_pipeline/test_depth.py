import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
import time

def test_depth_estimation():
    # Модель Depth Anything V2 (Small)
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Загрузка модели оценки глубины: {model_name}...")
    depth_estimator = pipeline("depth-estimation", model=model_name)

    # Список тестовых изображений (пути с кириллицей и пробелами – ок, т.к. читаем через imdecode)
    test_images = [
        "test_images/масштаб 4 к 3/animal_1.jpg",
        "test_images/масштаб 4 к 3/animal_2.jpg",
        "test_images/масштаб 4 к 3/animal_3.jpg",
        "test_images/масштаб 4 к 3/animal_4.jpg",
        "test_images/масштаб 4 к 3/animal_5.jpg",
        "test_images/масштаб 4 к 3/landscape_1.jpg",
        "test_images/масштаб 4 к 3/landscape_2.jpg",
        "test_images/масштаб 4 к 3/landscape_3.jpg",
        "test_images/масштаб 4 к 3/landscape_4.jpg",
        "test_images/масштаб 4 к 3/landscape_5.jpg",
        "test_images/масштаб 4 к 3/naturmort_1.jpg",
        "test_images/масштаб 4 к 3/naturmort_2.jpg",
        "test_images/масштаб 4 к 3/naturmort_3.jpg",
        "test_images/масштаб 4 к 3/naturmort_4.jpg",
        "test_images/масштаб 4 к 3/naturmort_5.jpg",
        "test_images/масштаб 4 к 3/portret_1.jpg",
        "test_images/масштаб 4 к 3/portret_2.jpg",
        "test_images/масштаб 4 к 3/portret_3.jpg",
        "test_images/масштаб 4 к 3/portret_4.jpg",
        "test_images/масштаб 4 к 3/portret_5.jpg",
        "test_images/масштаб 4 к 3/jane.jpg",
        "test_images/масштаб 4 к 3/andrey.jpg",
        "test_images/масштаб 4 к 3/andrey_2.jpg",
        "test_images/масштаб 4 к 3/andrey_dog.jpg",
        "test_images/масштаб 4 к 3/andrey_dog_2.jpg",
        "test_images/масштаб 4 к 3/pacific.jpg"
    ]

    os.makedirs("results/depth_tests", exist_ok=True)

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")
            continue

        print(f"\nАнализ глубины для: {os.path.basename(img_path)}")

        # --- Чтение изображения с поддержкой кириллицы и пробелов ---
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Не удалось декодировать: {img_path}")
            continue
        original_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(original_img)

        # --- Запуск модели глубины ---
        start_time = time.time()
        predictions = depth_estimator(pil_image)
        elapsed = time.time() - start_time
        print(f"Время обработки: {elapsed:.2f} сек")

        # --- Извлечение карты глубины (uint8, 0-255) ---
        # Модель возвращает PIL Image, преобразуем в numpy
        depth_pil = predictions["depth"]
        depth_map = np.array(depth_pil, dtype=np.uint8)

        # Приводим к размеру оригинала (на всякий случай)
        h, w = original_img.shape[:2]
        if depth_map.shape[:2] != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Нормализация уже не нужна – модель выдаёт uint8 0..255, но для красоты сделаем процентильное растяжение
        # Это улучшит контраст на изображениях с узкой гистограммой
        p_low, p_high = np.percentile(depth_map, (2, 98))
        depth_norm = np.clip((depth_map - p_low) / (p_high - p_low + 1e-6), 0, 1)
        depth_norm_uint8 = (depth_norm * 255).astype(np.uint8)

        # --- Тепловая карта (colormap inferno) через OpenCV ---
        heatmap_cv = cv2.applyColorMap(depth_norm_uint8, cv2.COLORMAP_INFERNO)
        heatmap_cv_rgb = cv2.cvtColor(heatmap_cv, cv2.COLOR_BGR2RGB)

        # --- Адаптивное разбиение на три плана (по процентилям) ---
        # Используем распределение глубины на текущем изображении
        flat = depth_norm_uint8.flatten()
        threshold_bg = np.percentile(flat, 33)   # нижняя треть -> дальний фон
        threshold_fg = np.percentile(flat, 66)   # верхняя треть -> передний план

        bg_mask = (depth_norm_uint8 <= threshold_bg).astype(np.uint8) * 255
        mg_mask = ((depth_norm_uint8 > threshold_bg) & (depth_norm_uint8 <= threshold_fg)).astype(np.uint8) * 255
        fg_mask = (depth_norm_uint8 > threshold_fg).astype(np.uint8) * 255

        # --- Функция для наложения маски с ярким цветом ---
        def overlay_mask(img, mask, color, alpha=0.6):
            overlay = img.copy()
            overlay[mask == 255] = color
            return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        # Накладываем маски на оригинал
        fg_vis = overlay_mask(original_img, fg_mask, (255, 80, 80))   # красный
        mg_vis = overlay_mask(original_img, mg_mask, (80, 255, 80))   # зелёный
        bg_vis = overlay_mask(original_img, bg_mask, (80, 80, 255))   # синий

        # --- Визуализация в matplotlib и сохранение ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Оригинал")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(heatmap_cv_rgb)
        axes[0, 1].set_title("Тепловая карта глубины (Inferno)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(depth_norm_uint8, cmap='gray')
        axes[0, 2].set_title("Монохромная глубина")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(fg_vis)
        axes[1, 0].set_title(f"Передний план ( > {threshold_fg} )")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(mg_vis)
        axes[1, 1].set_title(f"Средний план ({threshold_bg} – {threshold_fg})")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(bg_vis)
        axes[1, 2].set_title(f"Дальний фон ( ≤ {threshold_bg} )")
        axes[1, 2].axis("off")

        plt.tight_layout()
        out_filename = f"results/depth_tests/depth_{os.path.basename(img_path)}"
        plt.savefig(out_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Результат сохранён: {out_filename}")

        # Дополнительно выводим информацию о распределении глубины
        print(f"  Глубина: min={depth_map.min()}, max={depth_map.max()}, mean={depth_map.mean():.1f}")
        print(f"  Пороги: фон ≤ {threshold_bg}, передний план > {threshold_fg}")

if __name__ == "__main__":
    print("Текущая рабочая папка:", os.getcwd())
    test_depth_estimation()
