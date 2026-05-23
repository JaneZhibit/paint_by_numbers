import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger("PaintByNumbers")


class HybridSegmenter:
    def __init__(self, config: dict):
        ml_config = config.get("ml_models", {})
        fg_model_name = ml_config.get("foreground_model", "yolo11n-seg.pt")
        bg_model_name = ml_config.get("background_model", "openmmlab/upernet-convnext-tiny")
        self.conf = ml_config.get("confidence_threshold", 0.25)

        logger.info(f"Загрузка Hybrid Segmenter...")
        logger.info(f" -> Foreground (YOLO): {fg_model_name}")
        logger.info(f" -> Background (Semantic): {bg_model_name}")

        from ultralytics import YOLO
        from transformers import pipeline

        # Грузим обе модели
        self.fg_model = YOLO(fg_model_name)
        self.bg_model = pipeline("image-segmentation", model=bg_model_name)

    def get_hybrid_masks(self, image: np.ndarray) -> list:
        """
        Возвращает список: [{'label': 'name', 'mask': np.array, 'area_pct': float, 'is_foreground': bool}]
        """
        h, w = image.shape[:2]
        total_pixels = h * w
        hybrid_results = []

        # 1. ЗАПУСК YOLO (FOREGROUND)
        yolo_results = self.fg_model.predict(image, conf=self.conf, verbose=False)[0]

        # Маска, в которой мы соберем ВСЕ объекты переднего плана (чтобы потом вычесть из фона)
        global_fg_mask = np.zeros((h, w), dtype=np.uint8)

        if yolo_results.masks is not None:
            masks_data = yolo_results.masks.data.cpu().numpy()
            class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
            class_names = yolo_results.names

            unique_classes = np.unique(class_ids)
            for cls_id in unique_classes:
                label = class_names[cls_id]
                class_mask_stack = masks_data[class_ids == cls_id]
                combined_mask = np.max(class_mask_stack, axis=0)

                mask_resized = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                binary_mask = np.where(mask_resized > 0.5, 255, 0).astype(np.uint8)

                area_pct = (np.count_nonzero(binary_mask) / total_pixels) * 100
                if area_pct > 0.1:
                    hybrid_results.append({
                        "label": f"[FG] {label}",
                        "mask": binary_mask,
                        "area_pct": area_pct,
                        "is_foreground": True
                    })
                    # Добавляем в глобальную маску переднего плана
                    global_fg_mask[binary_mask == 255] = 255

        # 2. ЗАПУСК SEMANTIC MODEL (BACKGROUND)
        pil_image = Image.fromarray(image)
        hf_results = self.bg_model(pil_image)

        for res in hf_results:
            label = res["label"]
            mask_np = np.array(res["mask"])
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            binary_mask = np.where(mask_resized > 128, 255, 0).astype(np.uint8)

            # ВЫЧИТАНИЕ ГРАНИЦ!
            # Затираем те места фона, где YOLO уже нашел человека/собаку
            binary_mask[global_fg_mask == 255] = 0

            area_pct = (np.count_nonzero(binary_mask) / total_pixels) * 100

            # Оставляем только значимые куски фона
            if area_pct > 1.0:
                hybrid_results.append({
                    "label": f"[BG] {label}",
                    "mask": binary_mask,
                    "area_pct": area_pct,
                    "is_foreground": False
                })

        # Сортируем (сначала передний план, потом по размеру)
        hybrid_results = sorted(hybrid_results, key=lambda x: (not x["is_foreground"], -x["area_pct"]))

        logger.info(f"Гибридная сегментация завершена. Найдено объектов: {len(hybrid_results)}")
        for item in hybrid_results:
            logger.info(f" - {item['label']}: {item['area_pct']:.1f}%")

        return hybrid_results
    