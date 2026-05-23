import cv2
import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger("PaintByNumbers")


class PoseEstimator:
    def __init__(self, config: dict):
        model_name = config.get("ml_models", {}).get("pose_model", "yolo11n-pose.pt")
        logger.info(f"Загрузка модели поиска ключевых точек: {model_name}...")
        self.model = YOLO(model_name)

        self.radius = config.get("postprocessing", {}).get("face_protection_radius_px", 15)

    def get_protection_mask(self, image: np.ndarray) -> np.ndarray:
        """Возвращает бинарную маску зон, которые запрещено агрессивно сглаживать."""
        h, w = image.shape[:2]
        protection_mask = np.zeros((h, w), dtype=np.uint8)

        results = self.model.predict(image, conf=0.3, verbose=False)[0]

        if results.keypoints is not None and len(results.keypoints) > 0:
            kpts_data = results.keypoints.data.cpu().numpy()

            faces_found = 0
            for person_kpts in kpts_data:
                # 0: Нос, 1: Левый глаз, 2: Правый глаз
                for pt_idx in [0, 1, 2]:
                    x, y, conf = person_kpts[pt_idx]
                    if conf > 0.5:
                        cv2.circle(protection_mask, (int(x), int(y)), self.radius, 255, -1)
                        faces_found += 1

            if faces_found > 0:
                logger.info(f"Найдено {faces_found} ключевых точек лица. Зоны защищены.")

        return protection_mask