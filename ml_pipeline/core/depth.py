import cv2
import numpy as np
import logging
from PIL import Image

import ml_pipeline.hf_offline  # noqa: F401
from ml_pipeline.hf_offline import create_hf_pipeline

logger = logging.getLogger("PaintByNumbers")


class DepthEstimator:
    def __init__(self, config: dict):
        model_name = config.get("ml_models", {}).get(
            "depth_model", "depth-anything/Depth-Anything-V2-Small-hf"
        )
        logger.info(f"Загрузка модели оценки глубины: {model_name}...")
        self.model = create_hf_pipeline("depth-estimation", model_name, config=config)

    def assign_depth(self, image: np.ndarray, semantic_objects: list) -> tuple:
        """Рассчитывает карту глубины. Возвращает (semantic_objects, depth_map_normalized)."""
        pil_image = Image.fromarray(image)
        predictions = self.model(pil_image)

        depth_map = np.array(predictions["depth"], dtype=np.float32)
        h, w = image.shape[:2]
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        d_min, d_max = depth_map.min(), depth_map.max()
        depth_norm = (255 * (depth_map - d_min) / (d_max - d_min + 1e-5)).astype(np.uint8)

        logger.info("Z-index объектов (0 - очень далеко, 255 - вплотную к камере):")
        for obj in semantic_objects:
            mask = obj["mask"]
            if np.any(mask == 255):
                median_z = np.median(depth_norm[mask == 255])
                obj["depth_score"] = float(median_z)
            else:
                obj["depth_score"] = 0.0

            logger.info(f" - {obj['label']:<20} : Z={obj['depth_score']:.1f}")

        semantic_objects.sort(
            key=lambda x: (not x.get("is_foreground", False), -x.get("depth_score", 0.0))
        )

        return semantic_objects, depth_norm
