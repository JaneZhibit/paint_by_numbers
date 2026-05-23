import cv2
import numpy as np
import logging

logger = logging.getLogger("PaintByNumbers")


def apply_preprocessing(image: np.ndarray, config: dict) -> np.ndarray:
    """Базовый ресайз и контраст для ML-пайплайна."""
    max_side = config.get("general", {}).get("target_max_side_px", 1000)

    # 1. Ресайз
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Легкий контраст (чтобы выделить объекты для сегментации)
    prep_config = config.get("preprocessing", {})
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=prep_config.get("clip_limit", 1.5),
        tileGridSize=prep_config.get("tile_grid_size", (8, 8))
    )
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return result
