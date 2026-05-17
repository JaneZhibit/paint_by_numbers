import cv2
import numpy as np


def resize_image(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def apply_smoothing(image: np.ndarray, config: dict) -> np.ndarray:
    """Диспетчер для этапа сглаживания."""
    method = config.get("method", "none")
    params = config.get("params", {}).get(method, {})

    if method == "mean_shift":
        sp = params.get("spatial_radius", 10)
        sr = params.get("color_radius", 20)
        return cv2.pyrMeanShiftFiltering(image, sp, sr)

    elif method == "bilateral":
        d = params.get("d", 9)
        sc = params.get("sigma_color", 75)
        ss = params.get("sigma_space", 75)
        return cv2.bilateralFilter(image, d, sc, ss)

    elif method == "none":
        return image

    else:
        raise ValueError(f"Неизвестный метод сглаживания: {method}")


def apply_contrast(image: np.ndarray, config: dict) -> np.ndarray:
    """Диспетчер для этапа улучшения контраста."""
    method = config.get("method", "none")
    params = config.get("params", {}).get(method, {})

    if method == "clahe":
        # CLAHE лучше всего работает в цветовом пространстве LAB (канал L)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=params.get("clip_limit", 2.0),
            tileGridSize=params.get("tile_grid_size", (8, 8))
        )
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    elif method == "none":
        return image

    else:
        raise ValueError(f"Неизвестный метод контраста: {method}")