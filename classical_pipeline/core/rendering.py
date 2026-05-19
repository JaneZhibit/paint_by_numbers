import cv2
import numpy as np


def draw_colorized_reference(high_res_labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = high_res_labels.shape
    colorized_image = palette[high_res_labels.flatten()].reshape((h, w, 3))

    # Извлекаем черные границы (скелет)
    edges = np.zeros((h, w), dtype=np.uint8)
    for cls in np.unique(high_res_labels):
        mask = np.uint8(high_res_labels == cls)
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        boundary = mask - eroded
        edges = cv2.bitwise_or(edges, boundary)

    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    # Накладываем границы на цветной холст
    colorized_image[edges == 1] = (0, 0, 0)

    return colorized_image


def draw_labels_on_canvas(canvas: np.ndarray, placements: list) -> np.ndarray:
    result_canvas = canvas.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (130, 130, 130)  # Чуть-чуть светлее, чтобы цифры мягко смотрелись на белом

    for p in placements:
        text = p["text"]
        cx, cy = p["x"], p["y"]
        radius = p["radius"]

        font_scale = max(0.4, radius / 25.0)
        thickness = max(1, int(font_scale * 1.5))

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        text_x = cx - text_w // 2
        text_y = cy + text_h // 2

        cv2.putText(result_canvas, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    return result_canvas