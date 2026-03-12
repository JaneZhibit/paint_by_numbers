import numpy as np
import cv2


def generate_contours(postprocessed_img, config):
    logging = config['logging']
    algo_params = config['algorithm']['contours']

    if logging: print('-' * 20, "\nМодуль generate_contours\n", '-' * 20)

    h, w = postprocessed_img.shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    pixels = postprocessed_img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    line_thickness = algo_params['line_thickness']
    line_color = algo_params['line_color']

    for color in unique_colors:
        lower = np.array(color, dtype=np.uint8)
        upper = np.array(color, dtype=np.uint8)

        mask = cv2.inRange(postprocessed_img, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, line_color, line_thickness)

    return canvas
