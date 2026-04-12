import numpy as np
import cv2


def generate_contours(postprocessed_img, output_components, config):
    logging = config['logging']
    algo_params = config['algorithm']['contours']

    if logging: print('-' * 20, "\nМодуль generate_contours\n", '-' * 20)

    h, w = postprocessed_img.shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    line_thickness = algo_params['line_thickness']
    line_color = algo_params['line_color']

    # Итерируемся по компонентам напрямую (быстрее чем inRange по цветам)
    unique_components = np.unique(output_components)
    for comp_id in unique_components:
        if comp_id == 0:
            continue

        mask = (output_components == comp_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, line_color, line_thickness)

    return canvas
