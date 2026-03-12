import numpy as np
import cv2


def quantize_image(preprocessed_img, config):
    colours_cnt = config['colours_cnt']
    logging = config['logging']
    algo_params = config['algorithm']['quantizing']

    if logging: print('-' * 20, "\nМодуль quantizing\n", '-' * 20)

    color_space = algo_params['color_space']

    # 1. Перевод в выбранное цветовое пространство
    if color_space == 'LAB':
        img_converted = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        # Нормализация L канала для лучшего распределения (опционально)
        img_converted[:, :, 0] *= (100.0 / 255.0)
    else:
        img_converted = preprocessed_img.astype(np.float32)  # RGB

    pixels = img_converted.reshape((-1, 3))

    # 2. Настройка K-Means
    km_params = algo_params['kmeans']
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, km_params['max_iter'], km_params['epsilon'])
    attempts = km_params['attempts']

    if logging: print(f"K-means (K={colours_cnt}, Пространство={color_space}, Attempts={attempts})...")

    _, labels, centers = cv2.kmeans(
        data=pixels, K=colours_cnt, bestLabels=None, criteria=criteria,
        attempts=attempts, flags=cv2.KMEANS_PP_CENTERS
    )

    quantized_converted = centers[labels.flatten()].reshape(img_converted.shape)
    cluster_labels = labels.reshape(preprocessed_img.shape[:2])

    # 3. Возврат в RGB
    if color_space == 'LAB':
        quantized_converted[:, :, 0] *= (255.0 / 100.0)
        quant_rgb = cv2.cvtColor(quantized_converted.astype(np.uint8), cv2.COLOR_LAB2RGB)
    else:
        quant_rgb = quantized_converted.astype(np.uint8)

    if logging: print("Квантование завершено.")
    return quant_rgb, quantized_converted, centers, cluster_labels
