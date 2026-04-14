import numpy as np
import cv2
from PIL import Image


def preprocess_image(config):
    logging = config['logging']
    algo_params = config['algorithm']['preprocessing']

    if logging: print('-' * 20, "\nМодуль preprocessing\n", '-' * 20)

    original_img = Image.open(config['img_path']).convert("RGB")
    w, h = original_img.size

    # Масштабирование
    target_max = config['target_max_side']
    scale = target_max / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_scaled = original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_bgr = cv2.cvtColor(np.array(img_scaled), cv2.COLOR_RGB2BGR)

    # Выбор алгоритма сглаживания
    filter_type = algo_params['filter_type']
    if logging: print(f"Применение фильтра: {filter_type}...")

    if filter_type == 'pyrMeanShift':
        p = algo_params['pyrMeanShift']
        img_smooth = cv2.pyrMeanShiftFiltering(img_bgr, sp=p['sp'], sr=p['sr'])
    elif filter_type == 'median':
        p = algo_params['median']
        img_smooth = cv2.medianBlur(img_bgr, p['kernel_size'])
    elif filter_type == 'bilateral':
        p = algo_params['bilateral']
        img_smooth = cv2.bilateralFilter(img_bgr, d=p['d'], sigmaColor=p['sigmaColor'], sigmaSpace=p['sigmaSpace'])
    else:
        img_smooth = img_bgr  # Без сглаживания

    preprocessed_img = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB)

    # ============================================================================
    # ГЕНЕРАЦИЯ КАРТЫ ВНИМАНИЯ (SALIENCY MAP)
    # ============================================================================
    saliency_map_config = algo_params.get('saliency_map', {})
    saliency_map_enabled = saliency_map_config.get('enabled', False)
    
    if saliency_map_enabled:
        if logging: print("Генерация карты внимания (Saliency Map)...")
        
        try:
            # Используем StaticSaliencyFineGrained для вычисления карты внимания
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            _, saliency_map = saliency.computeSaliency(img_smooth)
            
            # Нормализуем карту в диапазон [0.0, 1.0]
            saliency_map = saliency_map.astype(np.float32) / 255.0
            
            if logging: print(f"  Карта внимания создана, размер: {saliency_map.shape}")
        except Exception as e:
            if logging: print(f"  Ошибка при создании карты внимания: {e}")
            saliency_map = None
    else:
        saliency_map = None

    if logging: print("Предобработка завершена.")
    return original_img, preprocessed_img, saliency_map
