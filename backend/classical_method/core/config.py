def get_default_algo_config():
    """Возвращает параметры алгоритмов по умолчанию."""
    return {
        'preprocessing': {
            'filter_type': 'pyrMeanShift',  # Варианты: 'pyrMeanShift', 'median', 'bilateral', 'none'
            'pyrMeanShift': {'sp': 5, 'sr': 15},
            'median': {'kernel_size': 3},
            'bilateral': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75}
        },
        'quantizing': {
            'color_space': 'LAB',  # Варианты: 'LAB', 'RGB'
            'kmeans': {
                'max_iter': 40,
                'epsilon': 1.0,
                'attempts': 3
            }
        },
        'postprocessing': {
            'clean_iterations': 3,
            'morph_kernel_size': 3,
            'connectivity': 8
        },
        'contours': {
            'line_thickness': 1,
            'line_color': (0, 0, 0)  # Черный цвет по умолчанию
        }
    }


def update_dict_deep(d, u):
    """Рекурсивное обновление словаря (чтобы не затирать вложенные ключи)."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict_deep(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def validate_and_prepare_config(config):
    """Проверка обязательных параметров и подстановка алгоритмических настроек."""
    required = [
        'img_path', 'target_max_side', 'canvas_width_mm',
        'canvas_height_mm', 'min_diameter_mm', 'colours_cnt', 'logging'
    ]
    for key in required:
        if key not in config:
            raise ValueError(f"Отсутствует обязательный параметр: {key}")

    # Подстановка дефолтных настроек алгоритма
    default_algo = get_default_algo_config()
    user_algo = config.get('algorithm', {})

    config['algorithm'] = update_dict_deep(default_algo, user_algo)
    return config


# (Примечание: в постобработке вам нужно будет убедиться, что восстановление RGB в шаге 4 работает корректно,
# если в шаге квантования был выбран RGB, а не LAB. Либо просто храните оригинальные центры в RGB).
