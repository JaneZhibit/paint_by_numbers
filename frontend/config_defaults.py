def get_default_algo_config():
    """Возвращает параметры алгоритмов по умолчанию."""
    return {
        'preprocessing': {
            'filter_type': 'pyrMeanShift',  # Варианты: 'pyrMeanShift', 'median', 'bilateral', 'none'
            'contrast': 1.0,
            'pyrMeanShift': {'sp': 5, 'sr': 15},
            'median': {'kernel_size': 3},
            'bilateral': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
            'saliency_map': {
                'enabled': False,
                'saliency_bg_scale': 1.5,
                'saliency_focus_scale': 0.5
            }
        },
        'quantizing': {
            'color_space': 'LAB',  # Варианты: 'LAB', 'RGB'
            'palette_method': 'kmeans',  # Варианты: 'kmeans', 'median_cut', 'octree'
            'spatial_method': 'none',  # Варианты: 'none', 'slic', 'watershed'
            'kmeans': {
                'max_iter': 40,
                'epsilon': 1.0,
                'attempts': 3
            },
            'weighted_kmeans': {'enabled': True, 'edge_threshold1': 50, 'edge_threshold2': 150, 'weight_factor': 5},
            'superpixels': {
                'enabled': False,
                'region_size': 20,
                'ruler': 15.0
            },
            'auto_color': {
                'enabled': False,
                'k_min': 10,
                'k_max': 26,
                'k_step': 2,
                'subsample_ratio': 0.2,
                'stop_threshold_percent': 5.0
            },
            'hierarchical_split': {'enabled': False, 'min_area_pixels': 2000, 'variance_threshold': 30.0}
        },
        'postprocessing': {
            'clean_iterations': 3,
            'morph_kernel_size': 3,
            'connectivity': 8,
            'contrast_threshold': 100.0,
            'grow_details': {'enabled': False, 'max_dilation_steps': 5}
        },
        'vectorization': {
            'method': 'raster_shift',
            'smoothing': {
                'enabled': True,
                'method': 'approx_poly_dp',
                'approx_poly_dp': {'epsilon_factor': 0.005},
                'chaikin': {'iterations': 2}
            }
        }
    }


# ============================================================================
# МЕТАДАННЫЕ ПАРАМЕТРОВ ДЛЯ ГЕНЕРАЦИИ UI
# ============================================================================

UI_PARAMS_META = [
    # Предобработка
    {
        "param_path": "preprocessing.filter_type",
        "label": "Метод сглаживания",
        "type": "select",
        "options": ["pyrMeanShift", "bilateral", "median", "none"],
        "default": "pyrMeanShift",
        "help_key": "filter_type",
        "section": "Предобработка"
    },
    {
        "param_path": "preprocessing.contrast",
        "label": "Усиление контраста (x)",
        "type": "float",
        "min": 1.0,
        "max": 2.0,
        "step": 0.1,
        "default": 1.0,
        "help_key": None,
        "section": "Предобработка"
    },
    {
        "param_path": "preprocessing.pyrMeanShift.sp",
        "label": "pyrMeanShift: Spatial Radius (sp)",
        "type": "int",
        "min": 2,
        "max": 30,
        "step": 1,
        "default": 5,
        "help_key": None,
        "section": "Предобработка",
        "depends_on": {"path": "preprocessing.filter_type", "value": "pyrMeanShift"}
    },
    {
        "param_path": "preprocessing.pyrMeanShift.sr",
        "label": "pyrMeanShift: Color Radius (sr)",
        "type": "int",
        "min": 5,
        "max": 60,
        "step": 5,
        "default": 15,
        "help_key": None,
        "section": "Предобработка",
        "depends_on": {"path": "preprocessing.filter_type", "value": "pyrMeanShift"}
    },
    {
        "param_path": "preprocessing.bilateral.d",
        "label": "Bilateral: диаметр (d)",
        "type": "int",
        "min": 3,
        "max": 25,
        "step": 2,
        "default": 9,
        "help_key": None,
        "section": "Предобработка",
        "depends_on": {"path": "preprocessing.filter_type", "value": "bilateral"}
    },
    {
        "param_path": "preprocessing.bilateral.sigmaColor",
        "label": "Bilateral: sigmaColor",
        "type": "int",
        "min": 10,
        "max": 200,
        "step": 5,
        "default": 75,
        "help_key": None,
        "section": "Предобработка",
        "depends_on": {"path": "preprocessing.filter_type", "value": "bilateral"}
    },
    {
        "param_path": "preprocessing.bilateral.sigmaSpace",
        "label": "Bilateral: sigmaSpace",
        "type": "int",
        "min": 10,
        "max": 200,
        "step": 5,
        "default": 75,
        "help_key": None,
        "section": "Предобработка",
        "depends_on": {"path": "preprocessing.filter_type", "value": "bilateral"}
    },
    {
        "param_path": "preprocessing.median.kernel_size",
        "label": "Median: размер ядра",
        "type": "int",
        "min": 3,
        "max": 15,
        "step": 2,
        "default": 3,
        "help_key": None,
        "section": "Предобработка",
        "depends_on": {"path": "preprocessing.filter_type", "value": "median"}
    },
    {
        "param_path": "preprocessing.saliency_map.enabled",
        "label": "Карта внимания (Saliency Map)",
        "type": "bool",
        "default": False,
        "help_key": "saliency_map",
        "section": "Предобработка"
    },
    {
        "param_path": "preprocessing.saliency_map.saliency_bg_scale",
        "label": "Saliency: масштаб кисти на фоне",
        "type": "float",
        "min": 0.1,
        "max": 3.0,
        "step": 0.1,
        "default": 1.5,
        "help_key": None,
        "section": "Предобработка",
        "depends_on": {"path": "preprocessing.saliency_map.enabled", "value": True}
    },
    {
        "param_path": "preprocessing.saliency_map.saliency_focus_scale",
        "label": "Saliency: масштаб кисти на объекте",
        "type": "float",
        "min": 0.1,
        "max": 3.0,
        "step": 0.1,
        "default": 0.5,
        "help_key": None,
        "section": "Предобработка",
        "depends_on": {"path": "preprocessing.saliency_map.enabled", "value": True}
    },
    
    # Квантование
    {
        "param_path": "quantizing.palette_method",
        "label": "Метод палитры",
        "type": "select",
        "options": ["kmeans", "median_cut", "octree"],
        "default": "kmeans",
        "help_key": "palette_method",
        "section": "Квантование"
    },
    {
        "param_path": "quantizing.spatial_method",
        "label": "Метод сегментации",
        "type": "select",
        "options": ["none", "slic", "watershed"],
        "default": "none",
        "help_key": "spatial_method",
        "section": "Квантование"
    },
    {
        "param_path": "quantizing.weighted_kmeans.enabled",
        "label": "Взвешенный K-Means",
        "type": "bool",
        "default": True,
        "help_key": "weighted_kmeans",
        "section": "Квантование",
        "depends_on": {"path": "quantizing.palette_method", "value": "kmeans"}
    },
    {
        "param_path": "quantizing.weighted_kmeans.edge_threshold1",
        "label": "Canny: нижний порог",
        "type": "int",
        "min": 10,
        "max": 200,
        "step": 10,
        "default": 50,
        "help_key": None,
        "section": "Квантование",
        "depends_on": {"path": "quantizing.weighted_kmeans.enabled", "value": True}
    },
    {
        "param_path": "quantizing.weighted_kmeans.edge_threshold2",
        "label": "Canny: верхний порог",
        "type": "int",
        "min": 50,
        "max": 400,
        "step": 10,
        "default": 150,
        "help_key": None,
        "section": "Квантование",
        "depends_on": {"path": "quantizing.weighted_kmeans.enabled", "value": True}
    },
    {
        "param_path": "quantizing.weighted_kmeans.weight_factor",
        "label": "Вес контуров (×)",
        "type": "int",
        "min": 1,
        "max": 20,
        "step": 1,
        "default": 5,
        "help_key": None,
        "section": "Квантование",
        "depends_on": {"path": "quantizing.weighted_kmeans.enabled", "value": True}
    },
    {
        "param_path": "quantizing.superpixels.region_size",
        "label": "SLIC: размер региона (px)",
        "type": "int",
        "min": 5,
        "max": 80,
        "step": 5,
        "default": 20,
        "help_key": None,
        "section": "Квантование / SLIC",
        "depends_on": {"path": "quantizing.spatial_method", "value": "slic"}
    },
    {
        "param_path": "quantizing.superpixels.ruler",
        "label": "SLIC: ruler (форма)",
        "type": "float",
        "min": 1.0,
        "max": 40.0,
        "step": 1.0,
        "default": 15.0,
        "help_key": None,
        "section": "Квантование / SLIC",
        "depends_on": {"path": "quantizing.spatial_method", "value": "slic"}
    },
    {
        "param_path": "quantizing.auto_color.enabled",
        "label": "Авто-цвет (Elbow)",
        "type": "bool",
        "default": False,
        "help_key": "auto_color",
        "section": "Квантование / Авто-цвет"
    },
    {
        "param_path": "quantizing.auto_color.k_min",
        "label": "Авто-цвет: K минимум",
        "type": "int",
        "min": 4,
        "max": 16,
        "step": 2,
        "default": 10,
        "help_key": None,
        "section": "Квантование / Авто-цвет",
        "depends_on": {"path": "quantizing.auto_color.enabled", "value": True}
    },
    {
        "param_path": "quantizing.auto_color.k_max",
        "label": "Авто-цвет: K максимум",
        "type": "int",
        "min": 12,
        "max": 40,
        "step": 2,
        "default": 26,
        "help_key": None,
        "section": "Квантование / Авто-цвет",
        "depends_on": {"path": "quantizing.auto_color.enabled", "value": True}
    },
    {
        "param_path": "quantizing.auto_color.stop_threshold_percent",
        "label": "Авто-цвет: порог остановки (%)",
        "type": "float",
        "min": 1.0,
        "max": 15.0,
        "step": 1.0,
        "default": 5.0,
        "help_key": None,
        "section": "Квантование / Авто-цвет",
        "depends_on": {"path": "quantizing.auto_color.enabled", "value": True}
    },
    {
        "param_path": "quantizing.hierarchical_split.enabled",
        "label": "Иерархическое расщепление",
        "type": "bool",
        "default": False,
        "help_key": "hierarchical_split",
        "section": "Квантование / Доп.",
        "depends_on": {"path": "quantizing.palette_method", "value": "kmeans"}
    },
    {
        "param_path": "quantizing.hierarchical_split.min_area_pixels",
        "label": "Мин. площадь пятна (px)",
        "type": "int",
        "min": 200,
        "max": 10000,
        "step": 200,
        "default": 2000,
        "help_key": None,
        "section": "Квантование / Доп.",
        "depends_on": {"path": "quantizing.hierarchical_split.enabled", "value": True}
    },
    {
        "param_path": "quantizing.hierarchical_split.variance_threshold",
        "label": "Порог дисперсии",
        "type": "float",
        "min": 5.0,
        "max": 100.0,
        "step": 5.0,
        "default": 30.0,
        "help_key": None,
        "section": "Квантование / Доп.",
        "depends_on": {"path": "quantizing.hierarchical_split.enabled", "value": True}
    },
    
    # Постобработка
    {
        "param_path": "postprocessing.clean_iterations",
        "label": "Итерации очистки",
        "type": "int",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 3,
        "help_key": None,
        "section": "Постобработка"
    },
    {
        "param_path": "postprocessing.morph_kernel_size",
        "label": "Размер морф. ядра",
        "type": "int",
        "min": 3,
        "max": 15,
        "step": 2,
        "default": 3,
        "help_key": None,
        "section": "Постобработка"
    },
    {
        "param_path": "postprocessing.contrast_threshold",
        "label": "Порог контраста",
        "type": "float",
        "min": 10.0,
        "max": 200.0,
        "step": 5.0,
        "default": 100.0,
        "help_key": None,
        "section": "Постобработка"
    },
    {
        "param_path": "postprocessing.grow_details.enabled",
        "label": "Выращивание деталей",
        "type": "bool",
        "default": False,
        "help_key": "grow_details",
        "section": "Постобработка"
    },
    {
        "param_path": "postprocessing.grow_details.max_dilation_steps",
        "label": "Макс. шагов расширения",
        "type": "int",
        "min": 1,
        "max": 20,
        "step": 1,
        "default": 5,
        "help_key": None,
        "section": "Постобработка",
        "depends_on": {"path": "postprocessing.grow_details.enabled", "value": True}
    },

    # Векторизация / Контуры
    {
        "param_path": "vectorization.method",
        "label": "Метод векторизации",
        "type": "select",
        "options": ["raster_shift", "contours"],
        "default": "raster_shift",
        "help_key": None,
        "section": "Векторизация / Контуры"
    },
    {
        "param_path": "vectorization.smoothing.enabled",
        "label": "Сглаживание контуров",
        "type": "bool",
        "default": True,
        "help_key": None,
        "section": "Векторизация / Контуры",
        "depends_on": {"path": "vectorization.method", "value": "contours"}
    },
    {
        "param_path": "vectorization.smoothing.method",
        "label": "Метод сглаживания контуров",
        "type": "select",
        "options": ["approx_poly_dp", "chaikin"],
        "default": "approx_poly_dp",
        "help_key": None,
        "section": "Векторизация / Контуры",
        "depends_on": {"path": "vectorization.smoothing.enabled", "value": True}
    },
    {
        "param_path": "vectorization.smoothing.approx_poly_dp.epsilon_factor",
        "label": "Douglas-Peucker: epsilon factor",
        "type": "float",
        "min": 0.001,
        "max": 0.02,
        "step": 0.001,
        "default": 0.005,
        "help_key": None,
        "section": "Векторизация / Контуры",
        "depends_on": {"path": "vectorization.smoothing.method", "value": "approx_poly_dp"}
    },
    {
        "param_path": "vectorization.smoothing.chaikin.iterations",
        "label": "Chaikin: итерации скругления",
        "type": "int",
        "min": 1,
        "max": 6,
        "step": 1,
        "default": 2,
        "help_key": None,
        "section": "Векторизация / Контуры",
        "depends_on": {"path": "vectorization.smoothing.method", "value": "chaikin"}
    },
]


def get_param_by_path(param_path: str) -> dict:
    """
    Возвращает метаданные параметра по его dot-path.
    
    Args:
        param_path: Путь вида "preprocessing.filter_type"
    
    Returns:
        Словарь метаданных или None, если параметр не найден
    """
    for param in UI_PARAMS_META:
        if param["param_path"] == param_path:
            return param
    return None


def get_params_by_section(section: str) -> list:
    """
    Возвращает все параметры из указанной секции.
    
    Args:
        section: Название секции (например, "Предобработка")
    
    Returns:
        Список метаданных параметров
    """
    return [p for p in UI_PARAMS_META if p["section"] == section]


def get_all_sections() -> list:
    """Возвращает список всех уникальных секций."""
    sections = []
    seen = set()
    for param in UI_PARAMS_META:
        section = param["section"]
        if section not in seen:
            sections.append(section)
            seen.add(section)
    return sections
