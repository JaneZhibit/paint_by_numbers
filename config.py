import copy


def get_default_config():
    return {
        "canvas": {
            "width_mm": 300,
            "height_mm": 400,
            "min_brush_mm": 3.0,
        },
        "general": {
            "target_max_side_px": 1000,
            "debug_mode": True
        },
        "preprocessing": {
            "smoothing": {
                "method": "mean_shift",
                "params": {
                    "mean_shift": {
                        "spatial_radius": 15,
                        "color_radius": 30
                    },
                    "bilateral": {
                        "d": 9,
                        "sigma_color": 75,
                        "sigma_space": 75
                    }
                }
            },
            "contrast": {
                "method": "clahe",
                "params": {
                    "clahe": {
                        "clip_limit": 0.7,
                        "tile_grid_size": (8, 8)
                    }
                }
            }
        },
        "quantization": {
            "method": "kmeans",
            "colors_count": 16,
            "color_space": "lab",      # 'lab' или 'rgb'
            "params": {
                "kmeans": {
                    "attempts": 10,
                    "criteria_eps": 0.2,
                    "criteria_max_iter": 100
                }
            }
        },
        "postprocessing": {
            "contrast_threshold": 60.0  # Защита контрастных деталей (лиц) от удаления
        },
        "vectorization": {
            "scale_factor": 4,          # Во сколько раз увеличиваем холст. 4 удобно, т.к. каждого пикселя теперь 4
            "blur_size": 15,            # Сила сглаживания "ступенек" (должно быть нечетным)
            "epsilon_factor": 0.0005    # Очень маленький фактор, т.к. разрешение огромное
        }
    }


def merge_configs(user_config: dict) -> dict:
    base = copy.deepcopy(get_default_config())

    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update_dict(base, user_config)