import copy

def get_default_config():
    return {
        "general": {
            "target_max_side_px": 1000,
            "debug_mode": True
        },
        "preprocessing": {
            "clip_limit": 1.5,
            "tile_grid_size": (8, 8)
        },
        "ml_models": {
            "foreground_model": "yolo11x-seg.pt",
            "background_model": "openmmlab/upernet-convnext-base", # или base/large
            "pose_model": "yolo11s-pose.pt",  # <--- Модель для поиска лиц
            "confidence_threshold": 0.25
        },
         "quantization": {
            "auto_colors": True,         # Включить автоподбор
            "fg_threshold": 0.02,        # 2% порог для объектов (будет много цветов)
            "bg_threshold": 0.06,        # 6% порог для фона (будет мало цветов)
            "total_colors": 32,          # Лимит (если сумма превысит, мы их урежем пропорционально)
            "foreground_colors": 12,     # Используется как fallback, если auto_colors = False
            "background_colors": 4,
            "color_space": "lab",
            "params": {
                "kmeans": {
                    "attempts": 5,
                    "criteria_eps": 0.5,
                    "criteria_max_iter": 50
                }
            }
        },
        "postprocessing": {
            "contrast_threshold": 60.0,
            "merge_strategy": "color_weighted",
            "grow_high_contrast": True,
            # --- Пространственная морфология ---
            "brush_sizes_mm": {
                "background": 3.5,       # Крупная кисть для фона
                "foreground": 2.5,       # Средняя кисть для объектов/животных
                "face_keypoints": 1.0    # Микро-кисть для глаз/носов
            },
            "face_protection_radius_px": 20 # Радиус защиты вокруг найденных глаз
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
