import copy

def get_default_config():
    return {
        "general": {
            "target_max_side_px": 1000,
            "debug_mode": True
        },
        "preprocessing": {
            "clip_limit": 1.9,
            "tile_grid_size": (8, 8)
        },
        "ml_models": {
            "router_model": "openai/clip-vit-base-patch32",  # дирижер
            "foreground_model": "yolo11x-seg.pt",
            "background_model": "openmmlab/upernet-convnext-base", # или base/large
            "pose_model": "yolo11s-pose.pt",  # <--- Модель для поиска лиц
            "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
            "use_depth": True,
            "local_files_only": True,  # Только локальный кэш HF (~/.cache/huggingface), без VPN
            "confidence_threshold": 0.25
        },
         "quantization": {
            "auto_colors": True,
            "fg_threshold": 0.02,
            "bg_threshold": 0.02,
            "total_colors": 62,
            "foreground_colors": 12,
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
            "grow_high_contrast": False,
            # --- Пространственная морфология ---
            "use_depth_morphology": True,
            "brush_sizes_mm": {
                "background": 3,      # Фон (глубокий план, крупная кисть)
                "foreground": 1,      # Объекты (жесткий приоритет fg_mask)
                "face_keypoints": 1.0   # Лица/глаза (абсолютный приоритет protection_mask)
            },
            "face_protection_radius_px": 20
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
