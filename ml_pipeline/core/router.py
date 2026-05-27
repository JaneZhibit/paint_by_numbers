import logging

import ml_pipeline.hf_offline  # noqa: F401 — HF offline env before transformers
import numpy as np
from PIL import Image

from ml_pipeline.hf_offline import create_hf_pipeline

logger = logging.getLogger("PaintByNumbers")


class StyleRouter:
    def __init__(self, config: dict):
        model_name = config.get("ml_models", {}).get("router_model", "openai/clip-vit-base-patch32")
        logger.info(f"Загрузка Style Router (CLIP): {model_name}...")
        self.classifier = create_hf_pipeline(
            "zero-shot-image-classification", model_name, config=config
        )

        self.candidate_labels = [
            "portrait photography",
            "close-up animal photo",
            "landscape scenery",
            "cityscape architecture",
            "still life painting"
        ]

    def determine_style_and_config(self, image: np.ndarray) -> tuple:
        """
        Определяет жанр и возвращает словарь с переопределенными настройками (overrides).
        """
        pil_image = Image.fromarray(image)
        results = self.classifier(pil_image, candidate_labels=self.candidate_labels)

        # Берем Топ-1 и Топ-2
        top1_label = results[0]["label"]
        top1_score = results[0]["score"] * 100
        top2_label = results[1]["label"]
        top2_score = results[1]["score"] * 100

        logger.info(f"Анализ жанра: 1. {top1_label} ({top1_score:.1f}%), 2. {top2_label} ({top2_score:.1f}%)")

        # --- НАША ЭВРИСТИКА ФИЛЬТРАЦИИ ---
        # Очищаем ложный "шум" портрета на пейзажах (как ты заметил в тестах)
        if top1_label == "portrait photography" and top1_score < 40.0 and top2_score > 30.0:
            logger.info("Понижение приоритета портрета (подозрение на ложное срабатывание).")
            final_style = top2_label
        else:
            final_style = top1_label

        logger.info(f"Выбранный режим оркестрации: [{final_style.upper()}]")

        # --- ГЕНЕРАЦИЯ ИДЕАЛЬНЫХ НАСТРОЕК ПОД ЖАНР ---
        config_overrides = {}

        if final_style == "portrait photography":
            config_overrides = {
                "use_pose_estimator": True,  # Включаем поиск глаз/носа
                "quantization": {"fg_threshold": 0.015},  # Даем лицу максимум цветов
            }

        elif final_style == "close-up animal photo":
            config_overrides = {
                "use_pose_estimator": False,  # YOLO-Pose не знает животных
                "quantization": {"fg_threshold": 0.02},
                "postprocessing": {
                    "grow_high_contrast": True,  # Жестко сохраняем усы и блики
                }
            }

        elif final_style in ["landscape scenery", "cityscape architecture"]:
            config_overrides = {
                "use_pose_estimator": False,  # Деревьям глаза не нужны
                "quantization": {"bg_threshold": 0.08},  # Глушим фон, сливаем небо в крупные градиенты
                "postprocessing": {
                    "grow_high_contrast": False  # Отключаем, чтобы не множить пылинки на листьях
                }
            }

        elif final_style == "still life painting":
            config_overrides = {
                "use_pose_estimator": False,
                "quantization": {"fg_threshold": 0.025, "bg_threshold": 0.05},
            }

        return final_style, config_overrides
