import ml_pipeline.hf_offline  # noqa: F401 — HF offline before any model load
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ml_pipeline.core.rendering import draw_labels_on_canvas
from ml_pipeline.core.vectorization import draw_vectorized_canvas
from ml_pipeline.config import merge_configs
from ml_pipeline.core.labeling import calculate_label_placements
from ml_pipeline.core.postprocessing import apply_postprocessing
from ml_pipeline.core.rendering import draw_colorized_reference
from ml_pipeline.core.vectorization import apply_vectorization
from ml_pipeline.utils.logger import logger, Timer
from ml_pipeline.core.preprocessing import apply_preprocessing
from ml_pipeline.core.split_quantization import apply_split_quantization
from ml_pipeline.core.router import StyleRouter

# Вспомогательная функция слияния из твоего config.py
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class MLPaintPipeline:
    def __init__(self, user_config: dict = None):
        self.config = merge_configs(user_config or {})
        self.timings = {}

        self.original_image = None
        self.preprocessed_image = None

        self.detected_style = None

        # Список семантических объектов: [{'label': 'tree', 'mask': ..., 'area_pct': ...}]
        self.semantic_objects = []
        self._segmenter = None
        self._router = None
        self.protection_mask = None
        self._pose_estimator = None
        self.depth_map = None

        self.quantized_image = None
        self.cluster_labels = None
        self.palette = None

        self.final_labels = None
        self.final_image = None

    def load_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info(f"Изображение загружено. Исходный размер: {self.original_image.shape[:2]}")

        # блок классификации стиля
        if self._router is None:
            self._router = StyleRouter(self.config)

        with Timer("ML_Style_Routing", self.timings):
            style, overrides = self._router.determine_style_and_config(self.original_image)
            self.detected_style = style

            # Применяем переопределенные настройки к нашему глобальному конфигу!
            self.config = deep_update(self.config, overrides)

        return self.original_image

    def preprocess(self):
        with Timer("Preprocessing", self.timings):
            self.preprocessed_image = apply_preprocessing(self.original_image, self.config)

    def segment(self):
        """Этап ML: Семантическая сегментация + (Опционально) Поиск ключевых точек."""
        if self.preprocessed_image is None:
            raise ValueError("Вызовите preprocess() перед сегментацией.")

        if self._segmenter is None:
            from ml_pipeline.core.segmentation import HybridSegmenter
            self._segmenter = HybridSegmenter(self.config)

        with Timer("ML_Hybrid_Segmentation", self.timings):
            self.semantic_objects = self._segmenter.get_hybrid_masks(self.preprocessed_image)

        # === ДИНАМИЧЕСКИЙ ВЫЗОВ POSE ESTIMATION ===
        if self.config.get("use_pose_estimator", False):
            if self._pose_estimator is None:
                from ml_pipeline.core.pose import PoseEstimator
                self._pose_estimator = PoseEstimator(self.config)

            with Timer("ML_Pose_Estimation", self.timings):
                self.protection_mask = self._pose_estimator.get_protection_mask(self.preprocessed_image)
        else:
            # Если это пейзаж или животное, маска защиты пустая
            h, w = self.preprocessed_image.shape[:2]
            self.protection_mask = np.zeros((h, w), dtype=np.uint8)

        if self.config.get("ml_models", {}).get("use_depth", True):
            if not hasattr(self, "_depth_estimator") or getattr(self, "_depth_estimator", None) is None:
                from ml_pipeline.core.depth import DepthEstimator
                self._depth_estimator = DepthEstimator(self.config)

            with Timer("ML_Depth_Estimation", self.timings):
                self.semantic_objects, self.depth_map = self._depth_estimator.assign_depth(
                    self.preprocessed_image, self.semantic_objects
                )
        else:
            h, w = self.preprocessed_image.shape[:2]
            self.depth_map = np.zeros((h, w), dtype=np.uint8)

    def quantize(self):
        """Этап раздельного ML-квантования цветов."""
        if self.preprocessed_image is None:
            raise ValueError("Вызовите preprocess() перед квантованием.")

        if not self.semantic_objects:
            logger.warning("Семантические маски пусты. Запускаем fallback-квантование на всё фото.")


        with Timer("ML_Split_Quantization", self.timings):
            q_img, labels, palette = apply_split_quantization(
                self.preprocessed_image,
                self.semantic_objects,
                self.config
            )

            self.quantized_image = q_img
            self.cluster_labels = labels
            self.palette = palette

    def postprocess(self):
        """Этап постобработки."""
        if self.cluster_labels is None or self.palette is None:
            raise ValueError("Вызовите quantize() перед постобработкой.")

        with Timer("Postprocessing (Spatially Varying)", self.timings):
            labels = self.cluster_labels.copy()
            h, w = labels.shape
            fg_mask = np.zeros((h, w), dtype=np.uint8)
            for obj in self.semantic_objects:
                if obj.get("is_foreground", False):
                    fg_mask[obj["mask"] == 255] = 255

            labels, image_rgb = apply_postprocessing(
                labels,
                self.palette,
                self.config,
                fg_mask,
                self.protection_mask,
                self.depth_map,
            )

            self.final_labels = labels
            self.final_image = image_rgb

    def vectorize(self):
        """Этап векторизации."""
        if self.final_labels is None:
            raise ValueError("Вызовите postprocess() перед векторизацией.")

        with Timer("Vectorization", self.timings):
            # Теперь мы сохраняем матрицу высокого разрешения для расстановки номеров!
            self.high_res_labels, self.polygons = apply_vectorization(self.final_labels, self.config)

            # Рисуем канвас (размер берем от high_res_labels)
            self.vectorized_image = draw_vectorized_canvas(self.high_res_labels)

    def render_all(self):
        """Этап расстановки номеров и генерации финальных изображений."""
        if self.high_res_labels is None or self.polygons is None:
            raise ValueError("Вызовите vectorize() перед render_all().")

        with Timer("Rendering", self.timings):
            # 1. Считаем, куда поставить цифры
            self.labels_placement = calculate_label_placements(self.high_res_labels, self.config)

            # 2. Рисуем цветной референс (High-Res)
            self.colorized_reference = draw_colorized_reference(self.high_res_labels, self.palette)

            # 3. Берем наш белый холст с контурами (из vectorize) и лепим на него цифры
            self.numbered_canvas = draw_labels_on_canvas(self.vectorized_image, self.labels_placement)

        logger.info("Рендеринг завершен. Картина готова!")


    def debug_show_segmentation(self, output_path: str = None):
        """Рисует семантическую карту и карту глубины с легендой."""
        if not self.semantic_objects or self.preprocessed_image is None:
            logger.warning("Нет масок или изображения для отображения.")
            return

        overlay = self.preprocessed_image.copy()
        colored_mask_layer = np.zeros_like(overlay)

        h, w = self.preprocessed_image.shape[:2]
        depth_layer = np.zeros((h, w), dtype=np.uint8)

        cmap = plt.get_cmap("tab20")
        legend_patches = []

        for idx, obj in enumerate(self.semantic_objects):
            color = np.array(cmap(idx % 20)[:3]) * 255
            mask = obj["mask"]
            colored_mask_layer[mask == 255] = color

            depth_layer[mask == 255] = int(obj.get("depth_score", 0))

            legend_patches.append(
                mpatches.Patch(
                    color=color / 255.0,
                    label=f"{obj['label']} (Z={obj.get('depth_score', 0):.0f})",
                )
            )

        mask_exists = np.any(colored_mask_layer > 0, axis=-1)
        blended = overlay.copy()
        blended[mask_exists] = cv2.addWeighted(
            overlay[mask_exists], 0.4,
            colored_mask_layer[mask_exists].astype(np.uint8), 0.6, 0
        )

        depth_heatmap = cv2.applyColorMap(depth_layer, cv2.COLORMAP_INFERNO)
        depth_heatmap = cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        axes[0].imshow(self.preprocessed_image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(blended)
        axes[1].set_title("Semantic Segmentation")
        axes[1].axis("off")
        axes[1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        axes[2].imshow(depth_heatmap)
        axes[2].set_title("Object-Aware Depth Map (Z-Index)")
        axes[2].axis("off")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Результат сохранен в {output_path}")
        plt.close()


    def show_stage(self, stage_name: str, save_path = None):
        """
        Отрисовка нужного этапа с помощью Matplotlib.
        Возможные stage_name: 'original', 'preprocessed'
        """
        mapping = {
            "original": self.original_image,
            "preprocessed": self.preprocessed_image,
            "quantized": self.quantized_image,
            "postprocessed": self.final_image,
            "vectorized": self.vectorized_image,
            "color_reference": self.colorized_reference,
            "numbered": self.numbered_canvas
        }

        img = mapping.get(stage_name)
        if img is None:
            logger.warning(f"Данных для этапа '{stage_name}' пока нет!")
            return

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Этап: {stage_name.upper()}")
        plt.axis("off")  # Отключаем оси с пикселями

        # Для удобства, если ты запускаешь скрипт из консоли
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.close()
