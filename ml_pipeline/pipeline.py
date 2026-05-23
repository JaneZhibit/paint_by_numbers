import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.rendering import draw_labels_on_canvas
from core.vectorization import draw_vectorized_canvas
from ml_pipeline.config import merge_configs
from ml_pipeline.core.labeling import calculate_label_placements
from ml_pipeline.core.postprocessing import apply_postprocessing
from ml_pipeline.core.rendering import draw_colorized_reference
from ml_pipeline.core.vectorization import apply_vectorization
from ml_pipeline.utils.logger import logger, Timer
from ml_pipeline.core.preprocessing import apply_preprocessing
from ml_pipeline.core.split_quantization import apply_split_quantization
from ml_pipeline.core.segmentation import HybridSegmenter


class MLPaintPipeline:
    def __init__(self, user_config: dict = None):
        self.config = merge_configs(user_config or {})
        self.timings = {}

        self.original_image = None
        self.preprocessed_image = None

        # Список семантических объектов: [{'label': 'tree', 'mask': ..., 'area_pct': ...}]
        self.semantic_objects = []
        self._segmenter = None
        self.protection_mask = None
        self._pose_estimator = None

        self.quantized_image = None
        self.cluster_labels = None
        self.palette = None

        self.final_labels = None
        self.final_image = None

    def load_image(self, image_path: str):
        # ... (Код без изменений) ...
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.original_image

    def preprocess(self):
        # ... (Код без изменений) ...
        self.preprocessed_image = apply_preprocessing(self.original_image, self.config)

    def segment(self):
        """Этап ML: Семантическая сегментация + Поиск ключевых точек (Лиц)."""
        if self.preprocessed_image is None:
            raise ValueError("Вызовите preprocess() перед сегментацией.")

        if self._segmenter is None:
            from ml_pipeline.core.segmentation import HybridSegmenter
            self._segmenter = HybridSegmenter(self.config)

        if self._pose_estimator is None:
            from ml_pipeline.core.pose import PoseEstimator
            self._pose_estimator = PoseEstimator(self.config)

        with Timer("ML_Hybrid_Segmentation & Pose", self.timings):
            # Сегментируем объекты
            self.semantic_objects = self._segmenter.get_hybrid_masks(self.preprocessed_image)
            # Ищем лица
            self.protection_mask = self._pose_estimator.get_protection_mask(self.preprocessed_image)

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

            # Создаем объединенную маску Foreground из объектов семантики
            h, w = labels.shape
            fg_mask = np.zeros((h, w), dtype=np.uint8)
            for obj in self.semantic_objects:
                if obj.get("is_foreground", False):
                    fg_mask[obj["mask"] == 255] = 255

            labels, image_rgb = apply_postprocessing(
                labels, self.palette, self.config,
                fg_mask, self.protection_mask  # <--- Передаем наши маски!
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
        """Рисует семантическую карту с легендой."""
        if not self.semantic_objects or self.preprocessed_image is None:
            logger.warning("Нет масок или изображения для отображения.")
            return

        overlay = self.preprocessed_image.copy()
        colored_mask_layer = np.zeros_like(overlay)

        # Генерация ярких цветов для классов
        cmap = plt.get_cmap("tab20")
        legend_patches = []

        for idx, obj in enumerate(self.semantic_objects):
            # Получаем цвет из палитры matplotlib [0, 1] -> [0, 255]
            color = np.array(cmap(idx % 20)[:3]) * 255
            mask = obj["mask"]

            colored_mask_layer[mask == 255] = color

            # Создаем патч для легенды графика
            legend_patches.append(
                mpatches.Patch(color=color / 255.0, label=f"{obj['label']} ({obj['area_pct']:.1f}%)")
            )

        mask_exists = np.any(colored_mask_layer > 0, axis=-1)

        blended = overlay.copy()
        blended[mask_exists] = cv2.addWeighted(
            overlay[mask_exists], 0.4,
            colored_mask_layer[mask_exists].astype(np.uint8), 0.6, 0
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(self.preprocessed_image)
        axes[0].set_title("Original (Preprocessed)")
        axes[0].axis("off")

        axes[1].imshow(blended)
        axes[1].set_title(f"Semantic Segmentation (SegFormer ADE20K)")
        axes[1].axis("off")

        # Добавляем легенду с названиями классов!
        axes[1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

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
            #"numbered": self.numbered_canvas
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
