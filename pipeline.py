import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import merge_configs
from utils.logger import logger, Timer
from core.postprocessing import apply_postprocessing
from core.quantization import apply_quantization
from core.vectorization import apply_vectorization, draw_vectorized_canvas
from core.preprocessing import resize_image, apply_smoothing, apply_contrast
from core.labeling import calculate_label_placements
from core.rendering import draw_colorized_reference, draw_labels_on_canvas
from core.export import export_to_svg


class PaintPipeline:
    def __init__(self, user_config: dict = None):
        self.config = merge_configs(user_config or {})
        self.timings = {}

        # Состояние пайплайна
        self.original_image = None
        self.preprocessed_image = None

        self.quantized_image = None
        self.cluster_labels = None
        self.palette = None

        self.final_labels = None
        self.final_image = None

        self.polygons = None
        self.vectorized_image = None
        self.high_res_labels = None

        self.labels_placement = None
        self.colorized_reference = None
        self.numbered_canvas = None

    def load_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        # Чтение файла с поддержкой Unicode
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError(f"Не удалось декодировать изображение: {image_path}. Проверьте формат файла.")

        self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info(f"Изображение загружено: {image_path}. Размер: {self.original_image.shape[:2]}")
        return self.original_image

    def preprocess(self):
        """Этап предобработки, управляемый конфигурацией."""
        if self.original_image is None:
            raise ValueError("Изображение не загружено.")

        max_side = self.config["general"].get("target_max_side_px", 1000)
        prep_config = self.config.get("preprocessing", {})

        with Timer("Preprocessing", self.timings):
            # 1. Ресайз (базовая оптимизация)
            img = resize_image(self.original_image, max_side)

            # 2. Сглаживание (Mean Shift / Bilateral / None)
            img = apply_smoothing(img, prep_config.get("smoothing", {}))

            # 3. Контраст (CLAHE / None)
            img = apply_contrast(img, prep_config.get("contrast", {}))

            self.preprocessed_image = img

        logger.info(f"Предобработка завершена. Итоговый размер: {self.preprocessed_image.shape[:2]}")

    def quantize(self):
        """Этап квантования цветов (поиск палитры)."""
        if self.preprocessed_image is None:
            raise ValueError("Вызовите preprocess() перед квантованием.")

        q_config = self.config.get("quantization", {})
        colors_count = q_config.get("colors_count", 16)

        with Timer("Quantization", self.timings):
            q_img, labels, palette = apply_quantization(self.preprocessed_image, q_config)

            self.quantized_image = q_img
            self.cluster_labels = labels
            self.palette = palette

        logger.info(f"Квантование завершено. Найдено цветов: {colors_count}")

    def postprocess(self):
        """Этап постобработки (очистка от мусора и подготовка к векторизации)."""
        if self.cluster_labels is None or self.palette is None:
            raise ValueError("Вызовите quantize() перед постобработкой.")

        with Timer("Postprocessing", self.timings):
            labels, image_rgb = apply_postprocessing(
                self.cluster_labels,
                self.palette,
                self.config
            )

            self.final_labels = labels
            self.final_image = image_rgb

        logger.info("Постобработка завершена.")

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

    def export(self, output_path: str):
        """Этап экспорта в векторный формат."""
        if self.high_res_labels is None or self.labels_placement is None:
            raise ValueError("Вызовите render_all() перед экспортом.")

        with Timer("Export", self.timings):
            export_to_svg(
                self.polygons,
                self.labels_placement,
                self.high_res_labels.shape,
                output_path
            )

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
        # plt.show()

    def run_all(self, image_path: str):
        """Пример запуска пайплайна"""
        self.load_image(image_path)
        self.preprocess()
        self.quantize()
        self.postprocess()
        self.vectorize()
        self.render_all()
