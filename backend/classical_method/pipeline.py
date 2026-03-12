from core.config import validate_and_prepare_config
from core.preprocessing import preprocess_image
from core.quantizing import quantize_image
from core.postprocessing import postprocess_image
from core.contours import generate_contours


class ClassicalPaintByNumbers:
    def __init__(self, config):
        self.config = validate_and_prepare_config(config)

        self.original_img = None
        self.preprocessed_img = None
        self.quant_rgb = None
        self.quant_space = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.postprocessed_img = None
        self.final_colors = None
        self.contours_img = None  # Храним контуры

    def preprocessing(self):
        self.original_img, self.preprocessed_img = preprocess_image(self.config)

    def quantizing(self):
        res = quantize_image(self.preprocessed_img, self.config)
        self.quant_rgb, self.quant_space, self.cluster_centers, self.cluster_labels = res

    def postprocessing(self):
        res = postprocess_image(self.quant_rgb, self.cluster_labels, self.cluster_centers, self.config)
        self.postprocessed_img, self.final_colors, self.cluster_labels = res

    def generate_contours(self):
        self.contours_img = generate_contours(self.postprocessed_img, self.config)
        return self.contours_img

    def run_all(self):
        """Удобный метод, чтобы прогнать весь пайплайн разом."""
        self.preprocessing()
        self.quantizing()
        self.postprocessing()
        self.generate_contours()
