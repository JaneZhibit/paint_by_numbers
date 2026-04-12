from core.config import validate_and_prepare_config
from core.preprocessing import preprocess_image
from core.quantizing import quantize_image
from core.postprocessing import postprocess_image
from core.contours import generate_contours
from core.numbering import add_numbers
from core.export import export_svg, export_pdf
from core.layout import compute_label_layout
from core.legend import generate_legend


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
        self.color_index_map = None
        self.output_components = None
        self.comp_to_cluster = None
        self.comp_sizes = None
        self.contours_img = None  # Храним контуры
        self.numbered_img = None
        self.label_layout = None
        self.legend_img = None

    def preprocessing(self):
        self.original_img, self.preprocessed_img = preprocess_image(self.config)

    def quantizing(self):
        res = quantize_image(self.preprocessed_img, self.config)
        self.quant_rgb, self.quant_space, self.cluster_centers, self.cluster_labels = res

    def postprocessing(self):
        res = postprocess_image(self.quant_rgb, self.cluster_labels, self.cluster_centers, self.config)
        self.postprocessed_img, self.final_colors, self.cluster_labels, self.color_index_map, self.output_components, self.comp_to_cluster, self.comp_sizes = res

    def generate_contours(self):
        self.contours_img = generate_contours(self.postprocessed_img, self.config)
        return self.contours_img

    def add_numbers(self):
        self.label_layout = compute_label_layout(
            self.output_components,
            self.color_index_map,
            self.comp_to_cluster,
            self.comp_sizes,
            self.final_colors,
            self.config
        )
        self.numbered_img = add_numbers(
            self.contours_img,
            self.label_layout
        )

    def export_svg(self, save_path: str):
        export_svg(
            self.output_components,
            self.label_layout,
            self.config,
            save_path
        )

    def generate_legend(self, save_path: str):
        self.legend_img = generate_legend(
            self.label_layout,
            self.config,
            save_path
        )

    def export_pdf(self, svg_path: str, save_path: str):
        export_pdf(
            svg_path,
            self.legend_img,
            save_path
        )

    def run_all(self):
        """Удобный метод, чтобы прогнать весь пайплайн разом."""
        self.preprocessing()
        self.quantizing()
        self.postprocessing()
        self.generate_contours()
        self.add_numbers()
