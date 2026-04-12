from pipeline import ClassicalPaintByNumbers
from utils.comparison import save_stages_comparison

# --- Эксперимент 1: Классический метод (Сглаживание pyrMeanShift) ---
config_v1 = {
    'img_path': 'test_images/anime.webp',
    'target_max_side': 1000,
    'canvas_width_mm': 400,
    'canvas_height_mm': 300,
    'min_diameter_mm': 3,
    'colours_cnt': 16,
    'logging': False,
    'algorithm': {
        'preprocessing': {
            'filter_type': 'pyrMeanShift', 
            'pyrMeanShift': {'sp': 5, 'sr': 15}
        }
    }
}

config_v2 = {
    'img_path': 'test_images/wolf.webp',
    'target_max_side': 1000,
    'canvas_width_mm': 400,
    'canvas_height_mm': 300,
    'min_diameter_mm': 3,
    'colours_cnt': 16,
    'logging': False,
    'algorithm': {
        'preprocessing': {
            'filter_type': 'median'
        }
    }
}

gen1 = ClassicalPaintByNumbers(config_v1)
gen1.run_all()
gen1.generate_legend("output/legend_1.png")
save_stages_comparison(
    gen1, 
    stages=["preprocessing", "postprocessing", "contours", "numbering", "legend"], 
    title="Опыт 1",
    save_path="output/experiment_3.png"
)
gen1.export_svg("output/experiment_1.svg")
gen1.export_pdf("output/experiment_1.svg", "output/experiment_1.pdf")


print("Все сравнения сохранены в папку 'output/'")
