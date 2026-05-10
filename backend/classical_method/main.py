from pipeline import ClassicalPaintByNumbers
from utils.comparison import save_stages_comparison

# Параметры для управления алгоритмами при тестировании
test_features = {
    'preprocessing': {
        'saliency_map_enabled': False,
    },
    'quantizing': {
        'palette_method': 'kmeans',  # 'kmeans', 'median_cut', 'octree'
        'spatial_method': 'none',    # 'none', 'slic', 'watershed'
        'hierarchical_split_enabled': False,  # Включить иерархическое расщепление
        'weighted_kmeans': True,
        'superpixels': True,
        'auto_color': True,
        'watershed_enabled': False,  # Включить Watershed вместо SLIC
    },
    'postprocessing': {
        'grow_details': True,
    }
}

path = r"C:\Users\Andrey\Downloads\мы.jpg"
config = {
    'img_path': path,
    'target_max_side': 1000,
    'canvas_width_mm': 400,
    'canvas_height_mm': 300,
    'min_diameter_mm': 3,
    'colours_cnt': 'auto',
    'logging': True,
    'algorithm': {
        'preprocessing': {
            'filter_type': 'bilateral',
            'median': {'kernel_size': 5},
            'pyrMeanShift': {'sp': 5, 'sr': 15},
            'bilateral': {'d': 9, 'sigmaColor': 35, 'sigmaSpace': 135},
            'saliency_map': {'enabled': test_features['preprocessing']['saliency_map_enabled']}
        },
        'quantizing': {
            'palette_method': test_features['quantizing']['palette_method'],
            'spatial_method': 'watershed' if test_features['quantizing']['watershed_enabled'] else 'slic' if test_features['quantizing']['superpixels'] else 'none',
            'weighted_kmeans': {'enabled': test_features['quantizing']['weighted_kmeans']},
            'superpixels': {'enabled': test_features['quantizing']['superpixels'], 'region_size': 10, 'ruler': 20.0},
            'auto_color': {'enabled': test_features['quantizing']['auto_color']},
            'hierarchical_split': {'enabled': test_features['quantizing']['hierarchical_split_enabled'], 'min_area_pixels': 2000, 'variance_threshold': 30.0}
        },
        'postprocessing': {'contrast_threshold': 30, 'grow_details': {'enabled': test_features['postprocessing']['grow_details']}}
    }
}

gen = ClassicalPaintByNumbers(config)
gen.run_all

save_path = '../../frontend/examples/preprocessing/filters/bilateral/4.png'
save_stages_comparison(gen, ['quantizing', 'postprocessing', 'vectorization'], save_path=save_path, title="")
print(gen.timings)

print("Pipeline completed successfully.")

