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

config = {
    'img_path': 'test_images/wolf.webp',
    'target_max_side': 1000,
    'canvas_width_mm': 600,
    'canvas_height_mm': 300,
    'min_diameter_mm': 3,
    'colours_cnt': 'auto',
    'logging': True,
    'algorithm': {
        'preprocessing': {
            'filter_type': 'none',
            'pyrMeanShift': {'sp': 5, 'sr': 15},
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
gen.run_all()
save_path = 'output/comparison_10.png'
save_stages_comparison(gen, ['original', 'preprocessing', 'quantizing', 'postprocessing'], save_path=save_path)

print("Pipeline completed successfully.")

