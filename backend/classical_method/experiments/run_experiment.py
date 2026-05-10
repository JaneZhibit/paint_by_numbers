import os
import sys
import pandas as pd
from tqdm import tqdm

# Добавляем путь к backend/classical_method в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import ClassicalPaintByNumbers

# Заглушка путей к тестовым изображениям.
# Замените на реальные пути к картинкам для тестирования.
IMAGE_PATHS = [
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\человек.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\дом.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\дорога.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\елки.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\лошадь.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\люди.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\магазин.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\нг.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\парус.jpg",
    r"C:\Users\Andrey\Desktop\картины\картины 4 к 3\тропинка.jpg"
]

def run_experiments():
    if not IMAGE_PATHS:
        print("Внимание: Список IMAGE_PATHS пуст. Добавьте пути к изображениям для тестирования.")
        return

    results = []
    
    # Создаем директорию для сохранения результатов
    history_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'workspace', 'history'))
    os.makedirs(history_dir, exist_ok=True)
    results_file = os.path.join(history_dir, 'experiment_results.csv')

    # Базовый конфиг
    base_config = {
        'target_max_side': 1000,
        'canvas_width_mm': 400,
        'canvas_height_mm': 300,
        'min_diameter_mm': 3.0,
        'logging': False,
        'algorithm': {
            'preprocessing': {
                'blur_type': 'pyrMeanShift',
                'pyrMeanShift': {'sp': 5, 'sr': 15},
                'saliency_map': {'enabled': False}
            },
            'quantizing': {
                'color_space': 'lab',
                'method': 'kmeans',
                'weighted_kmeans': {'enabled': True, 'edge_threshold1': 50, 'edge_threshold2': 150, 'weight_factor': 5},
                'auto_color': {'enabled': False}
            },
            'postprocessing': {
                'connectivity': 4,
                'contrast_threshold': 100.0,
                'grow_details': {'enabled': False}
            }
        }
    }

    # Проходим по всем картинкам
    for img_path in tqdm(IMAGE_PATHS, desc="Изображения"):
        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")
            continue
            
        img_name = os.path.basename(img_path)
        
        # Проходим по количеству цветов от 4 до 24 с шагом 2
        for colours_cnt in tqdm(range(4, 25, 2), desc=f"Цвета для {img_name}", leave=False):
            config = base_config.copy()
            config['img_path'] = img_path
            config['colours_cnt'] = colours_cnt
            
            try:
                gen = ClassicalPaintByNumbers(config)
                gen.preprocessing()
                gen.quantizing()
                gen.postprocessing()
                
                metrics = gen.complexity_metrics
                if metrics:
                    row = {
                        'image_name': img_name,
                        'colours_cnt': colours_cnt,
                    }
                    row.update(metrics)
                    results.append(row)
                    
            except Exception as e:
                print(f"Ошибка при обработке {img_name} с {colours_cnt} цветами: {e}")

    # Сохраняем результаты в CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        print(f"\nЭксперимент завершен. Результаты сохранены в: {results_file}")
    else:
        print("\nНет результатов для сохранения.")

if __name__ == "__main__":
    run_experiments()
