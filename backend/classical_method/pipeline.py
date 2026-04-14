import diskcache as dc
import hashlib
import json
import os

from core.config import validate_and_prepare_config
from core.preprocessing import preprocess_image
from core.quantizing import quantize_image
from core.postprocessing import postprocess_image
from core.auto_color import find_optimal_k
from utils.timer import time_stage, print_timing_report

# Инициализируем глобальный кэш (лимит 1 ГБ)
cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'workspace', 'cache')
os.makedirs(cache_dir, exist_ok=True)
cache = dc.Cache(cache_dir, size_limit=10**9)


class ClassicalPaintByNumbers:
    def __init__(self, config):
        self.config = validate_and_prepare_config(config)

        self.original_img = None
        self.preprocessed_img = None
        self.saliency_map = None
        self.quant_rgb = None
        self.quant_space = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.postprocessed_img = None
        self.final_colors = None
        self.timings = {}
        
        # Хеши для кэширования этапов
        self.preprocessing_hash = None
        self.quantizing_hash = None
        self.postprocessing_hash = None

    def _get_hash(self, data_dict):
        """Генерирует хеш для данных кэширования."""
        json_str = json.dumps(data_dict, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _get_preprocessing_hash(self):
        """Хеш для этапа preprocessing."""
        data = {
            'img_path': self.config['img_path'],
            'target_max_side': self.config['target_max_side'],
            'preprocessing': self.config['algorithm']['preprocessing']
        }
        return self._get_hash(data)
    
    def _get_quantizing_hash(self):
        """Хеш для этапа quantizing."""
        data = {
            'preprocessing_hash': self.preprocessing_hash,
            'quantizing': self.config['algorithm']['quantizing'],
            'colours_cnt': self.config['colours_cnt']
        }
        return self._get_hash(data)
    
    def _get_postprocessing_hash(self):
        """Хеш для этапа postprocessing."""
        data = {
            'quantizing_hash': self.quantizing_hash,
            'postprocessing': self.config['algorithm']['postprocessing']
        }
        return self._get_hash(data)

    @time_stage("preprocessing")
    def preprocessing(self):
        # Вычисляем хеш для этапа
        self.preprocessing_hash = self._get_preprocessing_hash()
        
        # Проверяем кэш
        if self.preprocessing_hash in cache:
            print(f"✓ Использован кэш для этапа preprocessing")
            self.original_img, self.preprocessed_img, self.saliency_map = cache[self.preprocessing_hash]
        else:
            # Выполняем вычисления
            self.original_img, self.preprocessed_img, self.saliency_map = preprocess_image(self.config)
            # Сохраняем в кэш
            cache[self.preprocessing_hash] = (self.original_img, self.preprocessed_img, self.saliency_map)

    @time_stage("quantizing")
    def quantizing(self):
        # Проверка: нужен ли автоматический подбор количества цветов
        auto_color_enabled = self.config['algorithm']['quantizing']['auto_color'].get('enabled', False)
        colours_cnt_is_auto = self.config.get('colours_cnt') == 'auto'
        
        if auto_color_enabled or colours_cnt_is_auto:
            # Находим оптимальное количество цветов
            optimal_k = find_optimal_k(self.preprocessed_img, self.config)
            self.config['colours_cnt'] = optimal_k
        
        # Вычисляем хеш для этапа
        self.quantizing_hash = self._get_quantizing_hash()
        
        # Проверяем кэш
        if self.quantizing_hash in cache:
            print(f"✓ Использован кэш для этапа quantizing")
            self.quant_rgb, self.quant_space, self.cluster_centers, self.cluster_labels = cache[self.quantizing_hash]
        else:
            # Выполняем вычисления
            res = quantize_image(self.preprocessed_img, self.config)
            self.quant_rgb, self.quant_space, self.cluster_centers, self.cluster_labels = res
            # Сохраняем в кэш
            cache[self.quantizing_hash] = (self.quant_rgb, self.quant_space, self.cluster_centers, self.cluster_labels)

    @time_stage("postprocessing")
    def postprocessing(self):
        # Вычисляем хеш для этапа
        self.postprocessing_hash = self._get_postprocessing_hash()
        
        # Проверяем кэш
        if self.postprocessing_hash in cache:
            print(f"✓ Использован кэш для этапа postprocessing")
            self.postprocessed_img, self.final_colors, self.cluster_labels = cache[self.postprocessing_hash]
        else:
            # Выполняем вычисления
            res = postprocess_image(self.quant_rgb, self.cluster_labels, self.cluster_centers, self.config, self.saliency_map)
            self.postprocessed_img, self.final_colors, self.cluster_labels = res
            # Сохраняем в кэш
            cache[self.postprocessing_hash] = (self.postprocessed_img, self.final_colors, self.cluster_labels)

    def run_all(self):
        """Удобный метод, чтобы прогнать весь пайплайн разом."""
        self.preprocessing()
        self.quantizing()
        self.postprocessing()
        print_timing_report(self.timings)
