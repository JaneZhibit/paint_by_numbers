import os
import json
from datetime import datetime
from PIL import Image
from copy import deepcopy


class HistoryManager:
    """Менеджер для управления историей генераций на диске."""
    
    def __init__(self):
        """Инициализирует менеджер истории, создаёт необходимые папки и загружает историю."""
        # Определяем пути
        self.history_dir = os.path.join(os.path.dirname(__file__), '..', 'workspace', 'history')
        self.thumbnails_dir = os.path.join(self.history_dir, 'thumbnails')
        self.history_file = os.path.join(self.history_dir, 'history.json')
        
        # Создаём папки, если их нет
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        
        # Загружаем историю из файла
        self.history = self._load_history()
    
    def _load_history(self) -> list:
        """Загружает историю из файла history.json."""
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history if isinstance(history, list) else []
        except (json.JSONDecodeError, IOError):
            return []
    
    def _save_history(self) -> None:
        """Сохраняет историю в файл history.json."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Ошибка при сохранении истории: {e}")
    
    def add_entry(self, config: dict, image_array, timings: dict, preset_name: str) -> None:
        """
        Добавляет новую запись в историю.
        
        Args:
            config: Конфигурация алгоритма
            image_array: Массив изображения (numpy array или PIL Image)
            timings: Словарь с временем выполнения этапов
            preset_name: Название использованного пресета
        """
        # Генерируем уникальное имя файла на основе timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Включаем миллисекунды
        
        # Сохраняем миниатюру в формате WebP
        img_filename = f"{timestamp}.webp"
        img_path = os.path.join(self.thumbnails_dir, img_filename)
        
        try:
            # Преобразуем в PIL Image, если это numpy array
            if not isinstance(image_array, Image.Image):
                image_array = Image.fromarray(image_array)
            
            # Сохраняем в WebP формате
            image_array.save(img_path, format='WEBP')
        except Exception as e:
            print(f"Ошибка при сохранении миниатюры: {e}")
            return
        
        # Создаём запись истории
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'preset': preset_name,
            'config': deepcopy(config),
            'timings': timings,
            'img_path': img_path
        }
        
        # Добавляем в начало списка (последние обработки первыми)
        self.history.insert(0, entry)
        
        # Ограничиваем историю до 20 записей
        if len(self.history) > 20:
            # Удаляем старые записи и их миниатюры
            for old_entry in self.history[20:]:
                try:
                    if os.path.exists(old_entry['img_path']):
                        os.remove(old_entry['img_path'])
                except Exception as e:
                    print(f"Ошибка при удалении старой миниатюры: {e}")
            
            self.history = self.history[:20]
        
        # Сохраняем историю в файл
        self._save_history()
    
    def get_history(self) -> list:
        """Возвращает список всех записей истории."""
        return self.history
    
    def clear_history(self) -> None:
        """Очищает историю и удаляет все миниатюры."""
        # Удаляем все миниатюры
        for entry in self.history:
            try:
                if os.path.exists(entry['img_path']):
                    os.remove(entry['img_path'])
            except Exception as e:
                print(f"Ошибка при удалении миниатюры: {e}")
        
        # Очищаем историю
        self.history = []
        self._save_history()
