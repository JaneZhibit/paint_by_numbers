import sys
import os
import json
from copy import deepcopy

# Добавляем путь к backend для импорта функций конфигурации
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'classical_method'))

from core.config import update_dict_deep

# Путь к файлу с пользовательскими пресетами
CUSTOM_PRESETS_FILE = os.path.join(os.path.dirname(__file__), 'custom_presets.json')


# ============================================================================
# ПРЕСЕТЫ НАСТРОЕК
# ============================================================================

PRESETS = {
    "🎭 Портрет (Фокус на лице)": {
        "preprocessing": {
            "saliency_map": {"enabled": True}
        },
        "quantizing": {
            "palette_method": "median_cut",
            "spatial_method": "none",
            "weighted_kmeans": {"enabled": True}
        },
        "postprocessing": {
            "grow_details": {"enabled": True}
        }
    },
    
    "🌿 Художественные мазки (Пейзаж)": {
        "preprocessing": {
            "filter_type": "bilateral"
        },
        "quantizing": {
            "palette_method": "kmeans",
            "spatial_method": "slic",
            "superpixels": {
                "enabled": True,
                "region_size": 10,
                "ruler": 15.0
            },
            "auto_color": {"enabled": True},
            "hierarchical_split": {"enabled": False}
        }
    },
    
    "✏️ Векторный стиль / Аниме": {
        "preprocessing": {
            "filter_type": "pyrMeanShift"
        },
        "quantizing": {
            "palette_method": "kmeans",
            "spatial_method": "watershed",
            "hierarchical_split": {"enabled": True},
            "saliency_map": {"enabled": False}
        }
    }
}


def apply_preset(preset_name: str, base_config: dict) -> dict:
    """
    Применяет пресет к базовой конфигурации.
    
    Args:
        preset_name: Название пресета (ключ из PRESETS или custom_presets)
        base_config: Базовая конфигурация (обычно из get_default_algo_config())
    
    Returns:
        Новая конфигурация с применённым пресетом
    
    Raises:
        ValueError: Если пресет не найден
    """
    # Объединяем встроенные и пользовательские пресеты
    all_presets = {**PRESETS, **load_custom_presets()}
    
    if preset_name not in all_presets:
        available = ", ".join(all_presets.keys())
        raise ValueError(f"Пресет '{preset_name}' не найден. Доступные: {available}")
    
    # Делаем глубокую копию базовой конфигурации
    result_config = deepcopy(base_config)
    
    # Получаем пресет
    preset = all_presets[preset_name]
    
    # Рекурсивно обновляем конфигурацию
    result_config = update_dict_deep(result_config, preset)
    
    return result_config


def get_preset_names() -> list:
    """Возвращает список всех доступных пресетов (встроенные + пользовательские)."""
    all_presets = {**PRESETS, **load_custom_presets()}
    return list(all_presets.keys())


def get_preset_description(preset_name: str) -> str:
    """Возвращает описание пресета (извлекается из названия)."""
    if preset_name not in PRESETS:
        return "Неизвестный пресет"
    
    # Описание — это часть названия после эмодзи
    parts = preset_name.split(" ", 1)
    if len(parts) > 1:
        return parts[1]
    return preset_name


def load_custom_presets() -> dict:
    """
    Загружает пользовательские пресеты из файла custom_presets.json.
    
    Returns:
        Словарь пользовательских пресетов (пустой, если файл не существует)
    """
    if not os.path.exists(CUSTOM_PRESETS_FILE):
        return {}
    
    try:
        with open(CUSTOM_PRESETS_FILE, 'r', encoding='utf-8') as f:
            custom_presets = json.load(f)
        return custom_presets if isinstance(custom_presets, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}


def save_custom_preset(preset_name: str, config: dict) -> None:
    """
    Сохраняет пользовательский пресет в файл custom_presets.json.
    
    Args:
        preset_name: Название пресета
        config: Конфигурация алгоритма (словарь)
    """
    # Загружаем существующие пресеты
    custom_presets = load_custom_presets()
    
    # Добавляем новый пресет
    custom_presets[preset_name] = config
    
    # Сохраняем в файл
    try:
        with open(CUSTOM_PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(custom_presets, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Ошибка при сохранении пресета: {e}")


def delete_custom_preset(preset_name: str) -> bool:
    """
    Удаляет пользовательский пресет из файла.
    
    Args:
        preset_name: Название пресета
    
    Returns:
        True, если пресет был удалён, False иначе
    """
    custom_presets = load_custom_presets()
    
    if preset_name not in custom_presets:
        return False
    
    del custom_presets[preset_name]
    
    try:
        with open(CUSTOM_PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(custom_presets, f, indent=2, ensure_ascii=False)
        return True
    except IOError:
        return False
