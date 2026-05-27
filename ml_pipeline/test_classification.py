import os
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import time

# --- НАША ЭВРИСТИКА (ПРАВИЛА НА ОСНОВЕ YOLO/SEGFORMER) ---
def rule_based_classification(semantic_objects: list) -> str:
    """
    Классификация жанра на основе процентного соотношения объектов.
    semantic_objects - список словарей: [{'label': 'person', 'area_pct': 45.0, ...}]
    """
    person_area = sum(obj['area_pct'] for obj in semantic_objects if 'person' in obj['label'].lower())
    animal_area = sum(obj['area_pct'] for obj in semantic_objects if any(
        animal in obj['label'].lower() for animal in ['dog', 'cat', 'bird', 'bear', 'horse', 'animal']))
    nature_area = sum(obj['area_pct'] for obj in semantic_objects if any(
        nature in obj['label'].lower() for nature in ['sky', 'tree', 'grass', 'mountain', 'water', 'sea', 'plant']))

    # Логика из твоего промпта:
    if person_area > 40.0:
        return "Portrait (Крупный план)"
    elif 15.0 <= person_area <= 40.0:
        return "People in Scene (Средний план)"
    elif animal_area > 15.0:
        return "Animal Focus (Животные)"
    elif nature_area > 50.0 and person_area < 5.0:
        return "Landscape (Пейзаж)"
    elif nature_area > 30.0 and (person_area + animal_area) < 15.0:
        return "Landscape with small objects (Пейзаж с объектами)"
    else:
        return "Still Life / Complex Scene (Натюрморт или сложная сцена)"


# --- ТЕСТ ZERO-SHOT ML КЛАССИФИКАТОРА (CLIP) ---
def test_clip_classification(image_paths: list):
    print("\nЗагрузка модели CLIP (Zero-Shot Classification)...")
    # Используем компактный CLIP, который отлично работает на CPU (весит около 600 МБ)
    classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

    # Категории, которые мы хотим, чтобы нейросеть искала
    candidate_labels = [
        "portrait photography",
        "landscape scenery",
        "close-up animal photo",
        "still life painting",
        "cityscape architecture"
    ]

    print("\n" + "=" * 60)
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue

        print(f"\nАнализ изображения: {os.path.basename(img_path)}")
        pil_image = Image.open(img_path).convert("RGB")

        start_time = time.time()
        # Запускаем классификатор
        results = classifier(pil_image, candidate_labels=candidate_labels)
        ml_time = time.time() - start_time

        print(f"ML Классификатор (CLIP) отработал за {ml_time:.2f} сек:")
        # Выводим Топ-2 предсказания
        for i in range(3):
            print(f"  {i + 1}. {results[i]['label']} ({results[i]['score'] * 100:.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Укажи тут свои тестовые картинки разных жанров
    test_images = [
    "test_images/масштаб 4 к 3/animal_1.jpg",
    "test_images/масштаб 4 к 3/animal_2.jpg",
    "test_images/масштаб 4 к 3/animal_3.jpg",
    "test_images/масштаб 4 к 3/animal_4.jpg",
    "test_images/масштаб 4 к 3/animal_5.jpg",
    "test_images/масштаб 4 к 3/landscape_1.jpg",
    "test_images/масштаб 4 к 3/landscape_2.jpg",
    "test_images/масштаб 4 к 3/landscape_3.jpg",
    "test_images/масштаб 4 к 3/landscape_4.jpg",
    "test_images/масштаб 4 к 3/landscape_5.jpg",
    "test_images/масштаб 4 к 3/naturmort_1.jpg",
    "test_images/масштаб 4 к 3/naturmort_2.jpg",
    "test_images/масштаб 4 к 3/naturmort_3.jpg",
    "test_images/масштаб 4 к 3/naturmort_4.jpg",
    "test_images/масштаб 4 к 3/naturmort_5.jpg",
    "test_images/масштаб 4 к 3/portret_1.jpg",
    "test_images/масштаб 4 к 3/portret_2.jpg",
    "test_images/масштаб 4 к 3/portret_3.jpg",
    "test_images/масштаб 4 к 3/portret_4.jpg",
    "test_images/масштаб 4 к 3/portret_5.jpg",
    "test_images/масштаб 4 к 3/jane.jpg",
    "test_images/масштаб 4 к 3/andrey.jpg",
    "test_images/масштаб 4 к 3/andrey_2.jpg",
    "test_images/масштаб 4 к 3/andrey_dog.jpg",
    "test_images/масштаб 4 к 3/andrey_dog_2.jpg",
    "test_images/масштаб 4 к 3/pacific.jpg"


]

    # 1. Проверяем чистый ML (может потребоваться pip install transformers)
    test_clip_classification(test_images)

