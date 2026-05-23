from pipeline import MLPaintPipeline

imgs = [
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
]


# Настройки
custom_config = {
    "general": {
        "target_max_side_px": 1000  # Целевой размер
    }
}

pipeline = MLPaintPipeline(user_config=custom_config)

for img in [r"D:\projects\paint_by_numbers\ml_pipeline\test_images/ivan.jpg"]:
    test_image_path = img

    pipeline.load_image(test_image_path)
    pipeline.preprocess()

    # Запускаем нейросеть!
    pipeline.segment()

    # Смотрим, как она отделила кошку/человека от заднего фона
    pipeline.quantize()
    pipeline.postprocess()
    pipeline.vectorize()
    pipeline.render_all()

    pipeline.debug_show_segmentation(f"results/mask_{img.split('/')[-1]}")
    pipeline.show_stage("quantized", save_path=f"results/quant_{img.split('/')[-1]}")
    pipeline.show_stage("postprocessed", save_path=f"results/post{img.split('/')[-1]}")
    pipeline.show_stage("vectorized", save_path=f"results/vect{img.split('/')[-1]}")
    pipeline.show_stage("color_reference", save_path=f"results/ref{img.split('/')[-1]}")


