from ml_pipeline.pipeline import MLPaintPipeline
import gc


def run_comparison():
    test_image_path = "../classical_pipeline/test_images/масштаб 4 к 3/landscape_2.jpg"  # Укажи путь к сложному фото

    # Список моделей, которые мы хотим протестировать
    models_to_test = [
        "yolo11x-seg.pt",
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "nvidia/segformer-b5-finetuned-ade-640-640",
        "openmmlab/upernet-convnext-tiny",
        "openmmlab/upernet-convnext-base",
        "openmmlab/upernet-convnext-large"

    ]

    for model_name in models_to_test:
        print(f"\n{'=' * 50}\nТестируем модель: {model_name}\n{'=' * 50}")

        # Подменяем конфиг "на лету"
        custom_config = {
            "ml_models": {
                "segmentation": model_name
            }
        }

        pipeline = MLPaintPipeline(user_config=custom_config)
        pipeline.load_image(test_image_path)
        pipeline.preprocess()

        try:
            pipeline.segment()

            # Формируем имя файла (заменяем слеши на подчеркивания)
            safe_name = model_name.replace("/", "_").replace(".pt", "")
            out_path = f"results/compare_{safe_name}.png"

            pipeline.debug_show_segmentation(output_path=out_path)

        except Exception as e:
            print(f"Ошибка при тестировании {model_name}: {e}")

        # Принудительно чистим память Raspberry Pi перед загрузкой следующей модели
        del pipeline
        gc.collect()


if __name__ == "__main__":
    run_comparison()