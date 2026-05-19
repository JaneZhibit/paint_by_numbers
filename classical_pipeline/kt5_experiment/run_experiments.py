"""
Запуск экспериментов КТ5: влияние сглаживания и числа цветов на метрики пайплайна.
Запускать из корня проекта: python kt5_experiment/run_experiments.py
"""
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import warnings

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from classical_pipeline.pipeline import PaintPipeline  # noqa: E402
from classical_pipeline.core.metrics import get_edge_preservation, get_image_complexity  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "exp_results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")
TEST_IMAGES_DIR = os.path.join(RESULTS_DIR, "test_images")

TEST_IMAGES = {
    "portrait": (
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/"
            "Portrait_of_a_woman%2C_by_John_Singer_Sargent.jpg/480px-"
            "Portrait_of_a_woman%2C_by_John_Singer_Sargent.jpg",
        ],
        "portrait.jpg",
        "portret_1.jpg",
    ),
    "landscape": (
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
            "Fronalpstock_big.jpg/800px-Fronalpstock_big.jpg",
        ],
        "landscape.jpg",
        "landscape_1.jpg",
    ),
    "still_life": (
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/"
            "Culinary_fruits_front_view.jpg/640px-Culinary_fruits_front_view.jpg",
        ],
        "still_life.jpg",
        "naturmort_1.jpg",
    ),
}

LOCAL_TEST_IMAGES_ROOT = os.path.join(PROJECT_ROOT, "test_images")

SMOOTHING_METHODS = ("mean_shift", "bilateral", "none")
COLORS_EXP1 = 16
COLORS_EXP2 = (8, 12, 16, 20, 24)

SAMPLE_STAGES = (
    "preprocessed",
    "quantized",
    "postprocessed",
    "vectorized",
    "color_reference",
    "numbered",
)


def _ensure_dirs():
    for d in (RAW_DIR, SAMPLES_DIR, TEST_IMAGES_DIR):
        os.makedirs(d, exist_ok=True)


def _find_local_fallback(local_name: str) -> str | None:
    if not os.path.isdir(LOCAL_TEST_IMAGES_ROOT):
        return None
    for root, _, files in os.walk(LOCAL_TEST_IMAGES_ROOT):
        if local_name in files:
            return os.path.join(root, local_name)
    return None


def _download_url(url: str, dest: str, retries: int = 3) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "kt5_experiment/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  попытка {attempt}/{retries} не удалась: {e}")
    return False


def download_test_images() -> dict[str, str]:
    """Скачивает тестовые изображения, если их ещё нет. Возвращает {type: path}."""
    paths = {}
    for image_type, (urls, filename, local_fallback) in TEST_IMAGES.items():
        dest = os.path.join(TEST_IMAGES_DIR, filename)
        if os.path.isfile(dest):
            print(f"Изображение {image_type} уже есть: {dest}")
            paths[image_type] = dest
            continue

        downloaded = False
        for url in urls:
            print(f"Скачивание {image_type}: {url}")
            if _download_url(url, dest):
                downloaded = True
                break

        if not downloaded:
            src = _find_local_fallback(local_fallback)
            if src:
                print(f"Сеть недоступна — копирование локального файла: {src}")
                shutil.copy2(src, dest)
                downloaded = True
            else:
                raise FileNotFoundError(
                    f"Не удалось получить изображение {image_type}. "
                    f"Проверьте сеть или наличие {local_fallback} в test_images/"
                )

        paths[image_type] = dest
    return paths


def build_config(smoothing_method: str, colors_count: int) -> dict:
    return {
        "quantization": {
            "colors_count": colors_count,
            "auto_colors": False,
        },
        "preprocessing": {
            "smoothing": {
                "method": smoothing_method,
                "params": {
                    "mean_shift": {
                        "spatial_radius": 15,
                        "color_radius": 30,
                    },
                    "bilateral": {
                        "d": 9,
                        "sigma_color": 75,
                        "sigma_space": 75,
                    },
                },
            },
        },
    }


def compute_zones_small_pct(high_res_labels: np.ndarray, config: dict) -> float:
    scale_factor = config.get("vectorization", {}).get("scale_factor", 4)
    min_area = 50 * (scale_factor ** 2)

    total = 0
    small = 0
    for cls in np.unique(high_res_labels):
        color_mask = np.uint8(high_res_labels == cls)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        for i in range(1, num_labels):
            total += 1
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                small += 1
    return 100.0 * small / total if total > 0 else 0.0


def _timing(pipeline: PaintPipeline, key: str) -> float:
    return float(pipeline.timings.get(key, 0.0))


def run_single(
    image_path: str,
    image_type: str,
    smoothing_method: str,
    colors_count: int,
    save_samples: bool = False,
) -> dict:
    config = build_config(smoothing_method, colors_count)
    pipeline = PaintPipeline(user_config=config)

    pipeline.load_image(image_path)
    image_complexity = float(get_image_complexity(pipeline.original_image))

    pipeline.preprocess()
    edge_preservation = float(
        get_edge_preservation(pipeline.original_image, pipeline.preprocessed_image)
    )

    pipeline.quantize()
    pipeline.postprocess()
    pipeline.vectorize()
    pipeline.render_all()

    zones_small_pct = compute_zones_small_pct(pipeline.high_res_labels, pipeline.config)

    t_pre = _timing(pipeline, "Preprocessing")
    t_quant = _timing(pipeline, "Quantization")
    t_post = _timing(pipeline, "Postprocessing")
    t_vec = _timing(pipeline, "Vectorization")
    t_render = _timing(pipeline, "Rendering")
    t_total = t_pre + t_quant + t_post + t_vec + t_render

    if save_samples:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        for stage in SAMPLE_STAGES:
            out_path = os.path.join(SAMPLES_DIR, f"portrait_ms16_{stage}.png")
            pipeline.show_stage(stage, save_path=out_path)

    return {
        "image_type": image_type,
        "smoothing_method": smoothing_method,
        "colors_count": colors_count,
        "time_preprocessing_s": round(t_pre, 4),
        "time_quantization_s": round(t_quant, 4),
        "time_postprocessing_s": round(t_post, 4),
        "time_vectorization_s": round(t_vec, 4),
        "time_rendering_s": round(t_render, 4),
        "time_total_s": round(t_total, 4),
        "zones_total": len(pipeline.labels_placement),
        "zones_small_pct": round(zones_small_pct, 2),
        "palette_size": len(pipeline.palette),
        "edge_preservation": round(edge_preservation, 4),
        "image_complexity": round(image_complexity, 2),
    }


def _json_name(image_type: str, smoothing_method: str, colors_count: int) -> str:
    return f"{image_type}_{smoothing_method}_{colors_count}.json"


def _save_result(data: dict, image_type: str, smoothing_method: str, colors_count: int):
    path = os.path.join(RAW_DIR, _json_name(image_type, smoothing_method, colors_count))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _run_and_record(
    image_paths: dict[str, str],
    image_type: str,
    smoothing_method: str,
    colors_count: int,
    save_samples: bool,
    stats: dict,
):
    name = _json_name(image_type, smoothing_method, colors_count)
    print(f"  Прогон: {name}")
    try:
        data = run_single(
            image_paths[image_type],
            image_type,
            smoothing_method,
            colors_count,
            save_samples=save_samples,
        )
        _save_result(data, image_type, smoothing_method, colors_count)
        stats["success"] += 1
    except Exception as e:
        warnings.warn(f"Ошибка в прогоне {name}: {e}")
        _save_result(
            {
                "error": str(e),
                "image_type": image_type,
                "smoothing_method": smoothing_method,
                "colors_count": colors_count,
            },
            image_type,
            smoothing_method,
            colors_count,
        )
    stats["total"] += 1


def main():
    _ensure_dirs()
    t0 = time.perf_counter()
    image_paths = download_test_images()

    stats = {"success": 0, "total": 0}

    print("\n=== Эксперимент 1: метод сглаживания (colors_count=16) ===")
    for smoothing in SMOOTHING_METHODS:
        for image_type in TEST_IMAGES:
            save_samples = (
                image_type == "portrait"
                and smoothing == "mean_shift"
                and COLORS_EXP1 == 16
            )
            _run_and_record(
                image_paths,
                image_type,
                smoothing,
                COLORS_EXP1,
                save_samples,
                stats,
            )

    print("\n=== Эксперимент 2: число цветов (smoothing=mean_shift) ===")
    for colors in COLORS_EXP2:
        for image_type in TEST_IMAGES:
            _run_and_record(
                image_paths,
                image_type,
                "mean_shift",
                colors,
                False,
                stats,
            )

    elapsed = time.perf_counter() - t0
    print(
        f"\nЭксперименты завершены. Успешных прогонов: {stats['success']} из {stats['total']}."
        f" (время: {elapsed:.1f} с)"
    )


if __name__ == "__main__":
    main()
