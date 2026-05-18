"""
Построение графиков по результатам экспериментов КТ5.
Запускать из корня проекта: python kt5_experiment/analyze.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "exp_results", "raw")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "exp_results", "plots")

IMAGE_TYPES = ("portrait", "landscape", "still_life")
IMAGE_LABELS = {
    "portrait": "Портрет",
    "landscape": "Пейзаж",
    "still_life": "Натюрморт",
}
SMOOTHING_ORDER = ("none", "bilateral", "mean_shift")
SMOOTHING_LABELS = {
    "none": "none",
    "bilateral": "bilateral",
    "mean_shift": "mean_shift",
}
COLORS_ORDER = (8, 12, 16, 20, 24)
COLORS_EXP1 = 16


def _set_style():
    for style in ("seaborn-v0_8-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def load_results() -> list[dict]:
    records = []
    if not os.path.isdir(RAW_DIR):
        print(f"Каталог {RAW_DIR} не найден. Сначала запустите run_experiments.py")
        return records
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        if "error" not in data:
            records.append(data)
    return records


def _filter_exp1(records: list[dict]) -> list[dict]:
    return [
        r
        for r in records
        if r.get("colors_count") == COLORS_EXP1 and r.get("smoothing_method") in SMOOTHING_ORDER
    ]


def _filter_exp2(records: list[dict]) -> list[dict]:
    return [
        r
        for r in records
        if r.get("smoothing_method") == "mean_shift" and r.get("colors_count") in COLORS_ORDER
    ]


def _grouped_bar(
    records: list[dict],
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    ylim=None,
):
    n_groups = len(SMOOTHING_ORDER)
    n_bars = len(IMAGE_TYPES)
    width = 0.25
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, img_type in enumerate(IMAGE_TYPES):
        vals = []
        for sm in SMOOTHING_ORDER:
            match = [
                r[metric]
                for r in records
                if r["image_type"] == img_type and r["smoothing_method"] == sm
            ]
            vals.append(match[0] if match else 0.0)
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=IMAGE_LABELS[img_type])

    ax.set_xticks(x)
    ax.set_xticklabels([SMOOTHING_LABELS[s] for s in SMOOTHING_ORDER], fontsize=11)
    ax.set_xlabel("Метод сглаживания", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {out}")


def plot_colors_zones(records: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for img_type in IMAGE_TYPES:
        xs, ys = [], []
        for c in COLORS_ORDER:
            match = [
                r["zones_total"]
                for r in records
                if r["image_type"] == img_type and r["colors_count"] == c
            ]
            if match:
                xs.append(c)
                ys.append(match[0])
        ax.plot(xs, ys, marker="o", linewidth=2, label=IMAGE_LABELS[img_type])

    ax.set_xlabel("Число цветов палитры", fontsize=11)
    ax.set_ylabel("Число зон", fontsize=11)
    ax.set_title(
        "Рис. 3. Зависимость числа зон от количества цветов палитры",
        fontsize=13,
    )
    ax.set_xticks(list(COLORS_ORDER))
    ax.legend(fontsize=10)
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "plot3_colors_zones.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Сохранён: {out}")


def main():
    _set_style()
    records = load_results()
    if not records:
        return

    exp1 = _filter_exp1(records)
    exp2 = _filter_exp2(records)

    if exp1:
        _grouped_bar(
            exp1,
            "time_total_s",
            "Рис. 1. Суммарное время обработки в зависимости от метода сглаживания",
            "Время обработки, с",
            "plot1_smoothing_time.png",
        )
        _grouped_bar(
            exp1,
            "zones_small_pct",
            "Рис. 2. Доля зон малой площади (< мин. кисть²) по методу сглаживания",
            "Доля малых зон, %",
            "plot2_small_zones.png",
        )
        _grouped_bar(
            exp1,
            "edge_preservation",
            "Рис. 4. Коэффициент сохранности границ по методу сглаживания",
            "Сохранность границ",
            "plot4_edge_preservation.png",
            ylim=(0, 1.05),
        )
    else:
        print("Нет данных для эксперимента 1 (colors_count=16)")

    if exp2:
        plot_colors_zones(exp2)
    else:
        print("Нет данных для эксперимента 2 (mean_shift)")

    print(f"\nГотово. Графики в {PLOTS_DIR}")


if __name__ == "__main__":
    main()
