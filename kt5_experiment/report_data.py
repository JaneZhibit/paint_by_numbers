"""
Агрегация результатов КТ5: таблицы и текст для отчёта.
Запускать после analyze.py: python kt5_experiment/report_data.py
"""
import csv
import json
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "exp_results", "raw")
TABLES_DIR = os.path.join(SCRIPT_DIR, "exp_results", "tables")

SMOOTHING_ORDER = ("none", "bilateral", "mean_shift")
COLORS_ORDER = (8, 12, 16, 20, 24)
COLORS_EXP1 = 16
IMAGE_TYPES = ("portrait", "landscape", "still_life")


def load_results() -> list[dict]:
    records = []
    if not os.path.isdir(RAW_DIR):
        return records
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        if "error" not in data:
            records.append(data)
    return records


def _avg(records: list[dict], key: str) -> float:
    vals = [r[key] for r in records if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_smoothing(records: list[dict]) -> dict[str, dict]:
    exp1 = [
        r
        for r in records
        if r.get("colors_count") == COLORS_EXP1 and r.get("smoothing_method") in SMOOTHING_ORDER
    ]
    out = {}
    for sm in SMOOTHING_ORDER:
        subset = [r for r in exp1 if r["smoothing_method"] == sm]
        if not subset:
            continue
        out[sm] = {
            "time_total_s": _avg(subset, "time_total_s"),
            "zones_total": _avg(subset, "zones_total"),
            "zones_small_pct": _avg(subset, "zones_small_pct"),
            "edge_preservation": _avg(subset, "edge_preservation"),
        }
    return out


def aggregate_colors(records: list[dict]) -> dict[int, dict]:
    exp2 = [
        r
        for r in records
        if r.get("smoothing_method") == "mean_shift" and r.get("colors_count") in COLORS_ORDER
    ]
    out = {}
    for c in COLORS_ORDER:
        subset = [r for r in exp2 if r["colors_count"] == c]
        if not subset:
            continue
        out[c] = {
            "time_quantization_s": _avg(subset, "time_quantization_s"),
            "zones_total": _avg(subset, "zones_total"),
            "zones_small_pct": _avg(subset, "zones_small_pct"),
            "edge_preservation": _avg(subset, "edge_preservation"),
        }
    return out


def print_table1(data: dict[str, dict]):
    print(
        "ТАБЛИЦА 1 — Сравнение методов сглаживания "
        "(среднее по 3 изображениям, colors_count=16)"
    )
    print("| Метод сглаживания | Время, с | Зон всего | Зон малых, % | Сохр. границ |")
    print("|-------------------|----------|-----------|--------------|--------------|")
    for sm in SMOOTHING_ORDER:
        if sm not in data:
            continue
        d = data[sm]
        print(
            f"| {sm:<17} | {d['time_total_s']:>7.2f} | "
            f"{d['zones_total']:>8.0f} | {d['zones_small_pct']:>10.1f}% | "
            f"{d['edge_preservation']:>12.2f} |"
        )
    print()


def print_table2(data: dict[int, dict]):
    print(
        "ТАБЛИЦА 2 — Влияние числа цветов палитры "
        "(среднее по 3 изображениям, smoothing=mean_shift)"
    )
    print("| Цветов | Время квант., с | Зон всего | Зон малых, % | Сохр. границ |")
    print("|--------|-----------------|-----------|--------------|--------------|")
    for c in COLORS_ORDER:
        if c not in data:
            continue
        d = data[c]
        print(
            f"| {c:>6} | {d['time_quantization_s']:>15.2f} | "
            f"{d['zones_total']:>8.0f} | {d['zones_small_pct']:>10.1f}% | "
            f"{d['edge_preservation']:>12.2f} |"
        )
    print()


def save_csv_table1(data: dict[str, dict]):
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, "table1_smoothing.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["smoothing_method", "time_s", "zones_total", "zones_small_pct", "edge_preservation"]
        )
        for sm in SMOOTHING_ORDER:
            if sm not in data:
                continue
            d = data[sm]
            w.writerow(
                [
                    sm,
                    f"{d['time_total_s']:.2f}",
                    f"{d['zones_total']:.0f}",
                    f"{d['zones_small_pct']:.1f}",
                    f"{d['edge_preservation']:.2f}",
                ]
            )
    print(f"CSV: {path}")


def save_csv_table2(data: dict[int, dict]):
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, "table2_colors.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["colors_count", "time_quant_s", "zones_total", "zones_small_pct", "edge_preservation"]
        )
        for c in COLORS_ORDER:
            if c not in data:
                continue
            d = data[c]
            w.writerow(
                [
                    c,
                    f"{d['time_quantization_s']:.2f}",
                    f"{d['zones_total']:.0f}",
                    f"{d['zones_small_pct']:.1f}",
                    f"{d['edge_preservation']:.2f}",
                ]
            )
    print(f"CSV: {path}")


def _best_smoothing(data: dict[str, dict], metric: str, higher_better: bool) -> str:
    items = [(sm, data[sm][metric]) for sm in data]
    if not items:
        return "—"
    return max(items, key=lambda x: x[1])[0] if higher_better else min(items, key=lambda x: x[1])[0]


def generate_conclusions(smooth: dict[str, dict], colors: dict[int, dict]) -> str:
    if not smooth or not colors:
        return (
            "Данные экспериментов отсутствуют или неполны; "
            "необходимо выполнить run_experiments.py."
        )

    fastest = _best_smoothing(smooth, "time_total_s", False)
    slowest = _best_smoothing(smooth, "time_total_s", True)
    t_none = smooth.get("none", {}).get("time_total_s", 0)
    t_ms = smooth.get("mean_shift", {}).get("time_total_s", 0)
    t_bi = smooth.get("bilateral", {}).get("time_total_s", 0)

    edge_none = smooth.get("none", {}).get("edge_preservation", 0)
    edge_ms = smooth.get("mean_shift", {}).get("edge_preservation", 0)
    edge_bi = smooth.get("bilateral", {}).get("edge_preservation", 0)
    best_edge = _best_smoothing(smooth, "edge_preservation", True)

    small_none = smooth.get("none", {}).get("zones_small_pct", 0)
    small_ms = smooth.get("mean_shift", {}).get("zones_small_pct", 0)

    z8 = colors.get(8, {}).get("zones_total", 0)
    z24 = colors.get(24, {}).get("zones_total", 0)
    q8 = colors.get(8, {}).get("time_quantization_s", 0)
    q24 = colors.get(24, {}).get("time_quantization_s", 0)

    paragraphs = [
        (
            f"В ходе промежуточных испытаний классического CV-пайплайна на трёх типах "
            f"тестовых изображений (портрет, пейзаж, натюрморт) было установлено, что "
            f"метод сглаживания существенно влияет на вычислительную стоимость и "
            f"структуру зон раскраски. Среднее суммарное время обработки при "
            f"отсутствии сглаживания (none) составило {t_none:.2f} с, при bilateral — "
            f"{t_bi:.2f} с, при mean shift — {t_ms:.2f} с; наиболее затратным оказался "
            f"метод «{slowest}», наименее — «{fastest}». Коэффициент сохранности границ "
            f"относительно исходника составил {edge_none:.2f} (none), {edge_bi:.2f} "
            f"(bilateral) и {edge_ms:.2f} (mean shift); максимальное значение "
            f"зафиксировано для «{best_edge}»."
        ),
        (
            f"Доля физически нераскрашиваемых зон (площадь менее минимального размера "
            f"кисти) при 16 цветах составила в среднем {small_none:.1f}% без сглаживания "
            f"и {small_ms:.1f}% при mean shift, что указывает на компромисс между "
            f"упрощением палитры и фрагментацией контуров. При варьировании числа "
            f"цветов (8–24, mean shift) среднее число зон возросло с {z8:.0f} до "
            f"{z24:.0f}, время квантования — с {q8:.2f} до {q24:.2f} с."
        ),
        (
            f"Результаты не подтверждают исследовательскую гипотезу о превосходстве "
            f"гибридного пайплайна с семантической сегментацией: на данном этапе "
            f"реализован только классический CV-конвейер без ML-модуля, метрика IoU "
            f"и субъективная оценка качества не измерялись. Полученные количественные "
            f"показатели задают базовую линию (baseline) для последующего сравнения "
            f"после интеграции сегментации. Ограничения текущего этапа: отсутствие "
            f"нейросетевой постобработки, единый набор гиперпараметров mean shift/bilateral "
            f"и отсутствие пользовательского A/B-тестирования субъективного качества."
        ),
    ]
    return "\n\n".join(paragraphs)


def main():
    records = load_results()
    if not records:
        print("Нет результатов. Запустите: python kt5_experiment/run_experiments.py")
        sys.exit(1)

    smooth = aggregate_smoothing(records)
    colors = aggregate_colors(records)

    print_table1(smooth)
    print_table2(colors)
    save_csv_table1(smooth)
    save_csv_table2(colors)

    text = generate_conclusions(smooth, colors)
    print('=== ТЕКСТ ДЛЯ ОТЧЁТА: РАЗДЕЛ "АНАЛИЗ ПРОМЕЖУТОЧНЫХ РЕЗУЛЬТАТОВ" ===\n')
    print(text)
    print("\n=== КОНЕЦ БЛОКА ===")


if __name__ == "__main__":
    main()
